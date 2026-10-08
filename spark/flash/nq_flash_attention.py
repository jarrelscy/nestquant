"""Adapt NoPE MLA to SM120's fixed 64-element rotary storage.

Both query and key padding are zero; the attention scale and 512 latent
components are unchanged. This only affects the SM120 sparse backend.
"""
def install():
    import torch
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import FlashInferMLASparseSM120Impl as Impl
    if getattr(Impl, '_nq_nope_padding', False):
        return
    init = Impl.__init__
    update = Impl.do_kv_cache_update
    forward = Impl.forward_mqa

    def patched_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self._nq_nope = self.qk_rope_head_dim == 0
        if self._nq_nope:
            if self.kv_lora_rank != 512:
                raise ValueError('NoPE SM120 adapter requires latent dimension 512')
            self.qk_rope_head_dim = 64

    def patched_update(self, kv_c_normed, k_pe, *args, **kwargs):
        if self._nq_nope:
            if k_pe.shape[-1] != 0:
                raise ValueError('Expected empty NoPE rotary key')
            k_pe = k_pe.new_zeros((*k_pe.shape[:-1], 64))
        return update(self, kv_c_normed, k_pe, *args, **kwargs)

    def patched_forward(self, q, kv_cache, attn_metadata, layer):
        if self._nq_nope:
            if isinstance(q, tuple):
                q = torch.cat(q, dim=-1)
            if q.shape[-1] != 512:
                raise ValueError('Expected 512-dimensional NoPE query')
            q = torch.nn.functional.pad(q, (0, 64))
        from vllm.v1.attention.backends.mla.sparse_utils import triton_convert_req_index_to_global_index
        from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import _get_workspace_buffer
        from vllm.utils.flashinfer import flashinfer_trtllm_batch_decode_with_kv_cache_mla
        n = q.shape[0]
        indices = self.topk_indices_buffer[:n]
        physical = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:n], attn_metadata.block_table,
            indices, BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=indices.shape[1])
        if indices.shape[1] not in (2048, 2176):
            raise ValueError('Unexpected Flash sparse window width')
        # Global physical indices are token offsets. Re-viewing contiguous pages
        # as 64-token pages preserves every address and enables decode dispatch.
        cache = kv_cache.view(torch.uint8).view(-1, 1, 64, 656)
        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)
        parts = []
        for start, end in ((0, 2048), (2048, indices.shape[1])):
            if start == end:
                continue
            segment = physical[:, start:end].contiguous()
            # The prefill binary instantiates 2048 candidates, not 128.
            if n > 64 and segment.shape[1] == 128:
                segment = torch.nn.functional.pad(segment, (0, 1920), value=-1)
            width = segment.shape[1]
            out, lse = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
                query=q.unsqueeze(1), kv_cache=cache,
                workspace_buffer=self._workspace_buffer,
                qk_nope_head_dim=self.qk_nope_head_dim,
                kv_lora_rank=512, qk_rope_head_dim=64,
                block_tables=segment.unsqueeze(1),
                seq_lens=None, max_seq_len=width,
                bmm1_scale=self.scale, bmm2_scale=1.0,
                sparse_mla_top_k=width, kv_scale_format=self.kv_scale_format,
                return_lse=True)
            parts.append((out.squeeze(1), lse.reshape(n, self.num_heads)))
        if len(parts) == 1:
            return parts[0][0], None
        # Merge two disjoint attention sets with their log normalizers, preserving
        # one softmax over all DSA+SWA entries rather than averaging attentions.
        logs = torch.stack([p[1] for p in parts], -1)
        weights = torch.softmax(logs * 0.6931471805599453, -1)  # FlashInfer returns log2 LSE
        result = sum(torch.nan_to_num(out.float()) * weights[..., i, None]
                     for i, (out, _) in enumerate(parts))
        return result.to(q.dtype), None

    Impl.__init__ = patched_init
    Impl.do_kv_cache_update = patched_update
    Impl.forward_mqa = patched_forward
    Impl._nq_nope_padding = True
