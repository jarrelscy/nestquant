"""Opt-in startup diagnostics and release of unused allocator cache; not enabled live."""
import ctypes
import functools
import gc
import json
import time


def patch(loader):
    import torch
    if getattr(loader, '_nq_loader_memory_diagnostic', False):
        return
    original = loader.process_weights_after_loading

    def snapshot(phase, model):
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        print(json.dumps({'nq_loader_memory': phase, 'time': time.time(),
                          'model_class': type(model).__name__,
                          'allocated_bytes': torch.cuda.memory_allocated(),
                          'reserved_bytes': torch.cuda.memory_reserved(),
                          'cuda_free_bytes': free, 'cuda_total_bytes': total}), flush=True)

    @functools.wraps(original)
    def process(model, *args, **kwargs):
        snapshot('before_weight_postprocessing', model)
        result = original(model, *args, **kwargs)
        snapshot('after_weight_postprocessing', model)
        # Live weights remain referenced; this releases only unused allocator blocks.
        gc.collect()
        torch.cuda.empty_cache()
        # glibc may retain freed CPU loading arenas; return unused pages only.
        try:
            trim = ctypes.CDLL(None).malloc_trim
            trim.argtypes = [ctypes.c_size_t]
            trim.restype = ctypes.c_int
            trim(0)
        except AttributeError:
            pass
        snapshot('after_unused_cache_release', model)
        return result

    loader.process_weights_after_loading = process
    loader._nq_loader_memory_diagnostic = True



_pending=[]
_kv_reservation=None
_kv_reservation_range=None


def release_loaded_file_cache():
    import os,pathlib
    def snapshot():
        out={}
        for path in ['/diag-host-arcstats','/sys/fs/cgroup/memory.stat']:
            try:
                rows=pathlib.Path(path).read_text().splitlines()
                if path.endswith('arcstats'):
                    out['physical']={x.split()[0]:int(x.split()[2]) for x in rows[2:] if len(x.split())==3 and x.split()[0] in ['memory_available_bytes','memory_free_bytes','size']}
                else:out['cgroup']={a:int(b) for a,b in (x.split() for x in rows) if a in ['anon','file','shmem','file_mapped']}
            except OSError:pass
        return out
    gc.collect();before=snapshot();count=0;total=0
    paths=list(pathlib.Path('/data/fp8-backbone').glob('*.safetensors'))+list((pathlib.Path(os.environ['NQ_REPACK'])/'res/rank0').glob('L*.pt'))
    if not paths:raise RuntimeError('No loaded weight cache paths')
    for path in paths:
        fd=os.open(path,os.O_RDONLY)
        try:
            size=os.fstat(fd).st_size
            os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
            count+=1;total+=size
        finally:os.close(fd)
    print(json.dumps({'nq_loaded_file_cache_release':True,'files':count,'file_bytes':total,'before':before,'after':snapshot(),'time':time.time()}),flush=True)

def patch_runtime(module):
    original=module.Runtime.start
    @functools.wraps(original)
    def deferred(self,*args,**kwargs):
        if self.started:raise RuntimeError('Deferred runtime already started')
        if any(item[0] is self for item in _pending):raise RuntimeError('Runtime queued twice')
        _pending.append((self,original,args,kwargs))
        print(json.dumps({'nq_loader_memory':'pool_deferred_until_encoder_profile_complete','time':time.time(),'resident_layers':len(self.layers)}),flush=True)
    module.Runtime.start=deferred
    forward_original=module.Runtime.forward
    @functools.wraps(forward_original)
    def forward(self,L,x,*args,**kwargs):
        import torch,os
        if not self.started:
            if len(_pending)!=1 or _pending[0][0] is not self:raise RuntimeError('Expected one pending runtime before first expert')
            if set(self.layers)!=set(self.layer_ids):raise RuntimeError('Incomplete resident layers')
            runner=getattr(self,'_startup_runner',None)
            if runner is None:raise RuntimeError('First expert outside startup dummy')
            release_profile_cache('before_first_expert_pool')
            global _kv_reservation,_kv_reservation_range
            kv_bytes=int(runner.cache_config.kv_cache_memory_bytes or 0)
            if kv_bytes<=0:raise RuntimeError('KV reservation requires explicit fixed quota')
            if _kv_reservation is not None:raise RuntimeError('KV already reserved')
            print(json.dumps({'nq_kv_reservation':'begin','time':time.time(),'bytes':kv_bytes}),flush=True)
            _kv_reservation=torch.zeros(kv_bytes,dtype=torch.int8,device=x.device)
            _kv_reservation_range=(_kv_reservation.data_ptr(),kv_bytes)
            torch.cuda.synchronize()
            print(json.dumps({'nq_kv_reservation':'ready_before_pool','time':time.time(),'bytes':kv_bytes,'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved()}),flush=True)
            before=torch.cuda.memory_allocated()
            import prewarm_mla,startup_sentinel
            startup_sentinel.begin()
            prewarm_mla.run(runner,x)
            from moe import pf_scratch,pf_bytes
            H,I=4096,2048
            R=int(os.environ.get('NQ_PF_ROWS','8192'));G=int(os.environ.get('NQ_PF_G','16'))
            if (R,G)!=(8192,16):raise RuntimeError('Unexpected prefill scratch configuration')
            print(json.dumps({'nq_prescratch':'begin','time':time.time(),'bytes':pf_bytes(H,I,R,G)}),flush=True)
            scratch=pf_scratch(x.device,H,I,R,G)
            # Commit the same original shared buffers before host-slot pinning; values are scratch only.
            for key in ['Wgu','Wdn','xg','xu','acc','h','z']:scratch[key].zero_()
            torch.cuda.synchronize()
            print(json.dumps({'nq_prescratch':'ready','time':time.time(),'allocated_bytes':torch.cuda.memory_allocated()}),flush=True)
            release_loaded_file_cache()
            runtime,start,start_args,start_kwargs=_pending.pop()
            start(runtime,*start_args,**start_kwargs)
            if not self.started or sum(self.predictor.budgets.values())!=2630 or int(os.environ['NQ_FLASH_SPARE_SLOTS'])!=8:raise RuntimeError('First expert allocation invariant failed')
            runner.model_memory_usage+=torch.cuda.memory_allocated()-before
            startup_sentinel.mark_accounted(runner)
            print(json.dumps({'nq_loader_memory':'pool_ready_before_first_expert','layer':L,'time':time.time(),'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved()}),flush=True)
        diagnostic=getattr(self,'_startup_runner',None) is not None
        if diagnostic:print(json.dumps({'nq_layer_memory':'before_forward','layer':L,'tokens':len(x),'time':time.time(),'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved()}),flush=True)
        if diagnostic:
            print(json.dumps({'nq_gpu_completion':'before_expert_sync_begin','layer':L,'time':time.time()}),flush=True)
            torch.cuda.synchronize()
            print(json.dumps({'nq_gpu_completion':'before_expert_sync_end','layer':L,'time':time.time()}),flush=True)
        result=forward_original(self,L,x,*args,**kwargs)
        if diagnostic:
            print(json.dumps({'nq_gpu_completion':'after_expert_sync_begin','layer':L,'time':time.time()}),flush=True)
            torch.cuda.synchronize()
            print(json.dumps({'nq_gpu_completion':'after_expert_sync_end','layer':L,'time':time.time()}),flush=True)
        if diagnostic:print(json.dumps({'nq_layer_memory':'after_forward','layer':L,'time':time.time(),'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved()}),flush=True)
        return result
    module.Runtime.forward=forward

def trim_cpu_cache(phase):
    import pathlib
    class MallInfo(ctypes.Structure):
        _fields_=[(k,ctypes.c_size_t) for k in ['arena','ordblks','smblks','hblks','hblkhd','usmblks','fsmblks','uordblks','fordblks','keepcost']]
    libc=ctypes.CDLL(None)
    def stats():
        o={}
        try:
            libc.mallinfo2.restype=MallInfo
            m=libc.mallinfo2();o={k:getattr(m,k) for k in ['arena','hblkhd','uordblks','fordblks','keepcost']}
        except AttributeError:pass
        o['process_memory']={x.split(':')[0]:x.split(':',1)[1].strip() for x in pathlib.Path('/proc/self/status').read_text().splitlines() if x.startswith(('VmRSS:','RssAnon:','RssFile:','RssShmem:'))}
        return o
    before=stats();gc.collect()
    libc.malloc_trim.argtypes=[ctypes.c_size_t];libc.malloc_trim.restype=ctypes.c_int
    trimmed=libc.malloc_trim(0)
    print(json.dumps({'nq_cpu_cache':phase,'time':time.time(),'before':before,'after':stats(),'trimmed':trimmed}),flush=True)

def release_profile_cache(phase):
    import torch
    torch.cuda.synchronize()
    before=torch.cuda.memory_reserved()
    allocated=torch.cuda.memory_allocated()
    gc.collect();torch.cuda.empty_cache()
    trim_cpu_cache(phase)
    free,total=torch.cuda.mem_get_info()
    print(json.dumps({'nq_loader_memory':phase,'time':time.time(),
         'allocated_bytes':allocated,'reserved_before_bytes':before,
         'reserved_bytes':torch.cuda.memory_reserved(),
         'cuda_free_bytes':free,'cuda_total_bytes':total,'allocator_stats':{k:v for k,v in torch.cuda.memory_stats().items() if k in ('inactive_split_bytes.all.current','num_alloc_retries','num_ooms')}}),flush=True)

def patch_runner(module):
    import torch
    original=module.GPUModelRunner._dummy_run
    @functools.wraps(original)
    def dummy_run(self,*args,**kwargs):
        import nq_flash_runtime
        runtime=nq_flash_runtime.RT
        runtime._startup_runner=self
        handles=[]
        def boundary(phase,name):
            print(json.dumps({'nq_module_boundary':phase,'module':name,'time':time.time(),'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved()}),flush=True)
        for name,child in self.model.named_modules():
            if type(child).__name__=='Glm5NextDecoderLayer':
                handles.append(child.register_forward_pre_hook(lambda m,a,n=name:boundary('before',n)))
                handles.append(child.register_forward_hook(lambda m,a,o,n=name:boundary('after',n)))
        print(json.dumps({'nq_dummy_start':True,'tokens':args[0] if args else kwargs.get('num_tokens'),'max_num_tokens':self.max_num_tokens,'time':time.time()}),flush=True)
        try:result=original(self,*args,**kwargs)
        finally:
            for h in handles:h.remove()
            runtime._startup_runner=None
            import startup_sentinel
            startup_sentinel.release()
        release_profile_cache('after_language_dummy')
        return result
    module.GPUModelRunner._dummy_run=dummy_run
    original_profile=module.GPUModelRunner.profile_run
    @functools.wraps(original_profile)
    def profile_run(self,*args,**kwargs):
        result=original_profile(self,*args,**kwargs)
        release_profile_cache('after_complete_profile')
        return result
    module.GPUModelRunner.profile_run=profile_run


def patch_kv_admission(module):
    original=module._check_enough_kv_cache_memory
    @functools.wraps(original)
    def check(available_memory,get_needed_memory,max_model_len,estimate_max_model_len):
        needed=get_needed_memory()
        print(json.dumps({'nq_kv_admission':True,'time':time.time(),
          'max_model_len':max_model_len,'available_bytes':available_memory,
          'needed_bytes':needed,'fits':needed<=available_memory}),flush=True)
        return original(available_memory,get_needed_memory,max_model_len,estimate_max_model_len)
    module._check_enough_kv_cache_memory=check

def patch_mla_profile(module):
    import inspect,torch,os
    original=module.MLAAttention.forward_impl
    sig=inspect.signature(original)
    @functools.wraps(original)
    def forward_impl(self,*args,**kwargs):
        bound=sig.bind(self,*args,**kwargs);bound.apply_defaults();a=bound.arguments
        selected=(os.environ.get('NQ_FLASH_MLA_CACHE')=='fp4_g16' and getattr(self.impl,'_nq_fp4_kv',False) and self.prefill_backend is None)
        plain=(a['output_scale'] is None and a['output_block_scale'] is None)
        if selected and plain and a['attn_metadata'] is None:
            q=a['q'];out=a['output'];n=q.shape[0];h=self.num_heads
            if h!=64 or self.kv_lora_rank!=512 or q.dtype not in (torch.float16,torch.bfloat16):raise RuntimeError('Unexpected FP4 sparse profile dimensions')
            width=self.impl.topk_indices_buffer.shape[1]
            if width not in (2048,2176):raise RuntimeError('Unexpected FP4 sparse window')
            # Real FP4 path: latent query, latent output, plus a conservative
            # additional latent tensor; two physical-index buffers; bounded
            # 16-query eight-split FP32 partial and log-normalizer scratch.
            # The indexer independently reserves its gather/logits workspace.
            latent=3*n*h*512*q.element_size()
            indices=2*n*width*4
            partial=min(n,16)*8*h*(512+1)*4
            scratch_bytes=latent+indices+partial
            scratch=torch.empty(scratch_bytes,dtype=torch.uint8,device=q.device)
            print(json.dumps({'nq_sparse_profile':True,'time':time.time(),'tokens':n,'heads':h,'scratch_bytes':scratch_bytes,'unused_dense_sentinel_bytes':self.chunked_prefill_workspace_size*h*(self.qk_nope_head_dim+self.v_head_dim)*q.element_size(),'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved()}),flush=True)
            return out.fill_(0)
        return original(self,*args,**kwargs)
    module.MLAAttention.forward_impl=forward_impl

def patch_warmup(module):
    original=module.warmup_kernels
    @functools.wraps(original)
    def warmup(model_runner,worker_execute_model,worker_sample_tokens):
        release_profile_cache('before_scheduler_warmup')
        count={'execute':0,'sample':0}
        def wrap(fn,kind):
            def call(*args,**kwargs):
                count[kind]+=1
                phase='scheduler_warmup_'+kind+'_'+str(count[kind])
                print(json.dumps({'nq_warmup_phase':phase,'phase':'begin','time':time.time()}),flush=True)
                result=fn(*args,**kwargs)
                release_profile_cache(phase)
                print(json.dumps({'nq_warmup_phase':phase,'phase':'end','time':time.time()}),flush=True)
                return result
            return call
        return original(model_runner,wrap(worker_execute_model,'execute'),wrap(worker_sample_tokens,'sample'))
    module.warmup_kernels=warmup


def patch_loader_pressure(module):
    """Keep upstream UMA thresholds, using host physical counters in LXC."""
    import pathlib
    original=module.release_device_memory_under_pressure
    def release(device):
        import torch
        if device.type!='cuda':return original(device)
        from vllm.platforms import current_platform
        if not current_platform.is_integrated_gpu(device.index):return original(device)
        source=pathlib.Path('/diag-host-arcstats')
        rows={x.split()[0]:int(x.split()[2]) for x in source.read_text().splitlines()[2:] if len(x.split())==3}
        available=rows['memory_available_bytes'];total=rows['memory_all_bytes']
        if not 0<=available<=total:raise RuntimeError('Invalid physical host memory counters')
        reserved=torch.cuda.memory_reserved(device);allocated=torch.cuda.memory_allocated(device)
        releasable=reserved-allocated
        # Same upstream thresholds: at least512MiB cache, under20% host headroom.
        if releasable<512*1024**2 or available>0.2*total:return False
        torch.cuda.synchronize(device);torch.cuda.empty_cache()
        print(json.dumps({'nq_loader_physical_release':True,'time':time.time(),'host_available_bytes':available,'host_total_bytes':total,'allocated_bytes':allocated,'reserved_before_bytes':reserved,'reserved_after_bytes':torch.cuda.memory_reserved(device)}),flush=True)
        return True
    module.release_device_memory_under_pressure=release


def patch_resident(module):
    original=module.load
    @functools.wraps(original)
    def load(path,dev='cuda'):
        import torch
        experts,H,I=original(path,dev)
        live=[(E,x.lr) for E,x in experts.items() if x.lr is not None]
        if not live:return experts,H,I
        old_storages={v.untyped_storage().data_ptr():v.untyped_storage().nbytes() for _,v in live}
        if any(not v.is_contiguous() or v.dtype!=torch.float16 for _,v in live):raise RuntimeError('Unexpected resident low-rank layout')
        packed=torch.cat([v for _,v in live])
        offset=0
        for E,old in live:
            n=old.numel();new=packed.narrow(0,offset,n)
            if offset*packed.element_size()%128:raise RuntimeError('Low-rank packed row lost128byte alignment')
            if new.is_cuda and new.data_ptr()%128:raise RuntimeError('Low-rank CUDA pointer alignment')
            if not torch.equal(old.view(torch.uint8),new.view(torch.uint8)):raise RuntimeError('Low-rank byte mismatch')
            experts[E].lr=new;offset+=n
        if offset!=packed.numel():raise RuntimeError('Low-rank compact storage length mismatch')
        print(json.dumps({'nq_resident_lr_compact':True,'time':time.time(),'layer_file':str(path).rsplit('/',1)[-1],'experts':len(experts),'live_experts':len(live),'old_storage_bytes':sum(old_storages.values()),'new_storage_bytes':packed.numel()*packed.element_size(),'saved_bytes':sum(old_storages.values())-packed.numel()*packed.element_size(),'byte_equal':True,'row_alignment_bytes':128}),flush=True)
        # The mmap-backed CPU tensors are no longer referenced after original()
        # has copied every resident plane to CUDA.  Evict this clean source file
        # now instead of accumulating all 42 layer files until first forward;
        # on a 116 GiB LXC that otherwise consumes the final layer's headroom.
        import gc,os
        gc.collect()
        fd=os.open(path,os.O_RDONLY)
        try:
            size=os.fstat(fd).st_size
            os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        print(json.dumps({'nq_resident_source_cache_release':True,'time':time.time(),'layer_file':str(path).rsplit('/',1)[-1],'file_bytes':size}),flush=True)
        return experts,H,I
    module.load=load


def patch_renderer(module):
    cls=module.BaseRenderer
    original=cls._warmup_mm_processor
    @functools.wraps(original)
    def warm(self,*args,**kwargs):
        label=kwargs.get('log_prefix','unknown')
        done=getattr(self,'_nq_early_mm_done',{})
        processor=args[0] if args else kwargs.get('processor')
        if label in done:
            saved,length=done.pop(label)
            if saved is not processor or length!=self.model_config.max_model_len:raise RuntimeError('Early processor warmup identity/config changed')
            print(json.dumps({'nq_api_warmup':label,'phase':'already_completed_before_engine','time':time.time()}),flush=True)
            return None
        print(json.dumps({'nq_api_warmup':label,'phase':'begin','time':time.time()}),flush=True)
        try:return original(self,*args,**kwargs)
        finally:trim_cpu_cache('api_processor_'+label)
    cls._warmup_mm_processor=warm
    original_clear=cls.clear_mm_cache
    @functools.wraps(original_clear)
    def clear(self,*args,**kwargs):
        print(json.dumps({'nq_api_cache_clear':'begin','time':time.time()}),flush=True)
        result=original_clear(self,*args,**kwargs)
        gc.collect();trim_cpu_cache('api_mm_cache_cleared')
        return result
    cls.clear_mm_cache=clear
    original_static=cls._clear_processor_cache
    def clear_static(processor):
        result=original_static(processor)
        gc.collect();trim_cpu_cache('api_readonly_cache_cleared')
        return result
    cls._clear_processor_cache=staticmethod(clear_static)


def patch_early_renderer(module):
    original=module.renderer_from_config
    @functools.wraps(original)
    def create(*args,**kwargs):
        renderer=original(*args,**kwargs)
        from vllm.renderers.base import set_default_torch_num_threads
        done={}
        print(json.dumps({'nq_early_api_warmup':'begin_before_engine','time':time.time()}),flush=True)
        with set_default_torch_num_threads(1):
            processor=renderer.mm_processor
            if processor is not None:
                renderer._warmup_mm_processor(processor,log_prefix='Multi-modal')
                done['Multi-modal']=(processor,renderer.model_config.max_model_len)
                renderer.clear_mm_cache()
            processor=renderer._readonly_mm_processor
            if processor is not None:
                renderer._warmup_mm_processor(processor,log_prefix='Readonly multi-modal')
                done['Readonly multi-modal']=(processor,renderer.model_config.max_model_len)
                renderer._clear_processor_cache(processor)
        gc.collect();trim_cpu_cache('early_api_warmups_completed')
        renderer._nq_early_mm_done=done
        print(json.dumps({'nq_early_api_warmup':'completed_before_engine','processors':list(done),'time':time.time()}),flush=True)
        return renderer
    module.renderer_from_config=create

def patch_kv_allocation(module):
    original=module._allocate_kv_cache
    @functools.wraps(original)
    def allocate(config,shared_layers,device):
        import torch
        global _kv_reservation
        tensors=config.kv_cache_tensors
        packed=[int(t.size) for t in tensors if t.block_stride>0]
        needed=sum(int(t.size) for t in tensors if t.block_stride<=0)+(packed[0] if packed else 0)
        if packed and len(set(packed))!=1:raise RuntimeError('Unexpected packed KV backing sizes')
        if _kv_reservation is None:raise RuntimeError('Missing pre-pool KV reservation')
        start,capacity=_kv_reservation_range
        if needed>capacity:raise RuntimeError('Actual KV backing exceeds reserved fixed quota')
        print(json.dumps({'nq_kv_allocation':'begin','time':time.time(),'needed_bytes':needed,'reserved_quota_bytes':capacity,'tensor_sizes':[int(t.size) for t in tensors],'packed':[t.block_stride>0 for t in tensors]}),flush=True)
        torch.cuda.synchronize()
        # Return the live reservation to the native caching allocator immediately
        # before the unmodified allocator requests the same fixed KV storage.
        _kv_reservation=None
        result=original(config,shared_layers,device)
        torch.cuda.synchronize()
        storages={v.untyped_storage().data_ptr():v.untyped_storage().nbytes() for v in result.values()}
        reused=all(start<=ptr and ptr+size<=start+capacity for ptr,size in storages.items())
        print(json.dumps({'nq_kv_allocation':'complete','time':time.time(),'unique_storage_bytes':sum(storages.values()),'reused_reservation':reused,'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved()}),flush=True)
        return result
    module._allocate_kv_cache=allocate

def patch_predictor_request_trim(module):
    import functools,json,time,torch
    original=module.CommittedPredictor.reset
    @functools.wraps(original)
    def reset(self,*args,**kwargs):
        had_request=getattr(self,'t',0)>0
        result=original(self,*args,**kwargs)
        if had_request:
            allocated=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved()
            if reserved-allocated>=128*1024**2:
                torch.cuda.empty_cache()
                print(json.dumps({'request_allocator_trim':True,'time':time.time(),'allocated_before':allocated,'reserved_before':reserved,'allocated_after':torch.cuda.memory_allocated(),'reserved_after':torch.cuda.memory_reserved()}),flush=True)
        return result
    module.CommittedPredictor.reset=reset

def install():
    import sys
    import publication_observer
    import post_layer_publication
    targets={'jt_runtime':patch_predictor_request_trim,'executor':publication_observer.patch_executor,'vllm.v1.worker.gpu.attn_utils':patch_kv_allocation,'vllm.v1.engine.async_llm':patch_early_renderer,'vllm.renderers.base':patch_renderer,'resident':patch_resident,
             'vllm.model_executor.model_loader.utils':patch_loader_pressure,
             'vllm.v1.worker.gpu.warmup':patch_warmup,
             'vllm.model_executor.layers.attention.mla_attention':patch_mla_profile,
             'vllm.model_executor.model_loader.base_loader':patch,
             'nq_flash_runtime':lambda m:(patch_runtime(m),post_layer_publication.patch_runtime(m)),
             'vllm.v1.worker.gpu.model_runner':patch_runner,
             'vllm.v1.core.kv_cache_utils':patch_kv_admission}
    class Finder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in targets:return None
            for finder in sys.meta_path:
                if finder is self or not hasattr(finder,'find_spec'):continue
                spec=finder.find_spec(fullname,path,target)
                if spec is not None:
                    original_exec=spec.loader.exec_module
                    def execute(module):
                        original_exec(module)
                        targets[fullname](module)
                    spec.loader.exec_module=execute
                    return spec
            return None
    for name,fn in targets.items():
        if name in sys.modules:fn(sys.modules[name])
    sys.meta_path.insert(0,Finder())
