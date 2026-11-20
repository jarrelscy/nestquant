"""Read GLM-5.3-Flash KV cache chunks that vLLM + LMCache (MP server, `resp` L2 tier) stored in a shared Kvrocks.

The store holds one object per (7168-token chunk, TP rank, object group), written by bigbox's vLLM deployment
(TP=2, DCP=2, kv-cache-dtype fp8_ds_mla, B12X k-pool indexer, KDA state in align mode). The format is described in
bigbox:~/lmcache-shared-format.md; the constants below come from it. This module computes the keys from token
ids, fetches the objects over RESP and converts them into this runtime's cache layout:

  * MLA latent: fp8 e4m3 x per-128 fp32 scales, tokens interleaved across the two ranks (t % 2).
  * Indexer pooled keys: Hadamard-rotated fp8 with power-of-two scales, stored permuted by LMCache, pools
    interleaved across ranks (p % 2); this runtime keeps them unrotated.
  * KDA: conv state (last 3 inputs) and recurrent state at the end of the last chunk, heads split by rank.

Not stored there: the drafter's context and the expert predictor's history, so a restore is followed by a
prefill of the remaining tokens.
"""

import os
import socket
import struct
import threading
import time

import numpy as np

CHUNK = 7168
MODEL = os.environ.get('NQ_KVSHARE_MODEL', '/model')     # the vLLM model path; a test prefix keeps test keys apart
KV_RANKS = [(2 << 24) | (r << 16) | (2 << 8) | r for r in range(2)]          # 0x02000200, 0x02010201
G0_BYTES, G1_BYTES = 27_163_136, 79_937_536
INDEX_LAYER_BYTES, INDEX_SLOTS = 118_272, 896
MLA_OFFSET, MLA_LAYER_BYTES, MLA_ROWS, MLA_RECORD = 1_300_992, 2_351_104, 3584, 656
KDA_PAGE = 2_351_104
KDA_ORDER = [0, 5, 10, 16, 21, 26, 32, 37, 42, 1, 6, 12, 17, 22, 28, 33, 38, 44,
             2, 8, 13, 18, 24, 29, 34, 40, 4, 9, 14, 20, 25, 30, 36, 41]
FULL_ATTN = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]
CONV_ROWS, CONV_CHANNELS_RANK = 8, 12288       # bf16; rows 0-2 are the last 3 inputs, rank channels q|k|v
HEADS_RANK, HEAD_DIM = 32, 128


# ---------------------------------------------------------------------------------------------- keys

def _blake3(data):
    import blake3
    return blake3.blake3(data).digest()


NONE_HASH = None


def chunk_hashes(token_ids):
    """BLAKE3 chain over complete 7168-token chunks (LMCache TokenHasher)."""
    global NONE_HASH
    if NONE_HASH is None:
        NONE_HASH = _blake3((0).to_bytes(8, 'big', signed=True) + struct.pack('>I', 0))
    tokens = np.asarray(token_ids, dtype='>u4')
    h, out = NONE_HASH, []
    for start in range(0, len(tokens) - len(tokens) % CHUNK, CHUNK):
        h = _blake3(h + tokens[start:start + CHUNK].tobytes())
        out.append(h)
    return out


def object_key(chunk_hash, rank, group):
    return f'{MODEL}@{KV_RANKS[rank]:08x}@{group:x}@{chunk_hash.hex()}'


# ---------------------------------------------------------------------------------------------- RESP client

class Resp:
    """Minimal RESP2 client (AUTH, EXISTS, GET) with buffered bulk reads."""

    def __init__(self, host, port, password=None, timeout=30.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
        self.buffer = bytearray()
        if password:
            if self.command('AUTH', password) != b'OK':
                raise PermissionError('Kvrocks rejected the password')

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    def _send(self, *parts):
        out = [b'*%d\r\n' % len(parts)]
        for part in parts:
            data = part if isinstance(part, bytes) else str(part).encode()
            out += [b'$%d\r\n' % len(data), data, b'\r\n']
        self.sock.sendall(b''.join(out))

    def _fill(self, n):
        while len(self.buffer) < n:
            chunk = self.sock.recv(max(1 << 20, n - len(self.buffer)))
            if not chunk:
                raise ConnectionError('Kvrocks closed the connection')
            self.buffer += chunk

    def _line(self):
        while True:
            end = self.buffer.find(b'\r\n')
            if end >= 0:
                line = bytes(self.buffer[:end])
                del self.buffer[:end + 2]
                return line
            self._fill(len(self.buffer) + 1)

    def _reply(self):
        line = self._line()
        kind, rest = line[:1], line[1:]
        if kind == b'+':
            return rest
        if kind == b'-':
            raise RuntimeError(f'Kvrocks error: {rest.decode(errors="replace")}')
        if kind == b':':
            return int(rest)
        if kind == b'$':
            n = int(rest)
            if n < 0:
                return None
            self._fill(n + 2)
            data = bytes(self.buffer[:n])
            del self.buffer[:n + 2]
            return data
        if kind == b'*':
            return [self._reply() for _ in range(int(rest))]
        raise RuntimeError(f'Unexpected RESP reply {line[:40]!r}')

    def command(self, *parts):
        self._send(*parts)
        return self._reply()

    def pipeline(self, commands):
        for parts in commands:
            self._send(*parts)
        return [self._reply() for _ in commands]


# ---------------------------------------------------------------------------------------------- decoding

def _e4m3_table():
    codes = np.arange(256, dtype=np.uint32)
    sign = np.where(codes & 0x80, -1.0, 1.0)
    exponent = (codes >> 3) & 0xF
    mantissa = codes & 0x7
    value = np.where(exponent == 0, mantissa / 8.0 * 2.0 ** -6, (1 + mantissa / 8.0) * 2.0 ** (exponent.astype(np.int64) - 7))
    value = np.where((exponent == 15) & (mantissa == 7), np.nan, value)
    return (sign * value).astype(np.float32)


E4M3 = _e4m3_table()


def hadamard(n=128):
    h = np.array([[1.0]], dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


HADAMARD = hadamard(128)


def bf16_to_float(raw):
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def decode_mla(objects, layer):
    """Latent (7168, 512) float32 of one chunk for full-attention layer index `layer` (0..10).
    objects: (rank0 group0 bytes, rank1 group0 bytes)."""
    out = np.empty((CHUNK, 512), dtype=np.float32)
    for rank, data in enumerate(objects):
        start = MLA_OFFSET + layer * MLA_LAYER_BYTES
        rows = np.frombuffer(data, dtype=np.uint8, count=MLA_LAYER_BYTES, offset=start).reshape(MLA_ROWS, MLA_RECORD)
        values = E4M3[rows[:, :512]]
        scales = np.ascontiguousarray(rows[:, 512:528]).view('<f4')          # (3584, 4)
        out[rank::2] = (values.reshape(MLA_ROWS, 4, 128) * scales[:, :, None]).reshape(MLA_ROWS, 512)
    return out


def decode_index(objects, layer):
    """Unrotated pooled indexer keys (1792, 128) float32 of one chunk for full-attention layer index `layer`."""
    out = np.empty((CHUNK // 4, 128), dtype=np.float32)
    for rank, data in enumerate(objects):
        rows = np.frombuffer(data, dtype=np.uint8, count=INDEX_LAYER_BYTES,
                             offset=layer * INDEX_LAYER_BYTES).reshape(INDEX_SLOTS, 132)
        raw = np.empty(INDEX_LAYER_BYTES, dtype=np.uint8)               # undo LMCache's BSV_BSS permutation
        raw[:INDEX_SLOTS * 128] = rows[:, :128].reshape(-1)
        raw[INDEX_SLOTS * 128:] = rows[:, 128:].reshape(-1)
        pages = raw.reshape(14, 8448)
        values = E4M3[pages[:, :8192].reshape(INDEX_SLOTS, 128)]
        scales = np.ascontiguousarray(pages[:, 8192:]).view('<f4').reshape(INDEX_SLOTS)
        out[rank::2] = (values * scales[:, None]) @ HADAMARD           # k_rot = k @ H, H symmetric orthogonal
    return out


def decode_kda(objects):
    """{model layer: (conv (3, 24576) float32, state (64, 128, 128) float32)} at the end of the chunk.
    objects: (rank0 group1 bytes, rank1 group1 bytes)."""
    result = {}
    conv_bytes = CONV_ROWS * CONV_CHANNELS_RANK * 2
    state_bytes = HEADS_RANK * HEAD_DIM * HEAD_DIM * 4
    for n, layer in enumerate(KDA_ORDER):
        convs, states = [], []
        for data in objects:
            start = n * KDA_PAGE
            conv = bf16_to_float(data[start:start + conv_bytes]).reshape(CONV_ROWS, CONV_CHANNELS_RANK)[:3]
            convs.append(conv.reshape(3, 3, CONV_CHANNELS_RANK // 3))       # (row, q|k|v, rank channels)
            states.append(np.frombuffer(data, dtype='<f4', count=state_bytes // 4,
                                        offset=start + conv_bytes).reshape(HEADS_RANK, HEAD_DIM, HEAD_DIM))
        conv = np.concatenate(convs, axis=2).reshape(3, -1)                   # q_r0 q_r1 | k_r0 k_r1 | v_r0 v_r1
        result[layer] = (conv, np.concatenate(states, axis=0))
    return result


# ---------------------------------------------------------------------------------------------- store access

class SharedKV:
    """Looks up and fetches a prompt's longest stored prefix of complete chunks."""

    def __init__(self, host, port=6666, password=None, connections=4):
        self.host, self.port, self.password = host, port, password
        self.connections = connections
        self._pool = []
        self._lock = threading.Lock()
        self.stats = dict(lookups=0, hits=0, chunks=0, bytes=0, seconds=0.0, errors=0)

    @classmethod
    def from_env(cls):
        host = os.environ.get('NQ_KVSHARE_HOST')
        if not host:
            return None
        password = os.environ.get('NQ_KVSHARE_PASSWORD')
        path = os.environ.get('NQ_KVSHARE_PASSWORD_FILE', os.path.expanduser('~/.config/lmcache-shared/kvrocks.pass'))
        if not password and os.path.exists(path):
            password = open(path).read().strip()
        return cls(host, int(os.environ.get('NQ_KVSHARE_PORT', '6666')), password)

    def _client(self):
        with self._lock:
            if self._pool:
                return self._pool.pop()
        return Resp(self.host, self.port, self.password)

    def _release(self, client):
        with self._lock:
            self._pool.append(client)

    def _call(self, run):
        """Run run(client) on a pooled connection; a dropped connection (Kvrocks restarted, idle timeout) is
        replaced and the call retried once."""
        for attempt in (0, 1):
            client = self._client()
            try:
                result = run(client)
            except (ConnectionError, OSError):
                client.close()
                with self._lock:
                    for stale in self._pool:
                        stale.close()
                    self._pool.clear()
                if attempt:
                    raise
                continue
            except Exception:
                client.close()
                raise
            self._release(client)
            return result

    def available_chunks(self, token_ids):
        """Number of leading chunks of `token_ids` stored completely (group 0 for each, group 1 for the last)."""
        hashes = chunk_hashes(token_ids)
        if not hashes:
            return 0, hashes
        self.stats['lookups'] += 1
        keys = [object_key(h, r, 0) for h in hashes for r in (0, 1)]
        try:
            present = self._call(lambda client: client.pipeline([('EXISTS', k) for k in keys]))
            n = 0
            while n < len(hashes) and present[2 * n] and present[2 * n + 1]:
                n += 1
            while n:
                last = self._call(lambda client: client.pipeline(
                    [('EXISTS', object_key(hashes[n - 1], r, 1)) for r in (0, 1)]))
                if all(last):
                    break
                n -= 1
        except Exception:
            self.stats['errors'] += 1
            raise
        return n, hashes

    def fetch(self, hashes, n):
        """Objects for chunks 0..n-1: ({(chunk, rank): group0 bytes}, (rank0 group1, rank1 group1))."""
        jobs = [(c, r, 0) for c in range(n) for r in (0, 1)] + [(n - 1, r, 1) for r in (0, 1)]
        results, errors = {}, []
        started = time.perf_counter()

        def worker(part):
            try:
                for c, r, g in part:
                    data = self._call(lambda client: client.command('GET', object_key(hashes[c], r, g)))
                    expected = G0_BYTES if g == 0 else G1_BYTES
                    if data is None or len(data) != expected:
                        raise ValueError(f'object {(c, r, g)} missing or {0 if data is None else len(data)} bytes')
                    results[(c, r, g)] = data
            except Exception as error:      # reported below
                errors.append(error)

        # Large group-1 objects first so they overlap the many group-0 ones.
        jobs.sort(key=lambda job: -job[2])
        parts = [jobs[i::self.connections] for i in range(self.connections)]
        threads = [threading.Thread(target=worker, args=(part,)) for part in parts if part]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            self.stats['errors'] += 1
            raise errors[0]
        seconds = time.perf_counter() - started
        self.stats['hits'] += 1
        self.stats['chunks'] += n
        self.stats['bytes'] += sum(len(v) for v in results.values())
        self.stats['seconds'] += seconds
        group0 = {(c, r): results[(c, r, 0)] for c in range(n) for r in (0, 1)}
        return group0, (results[(n - 1, 0, 1)], results[(n - 1, 1, 1)]), seconds


def assemble(group0, group1, n):
    """Per-layer numpy arrays for an n-chunk prefix:
    {'latent': {layer: (N, 512)}, 'index': {layer: (N/4, 128)}, 'kda': {layer: (conv, state)}}."""
    latent, index = {}, {}
    for a, layer in enumerate(FULL_ATTN):
        latent[layer] = np.concatenate([decode_mla((group0[(c, 0)], group0[(c, 1)]), a) for c in range(n)])
        index[layer] = np.concatenate([decode_index((group0[(c, 0)], group0[(c, 1)]), a) for c in range(n)])
    return dict(latent=latent, index=index, kda=decode_kda(group1))
