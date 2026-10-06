// nq_pgprobe.cu (pg53 stage 1, experimental, default off): per-layer decode timing + end-to-end "pre-gate load lands
// before its MoE" probe on the live serve.
//  stamp(buf, L, pt, T, first)  graph-capturable 1-thread kernel at MoE start (pt 0, right before the mailbox apply)
//                               and MoE end (pt 1) of every NQ layer call with T <= 8 tokens. Writes %globaltimer into
//                               the per-step ring, at pt 0 also the ack value of layer L (the last step whose probe load
//                               for L the host finished), and posts a trigger (step, L, pt, t) to the host ring.
//  start(buf, path, rb, nrec, dev_scratch, nmax)  host thread (no GIL): spins on the trigger ring; for a trigger with
//                               pt == ctl_pt and (L + 1) % ctl_stride == 0 it reads ctl_n random records of `path`
//                               (io_uring, O_DIRECT, one SQE each) into pinned bounce buffers, cudaMemcpyAsync's them to
//                               dev_scratch, synchronizes, and writes ack[L + 1] = step. ctl_n = 0: ack at once (reaction
//                               latency only). Host times (CLOCK_REALTIME ns) of seen / read done / copy done go to hlog.
// Layout (int64 words, pinned host): see OFF_* below.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <liburing.h>
#include <fcntl.h>
#include <unistd.h>
#include <thread>
#include <atomic>
#include <random>
#include <time.h>

#define NLM 96
#define NS 4096
#define NR 8192
#define NF 4
enum { H_STEP = 0, H_TSEQ = 1, C_PT = 2, C_N = 3, C_STRIDE = 4, C_RUN = 5, H_HSEQ = 6 };
#define OFF_TRIG 16                      // NR x 4: step, L, pt, t
#define OFF_ACK (OFF_TRIG + NR * 4)      // NLM
#define OFF_TS (OFF_ACK + NLM)           // NS x NLM x NF: t0, t1, ack seen at pt0, T
#define OFF_HLOG (OFF_TS + NS * NLM * NF) // NR x 6: step, L, t_seen, t_read, t_done, n
#define TOTAL (OFF_HLOG + NR * 6)

__global__ void stamp_k(volatile long long* b, int L, int pt, int T, int first) {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  long long step = b[H_STEP];
  if (first && pt == 0) { step += 1; b[H_STEP] = step; }
  volatile long long* ts = b + OFF_TS + ((step % NS) * NLM + L) * NF;
  ts[pt] = (long long)t;
  if (pt == 0) { ts[2] = b[OFF_ACK + L]; ts[3] = T; }
  long long q = b[H_TSEQ];
  volatile long long* tr = b + OFF_TRIG + (q % NR) * 4;
  tr[0] = step; tr[1] = L; tr[2] = pt; tr[3] = (long long)t;
  __threadfence_system();
  b[H_TSEQ] = q + 1;
  __threadfence_system();
}
__global__ void now_k(long long* o) {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  o[0] = (long long)t;
}

void stamp(int64_t buf, int64_t L, int64_t pt, int64_t T, int64_t first) {
  stamp_k<<<1, 1, 0, at::cuda::getCurrentCUDAStream().stream()>>>((volatile long long*)buf, (int)L, (int)pt, (int)T, (int)first);
}
static long long hnow() { timespec t; clock_gettime(CLOCK_REALTIME, &t); return t.tv_sec * 1000000000LL + t.tv_nsec; }
// gpu globaltimer - host CLOCK_REALTIME offset (ns), and the round trip it was measured in
std::vector<int64_t> calib(int64_t pinned) {
  long long* o = (long long*)pinned;
  auto st = at::cuda::getCurrentCUDAStream().stream();
  long long best = 1LL << 62, off = 0;
  for (int i = 0; i < 50; i++) {
    long long a = hnow(); now_k<<<1, 1, 0, st>>>(o); cudaStreamSynchronize(st); long long c = hnow();
    if (c - a < best) { best = c - a; off = o[0] - (a + c) / 2; }
  }
  return {off, best};
}

static std::atomic<bool> g_stop{false};
static std::thread g_th;
void start(int64_t buf, std::string path, int64_t rb, int64_t nrec, int64_t dev_scratch, int64_t nmax, int64_t device) {
  g_stop = false;
  g_th = std::thread([=]() {
    cudaSetDevice((int)device);
    cudaStream_t st; cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
    volatile long long* b = (volatile long long*)buf;
    int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    std::vector<void*> bounce(nmax);
    for (auto& p : bounce) cudaHostAlloc(&p, rb, cudaHostAllocDefault);
    io_uring ring; io_uring_queue_init(64, &ring, 0);
    std::mt19937_64 rng(1234 + device);
    long long seen = b[H_TSEQ];
    while (!g_stop) {
      long long q = b[H_TSEQ];
      if (q == seen) { continue; }
      if (q - seen > NR) seen = q - NR;
      for (; seen < q; seen++) {
        volatile long long* tr = b + OFF_TRIG + (seen % NR) * 4;
        long long step = tr[0], L = tr[1], pt = tr[2];
        if (!b[C_RUN] || pt != b[C_PT] || L + 1 >= NLM || (L + 1) % (b[C_STRIDE] > 0 ? b[C_STRIDE] : 1)) continue;
        long long t0 = hnow(), n = b[C_N]; if (n > nmax) n = nmax;
        long long t1 = t0;
        if (n > 0 && fd >= 0) {
          for (int i = 0; i < n; i++) {
            io_uring_sqe* s = io_uring_get_sqe(&ring);
            io_uring_prep_read(s, fd, bounce[i], (unsigned)rb, (off_t)((rng() % nrec) * rb));
          }
          io_uring_submit(&ring);
          for (int i = 0; i < n; i++) { io_uring_cqe* c; io_uring_wait_cqe(&ring, &c); io_uring_cqe_seen(&ring, c); }
          t1 = hnow();
          for (int i = 0; i < n; i++) cudaMemcpyAsync((char*)dev_scratch + i * rb, bounce[i], rb, cudaMemcpyHostToDevice, st);
          cudaStreamSynchronize(st);
        }
        long long t2 = hnow();
        b[OFF_ACK + L + 1] = step;
        long long h = b[H_HSEQ];
        volatile long long* hl = b + OFF_HLOG + (h % NR) * 6;
        hl[0] = step; hl[1] = L + 1; hl[2] = t0; hl[3] = t1; hl[4] = t2; hl[5] = n;
        b[H_HSEQ] = h + 1;
      }
    }
    io_uring_queue_exit(&ring); if (fd >= 0) close(fd);
  });
}
void stop() { g_stop = true; if (g_th.joinable()) g_th.join(); }
int64_t total_words() { return TOTAL; }

TORCH_LIBRARY(nq_pgprobe, m) {
  m.def("stamp", &stamp); m.def("calib", &calib); m.def("start", &start); m.def("stop", &stop); m.def("total_words", &total_words);
}
