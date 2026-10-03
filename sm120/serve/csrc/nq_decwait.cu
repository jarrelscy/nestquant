// nq_decwait.cu (nq-kld): GPU-side bounded wait on a host-pinned flag, so the decode-step wait (nq_pfblock dec_wait)
// no longer blocks the host: execute_model enqueues spin(flag, n, timeout) ahead of step n's work and returns to its
// async prep; a host waiter thread writes flag = n once step n-1 finished, its hits were seen and the reads landed.
// The kernel returns at flag >= n or after timeout_ns of GPU time (%globaltimer) from its start.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

__global__ void nq_spin_k(const volatile long long* f, long long target, unsigned long long timeout_ns, long long* stat) {
  unsigned long long t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  bool ok = true;
  while (*f < target) {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    if (t - t0 > timeout_ns) { ok = false; break; }
    __nanosleep(500);
  }
  __threadfence_system();
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  if (stat) { stat[0] += 1; stat[1] += ok ? 0 : 1; stat[2] += (long long)(t - t0); }
}

void spin(int64_t flag_ptr, int64_t target, int64_t timeout_ns, int64_t stat_ptr) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  nq_spin_k<<<1, 1, 0, stream>>>((const volatile long long*)flag_ptr, (long long)target, (unsigned long long)timeout_ns,
                                 (long long*)stat_ptr);
}

TORCH_LIBRARY(nq_decwait, m) { m.def("spin(int flag_ptr, int target, int timeout_ns, int stat_ptr) -> ()"); }
TORCH_LIBRARY_IMPL(nq_decwait, CompositeExplicitAutograd, m) { m.impl("spin", &spin); }
