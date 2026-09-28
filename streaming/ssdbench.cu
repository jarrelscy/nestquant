// SSD read-path bench for NestQuant P4 streaming: io_uring + O_DIRECT into a pinned host bounce ring, then
// cudaMemcpyAsync into a device slot ring on a per-GPU side stream. One thread + one ring per GPU.
// Records are fixed size, 4 KiB aligned, read at random record indices (the upgrade pattern), optionally striped
// round-robin over several files (one per drive).
//
// build: nvcc -O2 -std=c++17 ssdbench.cu -I$LIBURING/include $LIBURING/lib/liburing.a -o ssdbench -lpthread
// run:   ssdbench --files a.bin[,b.bin] --rec 2420736 --qd 8 --gpus 0[,1,2,3] --secs 10 [--h2d 1] [--json out]
#include <cuda_runtime.h>
#include <liburing.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <pthread.h>
#include <time.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <atomic>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
static double now() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

struct Cfg { std::vector<std::string> files; size_t rec = 2420736; int qd = 8; std::vector<int> gpus{0}; double secs = 10; int h2d = 1; unsigned seed = 1; };
struct Res { double bytes = 0, t = 0; std::vector<float> lat_rd, lat_e2e; long nio = 0; };
struct Arg { const Cfg* c; int gpu, idx; Res r; };
static std::atomic<int> g_ready{0};

static void* worker(void* p)
{
    Arg* a = (Arg*)p; const Cfg& c = *a->c;
    CK(cudaSetDevice(a->gpu));
    cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    std::vector<int> fds; std::vector<size_t> nrec;
    for (auto& f : c.files)
    {
        int fd = open(f.c_str(), O_RDONLY | O_DIRECT); if (fd < 0) { perror(f.c_str()); exit(1); }
        struct stat sb; fstat(fd, &sb); fds.push_back(fd); nrec.push_back(sb.st_size / c.rec);
    }
    const int Q = c.qd;
    uint8_t* host; CK(cudaHostAlloc((void**)&host, (size_t)Q * c.rec, cudaHostAllocPortable));
    uint8_t* dev = nullptr; if (c.h2d) CK(cudaMalloc(&dev, (size_t)Q * c.rec));
    std::vector<cudaEvent_t> ev(Q); for (auto& e : ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    std::vector<double> t_sub(Q, 0), t_rd(Q, 0); std::vector<int> state(Q, 0);   // 0 free, 1 reading, 2 copying
    io_uring ring; if (io_uring_queue_init(Q, &ring, 0)) { perror("io_uring_queue_init"); exit(1); }
    std::mt19937_64 rng(c.seed * 7919 + a->idx);
    size_t tot_rec = 0; for (auto n : nrec) tot_rec += n;
    auto submit = [&](int s) {
        size_t r = rng() % tot_rec; int f = 0; size_t off;
        if (fds.size() > 1) { f = r % fds.size(); off = (r / fds.size()) % nrec[f] * c.rec; } else off = r * c.rec;
        io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        io_uring_prep_read(sqe, fds[f], host + (size_t)s * c.rec, c.rec, off);
        io_uring_sqe_set_data64(sqe, s); t_sub[s] = now(); state[s] = 1;
    };
    g_ready++; while (g_ready.load() < (int)c.gpus.size()) usleep(100);
    const double t0 = now(), warm = 1.0; double tstart = 0; bool meas = false;
    for (int s = 0; s < Q; ++s) submit(s);
    io_uring_submit(&ring);
    while (true)
    {
        double t = now();
        if (!meas && t - t0 > warm) { meas = true; tstart = t; a->r = Res(); }
        if (meas && t - tstart > c.secs) break;
        // retire finished H2D copies
        bool sub = false;
        for (int s = 0; s < Q; ++s)
            if (state[s] == 2 && cudaEventQuery(ev[s]) == cudaSuccess)
            {
                if (meas) { a->r.lat_e2e.push_back((float)(now() - t_sub[s])); a->r.bytes += c.rec; a->r.nio++; }
                submit(s); sub = true;
            }
        if (sub) io_uring_submit(&ring);
        io_uring_cqe* cqe; __kernel_timespec to{0, 200000};
        int rc = io_uring_wait_cqe_timeout(&ring, &cqe, &to);
        if (rc == -ETIME) continue;
        if (rc) { fprintf(stderr, "wait_cqe %d\n", rc); exit(1); }
        unsigned head; int n = 0;
        io_uring_for_each_cqe(&ring, head, cqe)
        {
            int s = (int)io_uring_cqe_get_data64(cqe);
            if (cqe->res != (int)c.rec) { fprintf(stderr, "short read %d\n", cqe->res); exit(1); }
            double tr = now(); t_rd[s] = tr;
            if (meas) a->r.lat_rd.push_back((float)(tr - t_sub[s]));
            if (c.h2d) { CK(cudaMemcpyAsync(dev + (size_t)s * c.rec, host + (size_t)s * c.rec, c.rec, cudaMemcpyHostToDevice, st)); CK(cudaEventRecord(ev[s], st)); state[s] = 2; }
            else { if (meas) { a->r.lat_e2e.push_back((float)(tr - t_sub[s])); a->r.bytes += c.rec; a->r.nio++; } submit(s); }
            ++n;
        }
        io_uring_cq_advance(&ring, n);
        if (!c.h2d && n) io_uring_submit(&ring);
    }
    a->r.t = now() - tstart;
    CK(cudaStreamSynchronize(st));
    // drain outstanding reads before freeing the bounce buffer
    int out = 0; for (int s = 0; s < Q; ++s) out += state[s] == 1;
    for (int s = 0; s < Q; ++s) if (state[s] == 1)
    {
        // count only; wait for all
    }
    while (out > 0) { io_uring_cqe* cqe; if (io_uring_wait_cqe(&ring, &cqe)) break; int s = (int)io_uring_cqe_get_data64(cqe); if (state[s] == 1) { state[s] = 0; --out; } io_uring_cqe_seen(&ring, cqe); }
    io_uring_queue_exit(&ring);
    for (auto fd : fds) close(fd);
    CK(cudaFreeHost(host)); if (dev) CK(cudaFree(dev));
    return nullptr;
}

static float pct(std::vector<float> v, double p) { if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[(size_t)(p * (v.size() - 1))]; }
static std::vector<std::string> split(const char* s) { std::vector<std::string> o; std::string cur; for (; *s; ++s) { if (*s == ',') { o.push_back(cur); cur.clear(); } else cur += *s; } o.push_back(cur); return o; }

int main(int argc, char** argv)
{
    Cfg c; const char* js = nullptr;
    for (int i = 1; i + 1 < argc; i += 2)
    {
        std::string k = argv[i]; const char* v = argv[i + 1];
        if (k == "--files") c.files = split(v);
        else if (k == "--rec") c.rec = strtoull(v, 0, 10);
        else if (k == "--qd") c.qd = atoi(v);
        else if (k == "--gpus") { c.gpus.clear(); for (auto& g : split(v)) c.gpus.push_back(atoi(g.c_str())); }
        else if (k == "--secs") c.secs = atof(v);
        else if (k == "--h2d") c.h2d = atoi(v);
        else if (k == "--seed") c.seed = atoi(v);
        else if (k == "--json") js = v;
    }
    if (c.files.empty() || c.rec % 4096) { fprintf(stderr, "need --files, rec %% 4096 == 0\n"); return 1; }
    std::vector<Arg> A(c.gpus.size()); std::vector<pthread_t> th(c.gpus.size());
    for (size_t i = 0; i < c.gpus.size(); ++i) { A[i].c = &c; A[i].gpu = c.gpus[i]; A[i].idx = i; pthread_create(&th[i], 0, worker, &A[i]); }
    for (auto& t : th) pthread_join(t, 0);
    double bytes = 0, tmax = 0; long nio = 0; std::vector<float> lr, le;
    for (auto& a : A) { bytes += a.r.bytes; tmax = std::max(tmax, a.r.t); nio += a.r.nio; lr.insert(lr.end(), a.r.lat_rd.begin(), a.r.lat_rd.end()); le.insert(le.end(), a.r.lat_e2e.begin(), a.r.lat_e2e.end()); }
    char buf[1024];
    snprintf(buf, sizeof buf, "{\"files\":%zu,\"rec\":%zu,\"qd\":%d,\"ngpu\":%zu,\"h2d\":%d,\"GBps\":%.3f,\"recs_per_s\":%.1f,"
             "\"rd_p50_ms\":%.3f,\"rd_p99_ms\":%.3f,\"e2e_p50_ms\":%.3f,\"e2e_p99_ms\":%.3f,\"n\":%ld}",
             c.files.size(), c.rec, c.qd, c.gpus.size(), c.h2d, bytes / tmax / 1e9, nio / tmax,
             1e3 * pct(lr, .5), 1e3 * pct(lr, .99), 1e3 * pct(le, .5), 1e3 * pct(le, .99), nio);
    printf("%s\n", buf);
    if (js) { FILE* f = fopen(js, "a"); fprintf(f, "%s\n", buf); fclose(f); }
    return 0;
}
