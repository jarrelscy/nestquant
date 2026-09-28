// NestQuant P4 upgrade engine (work items B2-B4): one engine per TP rank process, reading that rank's record file.
// Upgrade: record -> pinned host entry (io_uring + O_DIRECT, or a host LRU hit) -> cudaMemcpyAsync into the GPU slot
// -> stage row -> seq bump, all in order on one side stream. The captured mailbox kernel applies the row at the next
// replay, so an expert only goes live after its bytes landed; the compute stream never waits on any of this.
// Downgrade: stage row -> seq bump (no read). The host entries double as the bounce ring and the LRU cache of recent
// records (capacity >= qd); an entry is reusable once its H2D event completed and it is not pinned by an in-flight op.
//
// Python (torch extension):
//   e = Engine(path, rec_bytes, n_host, qd, device)
//   e.upgrade(tag, rec_index, slot_ptr, stage_ptr, row[int64 cpu], seq_ptr, seq)   # queued; returns immediately
//   e.post(tag, stage_ptr, row, seq_ptr, seq)                                       # downgrade / row-only op
//   e.poll() -> [(tag, hit, t_read_s, t_e2e_s)]  ops whose device writes are complete (the mailbox may apply them)
//   e.stats() -> dict ;  e.close()
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <liburing.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <time.h>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <list>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) TORCH_CHECK(false, "nqstream: ", cudaGetErrorString(e_), " at ", __LINE__); } while (0)
static double now() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
constexpr int ROW_W = 20;   // moe.py TBL_W

struct Op {
    int64_t tag, rec; bool read; uint64_t slot, stage, seqp; int seq; int64_t row[ROW_W];
    double t0 = 0, t_rd = 0; int entry = -1, ring = -1; bool hit = false; cudaEvent_t ev = nullptr;
};

struct Engine {
    int fd = -1, dev = 0, qd; size_t rb; int nh;
    uint8_t* host = nullptr;                     // nh entries of rb bytes (pinned, 4 KiB aligned)
    int64_t* rows = nullptr; int* seqs = nullptr; // pinned per-ring-slot row + seq staging (qd * 4 ring slots)
    int nring; std::vector<int> ring_free;
    std::vector<int64_t> ent_rec; std::vector<int> ent_pin;   // record held by an entry (-1 none), in-flight users
    std::list<int> lru; std::vector<std::list<int>::iterator> lru_it; std::unordered_map<int64_t, int> where;
    cudaStream_t st; io_uring ring; std::vector<cudaEvent_t> evpool;
    std::mutex mu; std::condition_variable cv; std::deque<Op*> q; std::vector<Op*> reading, copying;
    std::vector<std::tuple<int64_t, bool, double, double>> done; std::thread th; std::atomic<bool> stop{false};
    int64_t n_up = 0, n_post = 0, n_hit = 0, bytes = 0, n_inflight_read = 0;

    Engine(std::string path, int64_t rec_bytes, int64_t n_host, int64_t qd_, int64_t device) : dev(device), qd(qd_), rb(rec_bytes), nh(n_host)
    {
        TORCH_CHECK(rb % 4096 == 0 && nh >= qd && qd >= 1, "rec_bytes % 4096, n_host >= qd >= 1");
        fd = open(path.c_str(), O_RDONLY | O_DIRECT); TORCH_CHECK(fd >= 0, "open ", path);
        CK(cudaSetDevice(dev)); CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        CK(cudaHostAlloc((void**)&host, (size_t)nh * rb, cudaHostAllocPortable));
        nring = qd * 4; CK(cudaHostAlloc((void**)&rows, (size_t)nring * ROW_W * 8, cudaHostAllocPortable));
        CK(cudaHostAlloc((void**)&seqs, (size_t)nring * 4, cudaHostAllocPortable));
        for (int i = 0; i < nring; ++i) ring_free.push_back(i);
        ent_rec.assign(nh, -1); ent_pin.assign(nh, 0); lru_it.resize(nh);
        for (int i = 0; i < nh; ++i) { lru.push_back(i); lru_it[i] = std::prev(lru.end()); }
        TORCH_CHECK(io_uring_queue_init(qd, &ring, 0) == 0, "io_uring_queue_init");
        th = std::thread([this] { loop(); });
    }
    ~Engine() { close(); }
    void close()
    {
        if (stop.exchange(true)) return;
        cv.notify_all(); if (th.joinable()) th.join();
        cudaStreamSynchronize(st); io_uring_queue_exit(&ring); ::close(fd);
        for (auto e : evpool) cudaEventDestroy(e);
        cudaFreeHost(host); cudaFreeHost(rows); cudaFreeHost(seqs); cudaStreamDestroy(st);
    }
    void enqueue(Op* o) { { std::lock_guard<std::mutex> g(mu); q.push_back(o); } cv.notify_one(); }
    void upgrade(int64_t tag, int64_t rec, uint64_t slot, uint64_t stage, torch::Tensor row, uint64_t seqp, int64_t seq)
    {
        Op* o = new Op{tag, rec, true, slot, stage, seqp, (int)seq}; fill(o, row); enqueue(o);
    }
    void post(int64_t tag, uint64_t stage, torch::Tensor row, uint64_t seqp, int64_t seq)
    {
        Op* o = new Op{tag, -1, false, 0, stage, seqp, (int)seq}; fill(o, row); enqueue(o);
    }
    static void fill(Op* o, torch::Tensor row)
    {
        TORCH_CHECK(row.device().is_cpu() && row.scalar_type() == torch::kInt64 && row.numel() == ROW_W, "row: int64 cpu [20]");
        auto r = row.contiguous(); memcpy(o->row, r.data_ptr<int64_t>(), sizeof o->row); o->t0 = now();
    }
    std::vector<std::tuple<int64_t, bool, double, double>> poll()
    {
        std::lock_guard<std::mutex> g(mu); auto r = std::move(done); done.clear(); return r;
    }
    py::dict stats()
    {
        std::lock_guard<std::mutex> g(mu); py::dict d;
        d["upgrades"] = n_up; d["posts"] = n_post; d["host_hits"] = n_hit; d["bytes_read"] = bytes;
        d["queued"] = (int64_t)q.size(); d["reading"] = (int64_t)reading.size(); d["copying"] = (int64_t)copying.size();
        return d;
    }
    // ---- worker thread ----
    int take_entry(int64_t rec, bool& hit)    // caller holds no lock; entries are owned by this thread
    {
        auto it = where.find(rec);
        if (it != where.end()) { hit = true; int e = it->second; ent_pin[e]++; lru.splice(lru.end(), lru, lru_it[e]); return e; }
        hit = false;
        for (auto li = lru.begin(); li != lru.end(); ++li)
            if (ent_pin[*li] == 0)
            {
                int e = *li; if (ent_rec[e] >= 0) where.erase(ent_rec[e]);
                ent_rec[e] = -1; ent_pin[e] = 1; lru.splice(lru.end(), lru, li); return e;
            }
        return -1;
    }
    cudaEvent_t get_ev() { if (evpool.empty()) { cudaEvent_t e; CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming)); return e; } auto e = evpool.back(); evpool.pop_back(); return e; }
    void issue_copies(Op* o)   // bytes (upgrade) -> stage row -> seq, in order on the side stream
    {
        o->ring = ring_free.back(); ring_free.pop_back();
        int64_t* r = rows + (size_t)o->ring * ROW_W; memcpy(r, o->row, sizeof o->row); seqs[o->ring] = o->seq;
        if (o->read) CK(cudaMemcpyAsync((void*)o->slot, host + (size_t)o->entry * rb, rb, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync((void*)o->stage, r, ROW_W * 8, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync((void*)o->seqp, seqs + o->ring, 4, cudaMemcpyHostToDevice, st));
        o->ev = get_ev(); CK(cudaEventRecord(o->ev, st)); copying.push_back(o);
    }
    void loop()
    {
        CK(cudaSetDevice(dev));
        std::deque<Op*> wait_entry;      // ops waiting for a free host entry / ring slot
        std::deque<Op*> ready;           // reads landed, waiting for a ring slot
        while (!stop.load())
        {
            {
                std::unique_lock<std::mutex> g(mu);
                if (q.empty() && reading.empty() && copying.empty() && wait_entry.empty() && ready.empty())
                    cv.wait_for(g, std::chrono::milliseconds(2));
                while (!q.empty()) { wait_entry.push_back(q.front()); q.pop_front(); }
            }
            // retire completed copies
            for (size_t i = 0; i < copying.size();)
            {
                Op* o = copying[i];
                if (cudaEventQuery(o->ev) != cudaSuccess) { ++i; continue; }
                evpool.push_back(o->ev); ring_free.push_back(o->ring);
                if (o->read) ent_pin[o->entry]--;
                { std::lock_guard<std::mutex> g(mu); done.emplace_back(o->tag, o->hit, o->t_rd - o->t0, now() - o->t0); }
                delete o; copying[i] = copying.back(); copying.pop_back();
            }
            // start ops in arrival order (per-expert order matters: the host keeps one outstanding op per expert)
            bool sub = false;
            while (!ready.empty() && !ring_free.empty()) { issue_copies(ready.front()); ready.pop_front(); }
            while (ready.empty() && !wait_entry.empty() && !ring_free.empty())
            {
                Op* o = wait_entry.front();
                if (!o->read) { wait_entry.pop_front(); o->t_rd = o->t0; issue_copies(o); n_post++; continue; }
                if ((int)reading.size() >= qd) break;
                bool hit; int e = take_entry(o->rec, hit); if (e < 0) break;
                wait_entry.pop_front(); o->entry = e; o->hit = hit; n_up++;
                if (hit) { n_hit++; o->t_rd = now(); issue_copies(o); continue; }
                io_uring_sqe* sqe = io_uring_get_sqe(&ring);
                io_uring_prep_read(sqe, fd, host + (size_t)e * rb, rb, (uint64_t)o->rec * rb);
                io_uring_sqe_set_data(sqe, o); reading.push_back(o); sub = true;
            }
            if (sub) io_uring_submit(&ring);
            // reap reads
            io_uring_cqe* cqe; unsigned head; int n = 0;
            if (!reading.empty() && copying.empty() && wait_entry.empty())
            { __kernel_timespec to{0, 200000}; io_uring_wait_cqe_timeout(&ring, &cqe, &to); }
            io_uring_for_each_cqe(&ring, head, cqe)
            {
                Op* o = (Op*)io_uring_cqe_get_data(cqe); ++n;
                for (size_t i = 0; i < reading.size(); ++i) if (reading[i] == o) { reading[i] = reading.back(); reading.pop_back(); break; }
                if (cqe->res != (int)rb)
                {   // failed read: the expert stays at level 2 (row never posted); report with t_read < 0
                    ent_pin[o->entry]--; std::lock_guard<std::mutex> g(mu); done.emplace_back(o->tag, false, (double)std::min(cqe->res, -1), -1.0); delete o; continue;
                }
                o->t_rd = now(); ent_rec[o->entry] = o->rec; where[o->rec] = o->entry; bytes += rb;
                if (ring_free.empty()) ready.push_back(o); else issue_copies(o);
            }
            if (n) io_uring_cq_advance(&ring, n);
            if (!n && reading.empty() && !copying.empty()) usleep(20);   // copies in flight only: don't spin a core
        }
        // drain
        while (!reading.empty()) { io_uring_cqe* c; if (io_uring_wait_cqe(&ring, &c)) break; Op* o = (Op*)io_uring_cqe_get_data(c); io_uring_cqe_seen(&ring, c);
            for (size_t i = 0; i < reading.size(); ++i) if (reading[i] == o) { reading[i] = reading.back(); reading.pop_back(); break; } delete o; }
        cudaStreamSynchronize(st); for (auto o : copying) delete o; copying.clear();
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<Engine>(m, "Engine")
        .def(py::init<std::string, int64_t, int64_t, int64_t, int64_t>())
        .def("upgrade", &Engine::upgrade).def("post", &Engine::post)
        .def("poll", &Engine::poll).def("stats", &Engine::stats).def("close", &Engine::close);
}
