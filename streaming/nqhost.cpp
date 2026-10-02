// nq-io upgrade 5 (NQ_HOSTLOOP=cpp): the per-step host loop of the expert streaming, out of Python.
// Bit-exact ports (tests/test_hostloop_parity.py) of
//   SchedCore     the mechanical part of scheduler.Scheduler.step (score decay, EMA refresh, downs, candidate
//                 selection + stable ordering + budget/slot take) - operating in place on the Scheduler's numpy arrays;
//                 the predictor calls (P.step / P.target / P.order_score) stay in Python
//   FollowerCore  oplog.Follower / oplog.CoalescingFollower (record scan, in-order or net-effect replay, executor
//                 feedback, the NQ_FOLLOW_CHECK consistency check)
// Floating point: build with -ffp-contract=off (no FMA), so score*f+c rounds exactly like numpy's two ufuncs.
// Build: hostcore.py (plain pybind11 module from torch's bundled headers, no libtorch link).
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
namespace py = pybind11;
typedef py::array_t<double, 0> AF64;     // no forcecast: dtype must match (checked below)

template <class T> static T* buf(py::array& a, size_t n, const char* what, bool write)
{
    if (!py::isinstance<py::array_t<T>>(a)) throw std::invalid_argument(std::string(what) + ": wrong dtype");
    if (!(a.flags() & py::array::c_style)) throw std::invalid_argument(std::string(what) + ": not C-contiguous");
    if ((size_t)a.size() != n) throw std::invalid_argument(std::string(what) + ": wrong size");
    if (write && !a.writeable()) throw std::invalid_argument(std::string(what) + ": read-only");
    return (T*)(write ? a.mutable_data() : a.data());
}

// numpy argsort(kind='stable') order of the key -x (ascending): NaN last, ties keep index order
template <class T> static bool neg_less(T a, T b)
{
    T na = -a, nb = -b;
    if (std::isnan(nb)) return !std::isnan(na);
    if (std::isnan(na)) return false;
    return na < nb;
}

// strict total order of a stable argsort of -x: (neg_less key, then index)
template <class T, class I> static bool stable_less(const T* x, I a, I b)
{
    if (neg_less(x[a], x[b])) return true;
    if (neg_less(x[b], x[a])) return false;
    return a < b;
}

// Python float floor division (floatobject.c float_floor_div / _float_div_mod)
static double py_floordiv(double vx, double wx)
{
    double mod = std::fmod(vx, wx), div = (vx - mod) / wx;
    if (mod) { if ((wx < 0) != (mod < 0)) { mod += wx; div -= 1.0; } }
    double fd;
    if (div) { fd = std::floor(div); if (div - fd > 0.5) fd += 1.0; }
    else fd = std::copysign(0.0, vx / wx);
    return fd;
}

struct SchedCore
{
    int NL, NE; size_t N; std::vector<int> layers; std::vector<int32_t> cand;
    SchedCore(std::vector<int> L, int ne) : NL((int)L.size()), NE(ne), N((size_t)L.size() * ne), layers(L) {}
    py::tuple key(size_t k) const { return py::make_tuple(layers[k / NE], (int)(k % NE)); }
    // score = score * f + c   (f = Scheduler.a ** ntok, computed in Python)
    void decay(py::array score, py::array c, double f)
    {
        double* s = buf<double>(score, N, "score", true); const double* x = buf<double>(c, N, "counts", false);
        for (size_t i = 0; i < N; ++i) { double t = s[i] * f; s[i] = t + x[i]; }
    }
    // EMA refresh: layers with any count keep the top n_float non-fixed experts by score (argsort(-sc, stable))
    void ema_refresh(py::array score, py::array fixed, py::array want, int nf)
    {
        const double* s = buf<double>(score, N, "score", false); const bool* fx = buf<bool>(fixed, N, "fixed", false);
        bool* w = buf<bool>(want, N, "want", true);
        std::vector<int> o(NE); std::vector<double> sc(NE);
        for (int i = 0; i < NL; ++i)
        {
            const double* r = s + (size_t)i * NE; bool has = false;
            for (int e = 0; e < NE; ++e) { has |= r[e] > 0; sc[e] = fx[(size_t)i * NE + e] ? -INFINITY : r[e]; o[e] = e; }
            if (!has) continue;
            // the first nf of a stable sort = the nf smallest under (key, index): selection, no full sort
            int m = std::min(nf, NE);
            if (m < NE) std::nth_element(o.begin(), o.begin() + m, o.end(), [&](int a, int b) { return stable_less(sc.data(), a, b); });
            bool* wr = w + (size_t)i * NE;
            for (int e = 0; e < NE; ++e) wr[e] = false;
            for (int k = 0; k < m; ++k) wr[o[k]] = true;
        }
    }
    py::array_t<bool> resident(py::array state)                  // np.isin(state, (1, 2))
    {
        const int8_t* st = buf<int8_t>(state, N, "state", false);
        py::array_t<bool> r({NL, NE}); bool* p = r.mutable_data();
        for (size_t i = 0; i < N; ++i) p[i] = st[i] == 1 || st[i] == 2;
        return r;
    }
    long count_busy(py::array state)                               // int((state > 0).sum())
    {
        const int8_t* st = buf<int8_t>(state, N, "state", false); long n = 0;
        for (size_t i = 0; i < N; ++i) n += st[i] > 0;
        return n;
    }
    py::list downs(py::array state, py::array want)                // (state == 2) & ~want -> state 3, row-major
    {
        int8_t* st = buf<int8_t>(state, N, "state", true); const bool* w = buf<bool>(want, N, "want", false);
        py::list out;
        for (size_t i = 0; i < N; ++i) if (st[i] == 2 && !w[i]) { out.append(key(i)); st[i] = 3; }
        return out;
    }
    // big step guard + candidates (state == 0) & want & (hold <= tok), kept for take(); -> (big, any candidate)
    py::tuple select(py::array c, py::array state, py::array want, py::array hold, double tok, double big_thr)
    {
        const double* x = buf<double>(c, N, "counts", false); const int8_t* st = buf<int8_t>(state, N, "state", false);
        const bool* w = buf<bool>(want, N, "want", false); const double* h = buf<double>(hold, N, "hold", false);
        bool big = false;
        for (int i = 0; i < NL && !big; ++i)
        {
            long n = 0; const double* r = x + (size_t)i * NE;
            for (int e = 0; e < NE; ++e) n += r[e] > 0;
            big = (double)n > big_thr;
        }
        cand.clear();
        for (size_t i = 0; i < N; ++i) if (st[i] == 0 && w[i] && h[i] <= tok) cand.push_back((int32_t)i);
        return py::make_tuple(big, !cand.empty());
    }
    // order the candidates by -key (stable), take the first n (state -> 1); -> (ups, deferred = more than n candidates)
    template <class T> py::tuple take_t(const T* k, py::array state, long n)
    {
        int8_t* st = buf<int8_t>(state, N, "state", true);
        std::vector<int32_t> o(cand);                              // cand is in index order
        if (n < 0) n = 0;
        size_t m = std::min((size_t)n, o.size()); py::list ups;
        // first m of the stable order: partial sort under (key, index) (== stable_sort's prefix)
        std::partial_sort(o.begin(), o.begin() + m, o.end(), [&](int32_t a, int32_t b) { return stable_less(k, a, b); });
        for (size_t j = 0; j < m; ++j) { ups.append(key(o[j])); st[o[j]] = 1; }
        return py::make_tuple(ups, o.size() > (size_t)n);
    }
    py::object take(py::array key_, py::array state, long n)
    {
        if (py::isinstance<py::array_t<double>>(key_)) return take_t(buf<double>(key_, N, "key", false), state, n);
        if (py::isinstance<py::array_t<float>>(key_)) return take_t(buf<float>(key_, N, "key", false), state, n);
        return py::none();                                         // other dtype: caller falls back to numpy
    }
};

static const int32_t MAGIC = 0x4E514F50;

struct FollowerCore
{
    int NL, NE; size_t N; bool coal; std::vector<int> layers, li;
    std::vector<uint8_t> busy, up, chk, rerr, creq, held, inp; std::vector<int8_t> plv;
    std::vector<int32_t> nxt, prv; int32_t head = -1, tail = -1; long npend = 0;   // coalescing: ordered pending set
    std::vector<std::pair<int32_t, int8_t>> q;                                  // in-order: op queue
    long nbusy = 0; bool check_on = false;
    long ups = 0, downs_ = 0, dropped = 0, read_errors = 0, log_ops = 0, merged = 0, moot = 0, cancel_req = 0, cancelled_ = 0,
         check_ok = 0, check_bad = 0;
    std::string check_msg;
    FollowerCore(std::vector<int> L, int ne, bool coalesce) : NL((int)L.size()), NE(ne), N((size_t)L.size() * ne), coal(coalesce), layers(L)
    {
        int mx = 0; for (int x : L) mx = std::max(mx, x);
        li.assign(mx + 1, -1); for (int i = 0; i < NL; ++i) li[L[i]] = i;
        busy.assign(N, 0); up.assign(N, 0); chk.assign(N, 0); rerr.assign(N, 0); creq.assign(N, 0); held.assign(N, 0);
        inp.assign(N, 0); plv.assign(N, 0); nxt.assign(N, -1); prv.assign(N, -1);
    }
    size_t idx(int L, int E) const
    {
        if (L < 0 || L >= (int)li.size() || li[L] < 0 || E < 0 || E >= NE) throw std::out_of_range("expert (" + std::to_string(L) + "," + std::to_string(E) + ")");
        return (size_t)li[L] * NE + E;
    }
    py::tuple key(size_t k) const { return py::make_tuple(layers[k / NE], (int)(k % NE)); }
    void setb(size_t k, bool v) { if (busy[k] != v) { busy[k] = v; nbusy += v ? 1 : -1; } }
    // --- coalescing pending list
    void unlink(int32_t k)
    {
        if (prv[k] >= 0) nxt[prv[k]] = nxt[k]; else head = nxt[k];
        if (nxt[k] >= 0) prv[nxt[k]] = prv[k]; else tail = prv[k];
        prv[k] = nxt[k] = -1; inp[k] = 0; --npend;
    }
    void push(int32_t k, int8_t lv)
    {
        prv[k] = tail; nxt[k] = -1; if (tail >= 0) nxt[tail] = k; else head = k; tail = k; inp[k] = 1; plv[k] = lv; ++npend;
    }
    void addop(int32_t k, int8_t lv)
    {
        if (coal) { ++log_ops; if (inp[k]) { ++merged; unlink(k); } push(k, lv); }
        else q.emplace_back(k, lv);
    }
    void fold(int32_t k, bool isup) { if (check_on) { chk[k] = isup; rerr[k] = 0; } }
    // scan complete records of an int32 log buffer -> (int32 words consumed, rotation marker hit)
    py::tuple feed(py::array a)
    {
        if (!py::isinstance<py::array_t<int32_t>>(a) || !(a.flags() & py::array::c_style)) throw std::invalid_argument("feed: int32 C array");
        const int32_t* p = (const int32_t*)a.data(); size_t n = (size_t)a.size(), i = 0; bool rot = false;
        while (i + 3 <= n)
        {
            if (p[i] != MAGIC) throw std::runtime_error("oplog: bad record at word " + std::to_string(i));
            int32_t nu = p[i + 1], nd = p[i + 2];
            if (nu < 0) { rot = true; i += 3; break; }
            size_t j = i + 3 + 2 * (size_t)(nu + nd);
            if (j > n) break;                                        // record still being written
            const int32_t* r = p + i + 3;
            std::vector<int32_t> u(nu), d(nd);
            for (int32_t m = 0; m < nu; ++m) u[m] = (int32_t)idx(r[2 * m], r[2 * m + 1]);
            for (int32_t m = 0; m < nd; ++m) d[m] = (int32_t)idx(r[2 * (nu + m)], r[2 * (nu + m) + 1]);
            for (int32_t k : d) fold(k, false);                     // Follower._fold: downs, then ups
            for (int32_t k : u) fold(k, true);
            for (int32_t k : d) addop(k, 2);
            for (int32_t k : u) addop(k, 4);
            i = j;
        }
        return py::make_tuple((long)i, rot);
    }
    // one replay step (issue=True); -> (ups, downs, cancel requests) for the caller to execute: cancels first, then apply
    py::tuple step()
    {
        py::list U, D, C;
        if (!coal)
        {
            if (q.empty()) return py::make_tuple(U, D, C);
            std::vector<std::pair<int32_t, int8_t>> keep; std::vector<int32_t> touched;
            for (auto& op : q)
            {
                int32_t k = op.first;
                if (held[k] || busy[k]) { keep.push_back(op); if (!held[k]) { held[k] = 1; touched.push_back(k); } continue; }
                held[k] = 1; touched.push_back(k);
                if (op.second == 2) { if (!up[k]) { ++dropped; continue; } D.append(key(k)); }
                else U.append(key(k));
                setb(k, true);
            }
            for (int32_t k : touched) held[k] = 0;
            q.swap(keep);
        }
        else
        {
            if (!npend) return py::make_tuple(U, D, C);
            for (int32_t k = head; k >= 0;)
            {
                int32_t nx = nxt[k]; int8_t lv = plv[k];
                if (busy[k])
                {
                    if (lv == 2 && !up[k] && !creq[k]) { creq[k] = 1; C.append(key(k)); }
                    k = nx; continue;
                }
                unlink(k); creq[k] = 0;
                if ((up[k] ? 4 : 2) == lv) { ++moot; k = nx; continue; }
                (lv == 2 ? D : U).append(key(k)); setb(k, true);
                k = nx;
            }
        }
        ups += (long)py::len(U); downs_ += (long)py::len(D);
        return py::make_tuple(U, D, C);
    }
    // executor feedback
    void landed(int L, int E) { size_t k = idx(L, E); setb(k, false); up[k] = 1; if (check_on) rerr[k] = 0; }
    void released(int L, int E) { size_t k = idx(L, E); setb(k, false); up[k] = 0; }
    void failed(int L, int E, bool read_error) { size_t k = idx(L, E); setb(k, false); if (read_error) { ++read_errors; if (check_on) rerr[k] = 1; } }
    void cancelled(int L, int E) { size_t k = idx(L, E); setb(k, false); ++cancelled_; }
    void mark_busy(std::vector<std::pair<int, int>> ks) { for (auto& p : ks) setb(idx(p.first, p.second), true); }
    void enable_check(std::vector<std::pair<int, int>> init)
    {
        check_on = true; std::fill(chk.begin(), chk.end(), 0); std::fill(rerr.begin(), rerr.end(), 0);
        for (auto& p : init) chk[idx(p.first, p.second)] = 1;
        check_ok = check_bad = 0; check_msg.clear();
    }
    long backlog() const { return coal ? npend : (long)q.size(); }
    py::object check()
    {
        if (!check_on || backlog() || nbusy) return py::none();
        long miss = 0, extra = 0; std::string mm, ex; int nm = 0, nx = 0;
        for (size_t k = 0; k < N; ++k)
        {
            bool w = chk[k] && !rerr[k];
            if (w && !up[k]) { ++miss; if (nm++ < 4) mm += "(" + std::to_string(layers[k / NE]) + ", " + std::to_string(k % NE) + ") "; }
            if (!w && up[k]) { ++extra; if (nx++ < 4) ex += "(" + std::to_string(layers[k / NE]) + ", " + std::to_string(k % NE) + ") "; }
        }
        bool ok = !miss && !extra;
        if (ok) ++check_ok; else { ++check_bad; if (check_msg.empty()) check_msg = "missing [" + mm + "] extra [" + ex + "]"; }
        return py::make_tuple(ok, miss, extra);
    }
    long level_count() const { long n = 0; for (auto v : up) n += v; return n; }
    py::array_t<bool> up_view(py::object self)                     // [NL, NE] bool view of the landed set (no copy)
    {
        return py::array_t<bool>({NL, NE}, {(py::ssize_t)NE, (py::ssize_t)1}, (const bool*)up.data(), self);
    }
    std::vector<std::pair<int, int>> up_list() const
    {
        std::vector<std::pair<int, int>> r;
        for (size_t k = 0; k < N; ++k) if (up[k]) r.emplace_back(layers[k / NE], (int)(k % NE));
        return r;
    }
    py::dict stats() const
    {
        py::dict d; d["ups"] = ups; d["downs"] = downs_; d["dropped"] = dropped; d["read_errors"] = read_errors;
        if (coal) { d["log_ops"] = log_ops; d["merged"] = merged; d["moot"] = moot; d["cancel_req"] = cancel_req; d["cancelled"] = cancelled_; }
        if (check_on) { d["check_ok"] = check_ok; d["check_bad"] = check_bad; }
        return d;
    }
};

PYBIND11_MODULE(nqhost, m)
{
    m.def("py_floordiv", &py_floordiv);
    py::class_<SchedCore>(m, "SchedCore")
        .def(py::init<std::vector<int>, int>())
        .def("decay", &SchedCore::decay).def("ema_refresh", &SchedCore::ema_refresh).def("resident", &SchedCore::resident)
        .def("count_busy", &SchedCore::count_busy).def("downs", &SchedCore::downs).def("select", &SchedCore::select)
        .def("take", &SchedCore::take);
    py::class_<FollowerCore>(m, "FollowerCore")
        .def(py::init<std::vector<int>, int, bool>())
        .def("feed", &FollowerCore::feed).def("step", &FollowerCore::step)
        .def("landed", &FollowerCore::landed).def("released", &FollowerCore::released).def("failed", &FollowerCore::failed)
        .def("cancelled", &FollowerCore::cancelled).def("mark_busy", &FollowerCore::mark_busy)
        .def("enable_check", &FollowerCore::enable_check).def("check", &FollowerCore::check).def("backlog", &FollowerCore::backlog)
        .def("level_count", &FollowerCore::level_count)
        .def("up_view", [](py::object self) { return self.cast<FollowerCore&>().up_view(self); })
        .def("up_list", &FollowerCore::up_list).def("stats", &FollowerCore::stats)
        .def("is_up", [](FollowerCore& f, int L, int E) { return (bool)f.up[f.idx(L, E)]; })
        .def("is_busy", [](FollowerCore& f, int L, int E) { return (bool)f.busy[f.idx(L, E)]; })
        .def("add_cancel_req", [](FollowerCore& f) { ++f.cancel_req; })
        .def_readonly("check_msg", &FollowerCore::check_msg);
}
