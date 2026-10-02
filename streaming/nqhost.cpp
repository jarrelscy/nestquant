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
#include <deque>
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

// ---------------------------------------------------------------- TapCore (nq-tapc, NQ_SCHED=tap + NQ_HOSTLOOP=cpp)
// Bit-exact port of scheduler_tap.TapScheduler.step's array work: lazy-eviction bookkeeping (doom), the no-slot queue
// (todo), the big-step guard, the two-horizon value V / V0 (float32, numpy NEP 50 promotion: python floats are cast to
// float32 before each op, sums are numpy's pairwise float32 reduction), the per-layer pairing (stable argsorts), the
// gain-sorted issue loop with the issue budget, and the want mask. The live rate / latency estimate (io_all, clock)
// stays in Python and comes in as scalars (lat, H, budget).

// numpy FLOAT_pairwise_sum (numpy/_core/src/umath/loops_utils.h.src), unit stride
static float np_pairwise_f32(const float* a, long n)
{
    if (n < 8) { float r = -0.0f; for (long i = 0; i < n; ++i) r += a[i]; return r; }
    if (n <= 128)
    {
        float r[8]; for (int j = 0; j < 8; ++j) r[j] = a[j];
        long i;
        for (i = 8; i < n - (n % 8); i += 8) for (int j = 0; j < 8; ++j) r[j] += a[i + j];
        float res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    long n2 = n / 2; n2 -= n2 % 8;
    return np_pairwise_f32(a, n2) + np_pairwise_f32(a + n2, n - n2);
}
// np.add.reduce of a contiguous float32 row (axis=-1, numpy 2.x): the whole row pairwise (checked vs numpy 2.3/2.5)
static float np_rowsum_f32(const float* a, long n) { return n <= 0 ? 0.0f : np_pairwise_f32(a, n); }
// numpy argsort(kind='stable') ascending order of x: NaN last, ties by index
static bool asc_less(float a, float b)
{
    if (std::isnan(b)) return !std::isnan(a);
    if (std::isnan(a)) return false;
    return a < b;
}

struct TapCore
{
    int NL, NE; size_t N; std::vector<int> layers;
    std::vector<std::pair<int32_t, int32_t>> doom;   // (e, v) flat ids in insertion order (TapScheduler.doom dict)
    std::deque<int32_t> todo;                         // flat ids (TapScheduler.todo)
    std::vector<int32_t> U, D;                        // this step's ups / downs (flat), in TapScheduler order
    std::vector<double> V, V0; bool V64 = false, V064 = false;   // value arrays (float32 values stored exactly unless V64)
    long nfree = 0; bool big = false;
    TapCore(std::vector<int> L, int ne) : NL((int)L.size()), NE(ne), N((size_t)L.size() * ne), layers(L), V(N), V0(N) {}
    py::tuple key(size_t k) const { return py::make_tuple(layers[k / NE], (int)(k % NE)); }
    // start of step(), after the score decay and P.step: doom resolution, todo issue, big-step guard. -> big
    bool pre(py::array state, py::array doomed, py::array c, long slots, double big_thr)
    {
        int8_t* st = buf<int8_t>(state, N, "state", true); bool* dm = buf<bool>(doomed, N, "doomed", true);
        const double* x = buf<double>(c, N, "counts", false);
        U.clear(); D.clear();
        size_t w = 0;
        for (size_t r = 0; r < doom.size(); ++r)
        {
            int32_t e = doom[r].first, v = doom[r].second;
            if (st[e] == 2) { dm[v] = false; if (st[v] == 2) { D.push_back(v); st[v] = 3; } }
            else if (st[e] == 0) dm[v] = false;
            else doom[w++] = doom[r];
        }
        doom.resize(w);
        if (slots >= 0) { long b = 0; for (size_t i = 0; i < N; ++i) b += st[i] > 0; nfree = slots - b; }
        else nfree = 1000000000L;
        while (!todo.empty() && nfree > 0)
        {
            int32_t k = todo.front(); todo.pop_front();
            if (st[k] == 0) { U.push_back(k); st[k] = 1; --nfree; }
        }
        big = false;
        for (int i = 0; i < NL && !big; ++i)
        {
            long n = 0; const double* r = x + (size_t)i * NE;
            for (int e = 0; e < NE; ++e) n += r[e] > 0;
            big = (double)n > big_thr;
        }
        return big;
    }
    long queued(py::array state)                      // int((state == 1).sum()) + len(todo)
    {
        const int8_t* st = buf<int8_t>(state, N, "state", false); long n = 0;
        for (size_t i = 0; i < N; ++i) n += st[i] == 1;
        return n + (long)todo.size();
    }
    // numpy promotion of one window term rn * w (rn float32 array): w a numpy float64 scalar (strong) -> float64 math,
    // w a python float (weak) -> cast to float32 first
    static double term(float r, double w, bool strong) { return strong ? (double)r * w : (double)(r * (float)w); }
    // (rn * n + rf * f) * nrm, numpy dtype rules; -> value as double (exact float32 value when the result dtype is float32)
    static double win_val(float rn, float rf, double n, bool sn, double f, bool sf, float nrm)
    {
        double a = term(rn, n, sn), b = term(rf, f, sf);
        if (sn || sf) return (a + b) * (double)nrm;
        float t = (float)a + (float)b; return (double)(t * nrm);
    }
    // TapScheduler._value: V, V0. tail < 0: far = EMA rate (score * (1 - a)) rescaled to jF's mass; else tail x near.
    // w = (near, far) of [lat, lat + H) and of [0, lat) with their numpy-strong flags (computed in Python, same expressions)
    void value(const float* S0, const double* score, const bool* fx, double one_minus_a, double tail, double span,
               double n1, bool s1n, double f1, bool s1f, double n0, bool s0n, double f0, bool s0f, bool v0)
    {
        std::vector<float> S(NE), rn(NE), rf(NE);
        const float spanf = (float)span;
        V64 = s1n || s1f; V064 = v0 ? (s0n || s0f) : V64;          // V0 = zeros_like(V) when lat < 1
        for (int l = 0; l < NL; ++l)
        {
            size_t o = (size_t)l * NE;
            for (int e = 0; e < NE; ++e) { float x = S0[o + e]; float y = std::isnan(x) ? x : std::fmax(x, 0.0f); S[e] = fx[o + e] ? 0.0f : y; }
            float mass = np_rowsum_f32(S.data(), NE); bool ok = mass > 0; float m = ok ? mass : 1.0f;
            for (int e = 0; e < NE; ++e) rn[e] = S[e] / spanf;
            if (tail >= 0) { float t = (float)tail; for (int e = 0; e < NE; ++e) rf[e] = t * rn[e]; }
            else
            {
                for (int e = 0; e < NE; ++e) { double t = score[o + e] * one_minus_a; rf[e] = fx[o + e] ? 0.0f : (float)t; }
                float rs = np_rowsum_f32(rf.data(), NE);
                float f = rs > 0 ? (m / spanf) / rs : 0.0f;
                for (int e = 0; e < NE; ++e) rf[e] = rf[e] * f;
            }
            float nrm = 512.0f / m;
            for (int e = 0; e < NE; ++e)
            {
                V[o + e] = ok ? win_val(rn[e], rf[e], n1, s1n, f1, s1f, nrm) : 0.0;
                V0[o + e] = (v0 && ok) ? win_val(rn[e], rf[e], n0, s0n, f0, s0f, nrm) : 0.0;
            }
        }
    }
    // numpy scalar ops of a V element (dtype float32 unless 64) with a python float
    static bool gt(double x, double c, bool d64) { return d64 ? x > c : (float)x > (float)c; }
    static double sub(double x, double y, bool d64) { return d64 ? x - y : (double)((float)x - (float)y); }
    // refresh part of step(): value, pin, pairs, issue loop. -> (promotions, budget_cut, eager_evict, no_slot_skip)
    py::tuple refresh(py::array Sa, py::array score_, py::array fixed, py::array state, py::array doomed, py::array hold,
                      py::object pin_, double tok, long nf, double one_minus_a, double tail, double span, py::tuple w,
                      double tc, long budget)
    {
        const float* S0 = buf<float>(Sa, N, "S", false); const double* score = buf<double>(score_, N, "score", false);
        const bool* fx = buf<bool>(fixed, N, "fixed", false); int8_t* st = buf<int8_t>(state, N, "state", true);
        bool* dm = buf<bool>(doomed, N, "doomed", true); const double* hd = buf<double>(hold, N, "hold", false);
        value(S0, score, fx, one_minus_a, tail, span, w[0].cast<double>(), w[1].cast<bool>(), w[2].cast<double>(), w[3].cast<bool>(),
              w[4].cast<double>(), w[5].cast<bool>(), w[6].cast<double>(), w[7].cast<bool>(), w[8].cast<bool>());
        if (!pin_.is_none())
        {
            py::array pa = pin_.cast<py::array>(); const bool* pn = buf<bool>(pa, N, "pin", false);
            for (size_t i = 0; i < N; ++i) if (pn[i] && !fx[i]) V[i] = 1e9;
        }
        const bool d = V64, d0 = V064;
        struct P { double g; int32_t l, e, v; };
        std::vector<P> out; std::vector<int32_t> ce, rv;
        for (int l = 0; l < NL; ++l)
        {
            size_t o = (size_t)l * NE; ce.clear(); rv.clear(); long oc = 0;
            for (int e = 0; e < NE; ++e)
            {
                size_t i = o + e; int8_t s = st[i];
                if (fx[i] || dm[i]) continue;
                if (s == 1 || s == 2) ++oc;
                if (s == 2) rv.push_back(e);
                if (s == 0 && hd[i] <= tok) ce.push_back(e);
            }
            const double* Vl = V.data() + o;
            // numpy: argsort(-where(cand, V, -inf)) / argsort(where(res, V, inf)), stable; the members sort first unless a
            // member key is NaN (sorts after the +-inf fill) or equals the fill -> then rank all NE like numpy
            bool odd = false;
            for (int32_t e : ce) odd |= std::isnan(Vl[e]) || Vl[e] == -INFINITY;
            for (int32_t e : rv) odd |= std::isnan(Vl[e]) || Vl[e] == INFINITY;
            long ncand = (long)ce.size(), nres = (long)rv.size();
            auto alt = [](double a, double b) { if (std::isnan(b)) return !std::isnan(a); if (std::isnan(a)) return false; return a < b; };
            if (!odd)
            {
                std::stable_sort(ce.begin(), ce.end(), [&](int32_t a, int32_t b) { return neg_less(Vl[a], Vl[b]); });
                std::stable_sort(rv.begin(), rv.end(), [&](int32_t a, int32_t b) { return alt(Vl[a], Vl[b]); });
            }
            else
            {
                std::vector<double> kc(NE), kr(NE); std::vector<char> ic(NE, 0), ir(NE, 0);
                for (int32_t e : ce) ic[e] = 1;
                for (int32_t e : rv) ir[e] = 1;
                for (int e = 0; e < NE; ++e) { kc[e] = ic[e] ? Vl[e] : -INFINITY; kr[e] = ir[e] ? Vl[e] : INFINITY; }
                ce.resize(NE); rv.resize(NE);
                for (int e = 0; e < NE; ++e) ce[e] = rv[e] = e;
                std::stable_sort(ce.begin(), ce.end(), [&](int32_t a, int32_t b) { return neg_less(kc[a], kc[b]); });
                std::stable_sort(rv.begin(), rv.end(), [&](int32_t a, int32_t b) { return alt(kr[a], kr[b]); });
            }
            long free = nf - oc, k = 0;
            while (k < ncand)
            {
                int32_t e = ce[k]; double ve = Vl[e];
                if (k < free)
                {
                    if (gt(ve, tc, d)) { out.push_back({ve, l, e, -1}); ++k; continue; }
                    break;
                }
                long j = k - std::max(free, 0L);
                if (j >= nres) break;
                int32_t v = rv[j]; double g = sub(ve, Vl[v], d);
                if (d ? g <= tc : (float)g <= (float)tc) break;
                out.push_back({g, l, e, v}); ++k;
            }
        }
        // out.sort(key=lambda z: -z[0]) (python stable sort; z[0] = float(g))
        std::stable_sort(out.begin(), out.end(), [](const P& a, const P& b) { return -a.g < -b.g; });
        long k = 0, cut = 0, eager = 0, skip = 0;
        for (const P& p : out)
        {
            if (k >= budget) { ++cut; continue; }
            int32_t ie = p.l * NE + p.e;
            if (p.v < 0)
            {
                if (nfree > 0) { U.push_back(ie); st[ie] = 1; --nfree; }
                else todo.push_back(ie);
            }
            else
            {
                int32_t iv = p.l * NE + p.v;
                if (nfree > 0) { U.push_back(ie); st[ie] = 1; --nfree; doom.emplace_back(ie, iv); dm[iv] = true; }
                else if (gt(sub(p.g, V0[iv], d0), tc, d0)) { D.push_back(iv); st[iv] = 3; todo.push_back(ie); ++eager; }
                else { ++skip; continue; }
            }
            ++k;
        }
        return py::make_tuple(k, cut, eager, skip);
    }
    // end of step(): want = ((state == 1) | (state == 2)) & ~fixed; -> (ups, downs) of the step
    py::tuple finish(py::array state, py::array fixed, py::array want)
    {
        const int8_t* st = buf<int8_t>(state, N, "state", false); const bool* fx = buf<bool>(fixed, N, "fixed", false);
        bool* w = buf<bool>(want, N, "want", true);
        for (size_t i = 0; i < N; ++i) w[i] = (st[i] == 1 || st[i] == 2) && !fx[i];
        py::list u, d;
        for (int32_t k : U) u.append(key(k));
        for (int32_t k : D) d.append(key(k));
        return py::make_tuple(u, d);
    }
    // copies of the last value arrays in TapScheduler._value's dtype (tests)
    py::object vcopy(const std::vector<double>& x, bool d64) const
    {
        if (d64) { py::array_t<double> r({NL, NE}); std::copy(x.begin(), x.end(), r.mutable_data()); return r; }
        py::array_t<float> r({NL, NE}); float* p = r.mutable_data(); for (size_t i = 0; i < N; ++i) p[i] = (float)x[i]; return r;
    }
    std::vector<std::pair<int, int>> todo_list() const { std::vector<std::pair<int, int>> r; for (int32_t k : todo) r.emplace_back(k / NE, k % NE); return r; }
    std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> doom_list() const
    {
        std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> r;
        for (auto& p : doom) r.push_back({{p.first / NE, p.first % NE}, {p.second / NE, p.second % NE}});
        return r;
    }
    long todo_len() const { return (long)todo.size(); }
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
    m.def("rowsum_f32", [](py::array a, long n) { return np_rowsum_f32(buf<float>(a, (size_t)n, "a", false), n); });
    py::class_<SchedCore>(m, "SchedCore")
        .def(py::init<std::vector<int>, int>())
        .def("decay", &SchedCore::decay).def("ema_refresh", &SchedCore::ema_refresh).def("resident", &SchedCore::resident)
        .def("count_busy", &SchedCore::count_busy).def("downs", &SchedCore::downs).def("select", &SchedCore::select)
        .def("take", &SchedCore::take);
    py::class_<TapCore>(m, "TapCore")
        .def(py::init<std::vector<int>, int>())
        .def("pre", &TapCore::pre).def("queued", &TapCore::queued).def("refresh", &TapCore::refresh).def("finish", &TapCore::finish)
        .def("V", [](TapCore& t) { return t.vcopy(t.V, t.V64); }).def("V0", [](TapCore& t) { return t.vcopy(t.V0, t.V064); })
        .def("todo_list", &TapCore::todo_list).def("doom_list", &TapCore::doom_list).def("todo_len", &TapCore::todo_len);
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
