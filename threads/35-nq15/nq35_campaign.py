"""T35 full-model encode driver (pattern-rate base, e.g. b175 = base 1.75 + residual 2.25/2.25/2.5625).

  python nq35_campaign.py run --cfg b175 --root /tmp/nestquant/35-nq15/enc_b175 [--gpus 0,1,2,3,4,5,6,7] [--wpg 1]
                          [--order 3-10,40-50,11-39,51-77] [--hold-min 85] [--pause-s 120] [--max-holds 0]
  python nq35_campaign.py status --root R

Chained gpu.lock holds: take flock(/tmp/nestquant/33-search/gpu.lock), run until --hold-min minutes, stop launching
work that would not finish in time, kill only this driver's own children at the deadline (their process groups),
release, pause, re-acquire.  Everything restarts from filesystem state:
  encoded(L,E)  = R/L{L}/experts/E{E}.pt (T12 nq_layer saves atomically and skips existing experts)
  finalized(L)  = R/fin/L{L}.json rc 0   (nq35_fin.py = T25 nq25_finalize: nq_layer finalize, safetensors convert +
                                          round trip, artifact-from-st == E{E}.pt raw bytes, decode on 8 experts)
  spot(L)       = R/spot/L{L}.json       (nq35_spot.py = T25 nq25_spot, L2 vs EXL3-at-base-K 5%, L4 vs EXL3-4 2%)
Layers are encoded only once their text and vision stats are restored (stats/L{L} link present).
Encode chunks are sized to the time left in the hold from the observed s/expert.  No uploads of any kind.
"""
import os, sys, json, time, argparse, fcntl, signal, subprocess, re, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/home/coder/git/glm52/.venv/bin/python"
NQ = "/home/coder/git/nestquant"
sys.path.insert(0, HERE)
LOCK = "/tmp/nestquant/33-search/gpu.lock"
TXT = "/tmp/nestquant/19-capture-glmfmt"
MM = "/tmp/nestquant/19-capture-mm"
SRC = "/tmp/nestquant/src/glm53-fp8"
FIXED = f"{TXT}/fixed_set.json"
NEXP = 256
CFGS = {"b175": dict(base_K=1.75, res_k="2.25,2.25,2.5625"), "b15": dict(base_K=1.5, res_k="2.5,2.5,2.8125"),
        "b20": dict(base_K=2.0, res_k=None)}


def worker_env(omp, base_K):
    """T25 nq25_campaign.worker_env: cuda-compat libcuda (the box driver is older than the venv's CUDA), T06 lib,
    harness + orbit-duet on PYTHONPATH; no HF tokens in workers."""
    env = dict(os.environ)
    compat = "/home/coder/git/glm52/artifacts/shared-bit-graphs/runtime/cuda-compat/usr/local/cuda-13.0/compat"
    lib06 = f"{NQ}/threads/06-expert-objective/lib"
    env["LD_LIBRARY_PATH"] = ":".join(x for x in (lib06, compat, env.get("LD_LIBRARY_PATH", "")) if x)
    env["PYTHONPATH"] = ":".join(x for x in (f"{NQ}/threads/05-exl3-harness", "/home/coder/git/orbit-duet",
                                             env.get("PYTHONPATH", "")) if x)
    env.update(OMP_NUM_THREADS=str(omp), MKL_NUM_THREADS=str(omp), OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1",
               CUDA_DEVICE_ORDER="PCI_BUS_ID", NQ35_BASE_K=str(base_K))
    for k in [k for k in env if k.startswith("HF_TOKEN") or k == "HUGGING_FACE_HUB_TOKEN"]:
        env.pop(k)
    return env


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + "Z"


def jload(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def jdump(o, p):
    json.dump(o, open(p + ".tmp", "w"), indent=1); os.replace(p + ".tmp", p)


def parse_order(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    assert len(out) == len(set(out)), "duplicate layers in --order"
    return out


class Campaign:
    def __init__(self, a):
        self.a = a
        self.root = a.root
        self.cfg = CFGS[a.cfg]
        self.order = parse_order(a.order)
        self.gpus = [int(x) for x in a.gpus.split(",")]
        for d in ("logs", "fin", "spot", "fail", "timing"):
            os.makedirs(f"{self.root}/{d}", exist_ok=True)
        self.jobs = []                    # dict(kind, L, gpu, p, t0, n, log)
        self.env_base = worker_env(a.omp, self.cfg["base_K"])
        meta = dict(cfg=a.cfg, **self.cfg, order=self.order, stats=TXT, stats_mm=MM, mm_w=0.25, source=SRC,
                    fixed_set=FIXED, encoder="threads/35-nq15/nq35_layer.py (T12 nq_layer + nq15; L3-6 + nq35_t29 Had512 down as shipped v1)", started=now())
        old = jload(f"{self.root}/campaign.json")
        if old and (old["cfg"] != a.cfg or old.get("res_k") != self.cfg["res_k"]):
            raise SystemExit(f"{self.root} holds cfg {old['cfg']}, not {a.cfg}")
        if not old:
            jdump(meta, f"{self.root}/campaign.json")

    # ---------------- state ----------------
    def log(self, m):
        line = f"{now()} {m}"
        print(line, flush=True)
        open(f"{self.root}/driver.log", "a").write(line + "\n")

    def alert(self, kind, L, msg):
        self.log(f"ALERT {kind} L{L}: {msg}")
        open(f"{self.root}/ALERTS.jsonl", "a").write(json.dumps(dict(t=now(), kind=kind, layer=L, msg=msg)) + "\n")

    def done_experts(self, L):
        d = f"{self.root}/L{L}/experts"
        if not os.path.isdir(d):
            return set()
        return {int(f[1:-3]) for f in os.listdir(d) if re.fullmatch(r"E\d+\.pt", f)}

    def fin_ok(self, L):
        r = jload(f"{self.root}/fin/L{L}.json")
        return bool(r and r.get("rc") == 0)

    def spot_done(self, L):
        return os.path.exists(f"{self.root}/spot/L{L}.json")

    def fails(self, kind, L):
        return len((jload(f"{self.root}/fail/{kind}_L{L}.json") or {}).get("fails", []))

    def add_fail(self, kind, L, info):
        p = f"{self.root}/fail/{kind}_L{L}.json"
        r = jload(p) or dict(fails=[])
        r["fails"].append(info); jdump(r, p)

    def stats_ready(self, L):
        """fb_restore links stats/L{L} only after every file of the layer is fetched and sha256-verified."""
        return all(os.path.islink(f"{r}/stats/L{L}") and os.path.exists(f"{r}/stats/L{L}/meta.json") for r in (TXT, MM))

    def s_per_expert(self):
        t = [r["s"] / r["n"] for r in (jload(f"{self.root}/timing/enc.json") or {}).get("chunks", [])
             if r["n"] >= 2 and r.get("full")]
        return statistics.median(t[-32:]) if t else self.a.est_s

    def overhead_s(self):
        return self.a.overhead_s

    # ---------------- commands ----------------
    def enc_args(self, L):
        x = ["--layer", str(L), "--stats", TXT, "--stats-mm", MM, "--mm-w", "0.25", "--out", self.root,
             "--source", SRC, "--fixed-set", FIXED]
        return x + (["--res-k", self.cfg["res_k"]] if self.cfg["res_k"] else [])

    def enc_cmd(self, L, a, b):
        return [PY, f"{HERE}/nq35_layer.py"] + self.enc_args(L) + ["--experts", f"{a}:{b}", "--no-finalize"]

    def fin_cmd(self, L):
        fin = [PY, f"{HERE}/nq35_layer.py"] + self.enc_args(L) + ["--experts", "0:0"]
        return [PY, f"{HERE}/nq35_fin.py", "--layer", str(L), "--out", self.root, "--fin-cmd", json.dumps(fin),
                "--n-decode", "8"]

    def spot_cmd(self, L):
        return [PY, f"{HERE}/nq35_spot.py", "--layer", str(L), "--stats", TXT, "--out", self.root, "--source", SRC,
                "--fixed-set", FIXED, "--vision-stats", MM, "--vision-weight", "0.25", "--l2-thr", "5.0", "--l4-thr", "2.0"]

    # ---------------- processes ----------------
    def launch(self, kind, L, gpu, cmd, n=0, rng=None):
        tag = time.strftime("%m%d%H%M%S", time.gmtime())
        lp = f"{self.root}/logs/{kind}_L{L}_{rng[0] if rng else ''}_{tag}.log"
        env = dict(self.env_base, CUDA_VISIBLE_DEVICES=str(gpu))
        p = subprocess.Popen(cmd, env=env, stdout=open(lp, "w"), stderr=subprocess.STDOUT, start_new_session=True,
                             preexec_fn=lambda: os.nice(10), close_fds=True)
        self.jobs.append(dict(kind=kind, L=L, gpu=gpu, p=p, t0=time.time(), n=n, rng=rng, log=lp))
        self.log(f"launch {kind} L{L} {rng or ''} gpu{gpu} pid {p.pid}")

    def rss_gb(self):
        tot = 0
        pg = {j["p"].pid for j in self.jobs}
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                if os.getpgid(int(d)) in pg:
                    for line in open(f"/proc/{d}/status"):
                        if line.startswith("VmRSS:"):
                            tot += int(line.split()[1])
            except (OSError, ProcessLookupError):
                pass
        return tot / 2**20

    def mem_avail_gb(self):
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
        return 0

    def kill_job(self, j):
        try:
            os.killpg(j["p"].pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        for _ in range(30):
            if j["p"].poll() is not None:
                break
            time.sleep(1)
        try:
            os.killpg(j["p"].pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        j["p"].wait()

    def reap(self):
        for j in list(self.jobs):
            rc = j["p"].poll()
            if rc is None:
                continue
            self.jobs.remove(j)
            dt = time.time() - j["t0"]
            L = j["L"]
            tail = open(j["log"]).read()[-1500:]
            if j["kind"] == "enc":
                a, b = j["rng"]
                have = self.done_experts(L)
                got = len([e for e in range(a, b) if e in have])
                full = rc == 0 and got == j["n"]
                p = f"{self.root}/timing/enc.json"
                r = jload(p) or dict(chunks=[])
                r["chunks"].append(dict(L=L, a=a, b=b, n=j["n"], got=got, s=round(dt, 1), gpu=j["gpu"], rc=rc, full=full,
                                        t=now()))
                jdump(r, p)
                self.log(f"enc L{L} {a}:{b} rc {rc} {got}/{j['n']} in {dt:.0f}s ({dt / max(got, 1):.1f} s/expert)")
                if rc != 0 and not j.get("killed"):
                    self.add_fail("enc", L, dict(rng=[a, b], rc=rc, t=now(), tail=tail[-600:]))
                    self.alert("enc_fail", L, f"{a}:{b} rc {rc}: {tail[-300:]}")
            elif j["kind"] == "fin":
                if j.get("killed"):
                    continue
                jdump(dict(rc=rc, s=round(dt), t=now(), tail=tail[-1200:]), f"{self.root}/fin/L{L}.json")
                if rc != 0:
                    self.add_fail("fin", L, dict(rc=rc, t=now(), tail=tail[-600:]))
                    self.alert("fin_fail", L, f"rc {rc}: {tail[-400:]}")
                else:
                    self.log(f"fin L{L} ok in {dt:.0f}s")
            elif j["kind"] == "spot":
                if j.get("killed"):
                    continue
                sp = jload(f"{self.root}/spot/L{L}.json")
                if rc != 0 or sp is None:
                    self.add_fail("spot", L, dict(rc=rc, t=now(), tail=tail[-600:]))
                    self.alert("spot_fail", L, f"rc {rc}: {tail[-300:]}")
                else:
                    self.log(f"spot L{L} flag={sp.get('flag')} {sp.get('summary')} ({dt:.0f}s)")
                    if sp.get("flag"):
                        self.alert("spot_flag", L, json.dumps(sp.get("summary")))

    # ---------------- scheduling ----------------
    def running(self, kind=None, L=None):
        return [j for j in self.jobs if (kind is None or j["kind"] == kind) and (L is None or j["L"] == L)]

    def schedule(self, t_end):
        left = t_end - time.time()
        if self.rss_gb() > self.a.max_rss_gb or self.mem_avail_gb() < 64:
            return
        # finalize / spot: short jobs, extra slot on the least-busy GPU; need >= 20 min left
        if left > 1200:
            for L in self.order:
                busy = [j["gpu"] for j in self.jobs]
                g = min(self.gpus, key=lambda i: busy.count(i))
                if (len(self.done_experts(L)) == NEXP and not self.fin_ok(L) and not self.running("fin", L)
                        and not self.running("enc", L) and self.fails("fin", L) < 2 and len(self.running("fin")) < 2):
                    self.launch("fin", L, g, self.fin_cmd(L)); continue
                if (self.fin_ok(L) and not self.spot_done(L) and not self.running("spot", L) and self.fails("spot", L) < 2
                        and len(self.running("spot")) < 2):
                    self.launch("spot", L, g, self.spot_cmd(L))
        # encode chunks: one per free slot, in layer order
        spe = self.s_per_expert()
        for g in self.gpus:
            while len([j for j in self.running("enc") if j["gpu"] == g]) < self.a.wpg:
                n_fit = int((t_end - time.time() - self.overhead_s() - 60) / (spe * self.a.wpg))
                n_max = min(self.a.chunk, n_fit)
                if n_max < self.a.min_chunk:
                    return
                rng = self.next_chunk(n_max)
                if rng is None:
                    return
                L, a, b, n = rng
                self.launch("enc", L, g, self.enc_cmd(L, a, b), n=n, rng=(a, b))

    def next_chunk(self, n_max):
        for L in self.order:
            if self.fails("enc", L) >= 3 or not self.stats_ready(L):
                continue
            have = self.done_experts(L)
            taken = set()
            for j in self.running("enc", L):
                taken |= set(range(*j["rng"]))
            miss = [e for e in range(NEXP) if e not in have and e not in taken]
            if not miss:
                continue
            a = miss[0]; b = a; n = 0
            while b < NEXP and b not in taken and n < n_max:
                n += b not in have; b += 1
            return L, a, b, n
        return None

    def runnable(self):
        return self.next_chunk(1) is not None or any(
            (len(self.done_experts(L)) == NEXP and not self.fin_ok(L) and self.fails("fin", L) < 2) or
            (self.fin_ok(L) and not self.spot_done(L) and self.fails("spot", L) < 2) for L in self.order)

    def all_done(self):
        return all(self.fin_ok(L) and self.spot_done(L) for L in self.order)

    def write_status(self, extra=None):
        lay = {}
        for L in self.order:
            lay[L] = dict(enc=len(self.done_experts(L)), fin=self.fin_ok(L), spot=(jload(f"{self.root}/spot/L{L}.json") or {}).get("flag"))
        st = dict(t=now(), s_per_expert=self.s_per_expert(), jobs=[dict(kind=j["kind"], L=j["L"], gpu=j["gpu"], rng=j["rng"],
                  pid=j["p"].pid, age=round(time.time() - j["t0"])) for j in self.jobs], layers=lay,
                  n_enc=sum(v["enc"] for v in lay.values()), n_fin=sum(v["fin"] for v in lay.values()),
                  rss_gb=round(self.rss_gb(), 1), **(extra or {}))
        jdump(st, f"{self.root}/status.json")

    # ---------------- holds ----------------
    def hold(self, fd):
        t0 = time.time(); t_end = t0 + 60 * self.a.hold_min
        self.log(f"hold start (deadline {self.a.hold_min} min)")
        last = 0
        while True:
            self.reap()
            if os.path.exists(f"{self.root}/STOP"):
                self.log("STOP file: draining (no new launches)")
                t_end = min(t_end, time.time())
            if time.time() < t_end:
                self.schedule(t_end)
            if time.time() - last > 60:
                self.write_status(dict(hold_left_s=round(t_end - time.time()))); last = time.time()
            if not self.jobs and (self.all_done() or not self.runnable() or
                                  time.time() >= t_end - self.overhead_s() - 60 - self.a.min_chunk * self.s_per_expert()):
                break
            if time.time() >= t_end + 60 * self.a.grace_min:
                for j in list(self.jobs):
                    j["killed"] = True
                    self.log(f"deadline: kill own {j['kind']} L{j['L']} {j['rng'] or ''} pid {j['p'].pid}")
                    self.kill_job(j)
                self.reap()
                break
            time.sleep(5)
        self.write_status()
        self.log(f"hold end after {(time.time() - t0) / 60:.1f} min")

    def run(self):
        holds = 0
        while not self.all_done():
            if os.path.exists(f"{self.root}/STOP"):
                self.log("STOP file present: exit"); return
            if not self.runnable():
                self.log("nothing runnable (stats not restored yet / failures); waiting 300 s")
                time.sleep(300); continue
            fd = open(LOCK, "a")
            self.log("waiting for gpu.lock")
            fcntl.flock(fd, fcntl.LOCK_EX)
            self.log("gpu.lock acquired")
            try:
                self.hold(fd)
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN); fd.close()
                self.log("gpu.lock released")
            holds += 1
            if self.a.max_holds and holds >= self.a.max_holds:
                self.log(f"max holds {holds} reached"); return
            time.sleep(self.a.pause_s)
        self.log("ALL DONE")


def status(root):
    st = jload(f"{root}/status.json")
    print(json.dumps({k: v for k, v in st.items() if k != "layers"}, indent=1))
    print(" ".join(f"L{L}:{v['enc']}{'F' if v['fin'] else ''}{'S' if v['spot'] is not None else ''}{'!' if v['spot'] else ''}"
                   for L, v in st["layers"].items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "status"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--cfg", default="b175", choices=list(CFGS))
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--wpg", type=int, default=1, help="encode workers per GPU (GPU-bound: 1)")
    ap.add_argument("--omp", type=int, default=2)
    ap.add_argument("--order", default="3-10,40-50,11-39,51-77")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--min-chunk", type=int, default=2)
    ap.add_argument("--est-s", type=float, default=45.0, help="s/expert before any chunk is timed")
    ap.add_argument("--overhead-s", type=float, default=90.0, help="per-process startup (stats, cuda, ext)")
    ap.add_argument("--hold-min", type=float, default=85.0)
    ap.add_argument("--grace-min", type=float, default=3.0, help="kill own workers at hold-min + grace (<= 90 min)")
    ap.add_argument("--pause-s", type=float, default=120.0)
    ap.add_argument("--max-holds", type=int, default=0)
    ap.add_argument("--max-rss-gb", type=float, default=110.0)
    a = ap.parse_args()
    assert a.hold_min + a.grace_min <= 90
    if a.cmd == "status":
        return status(a.root)
    Campaign(a).run()


if __name__ == "__main__":
    main()
