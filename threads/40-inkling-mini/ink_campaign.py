"""Inkling-2-Mini NestQuant encode driver (from threads/37-flash/nq37_campaign.py, no gpu.lock holds).

  python ink_campaign.py run --root /tmp/nq_mini/enc_b175 [--gpus 0,...,7] [--wpg 3] [--order 0-47]
  python ink_campaign.py status --root R

Restart-safe from filesystem state:
  encoded(L,E) = R/L{L}/experts/E{E}.pt   (nq_layer saves atomically, skips existing experts)
  finalized(L) = R/fin/L{L}.json rc 0     (ink_fin.py = T25 nq25_finalize: nq_layer finalize -> tp{s}.pt + manifest,
                                           safetensors convert + round trip, decode compare on 8 experts)
A layer is encoded once its stats are complete (stats/L{L} link + meta.json).  STOP file in R = drain and exit.
No uploads of any kind."""
import os, sys, json, time, argparse, signal, subprocess, re, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/home/coder/git/glm52/.venv/bin/python"
NQ = "/home/coder/git/nestquant"
TXT = "/tmp/nq_mini/cap-txt"
NEXP = 128
CFGS = {"b175": dict(base_K=1.75, res_k="2.25,2.25,2.5625"), "b20": dict(base_K=2.0, res_k=None),
        "b175mm": dict(base_K=1.75, res_k="2.25,2.25,2.5625", stats_mm="/tmp/nq_mini/cap-mm", mm_w=0.25)}


def worker_env(omp, base_K):
    env = dict(os.environ)
    lib06 = f"{NQ}/threads/06-expert-objective/lib"
    env["LD_LIBRARY_PATH"] = ":".join(x for x in (lib06, env.get("LD_LIBRARY_PATH", "")) if x)
    env["PYTHONPATH"] = ":".join(x for x in (f"{NQ}/threads/05-exl3-harness", "/home/coder/git/orbit-duet",
                                             env.get("PYTHONPATH", "")) if x)
    env.update(OMP_NUM_THREADS=str(omp), MKL_NUM_THREADS=str(omp), OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1",
               CUDA_DEVICE_ORDER="PCI_BUS_ID", NQ35_BASE_K=str(base_K), TORCH_CUDA_ARCH_LIST="10.0")
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
    return out


class Campaign:
    def __init__(self, a):
        self.a = a
        self.root = a.root
        self.cfg = CFGS[a.cfg]
        self.order = parse_order(a.order)
        self.gpus = [int(x) for x in a.gpus.split(",")]
        for d in ("logs", "fin", "fail", "timing"):
            os.makedirs(f"{self.root}/{d}", exist_ok=True)
        self.jobs = []
        self.env_base = worker_env(a.omp, self.cfg["base_K"])
        old = jload(f"{self.root}/campaign.json")
        if old and old["cfg"] != a.cfg:
            raise SystemExit(f"{self.root} holds cfg {old['cfg']}, not {a.cfg}")
        if not old:
            jdump(dict(cfg=a.cfg, **self.cfg, order=self.order, stats=TXT, source="/tmp/nq_mini/model",
                       encoder="ink_layer.py (T12 nq_layer + nq15 under nq_ink_env: D 2304, F 1152, 128 experts, NSH 1 (Inkling-2-Mini))",
                       started=now()), f"{self.root}/campaign.json")

    def log(self, m):
        line = f"{now()} {m}"
        print(line, flush=True)
        open(f"{self.root}/driver.log", "a").write(line + "\n")

    def done_experts(self, L):
        d = f"{self.root}/L{L}/experts"
        if not os.path.isdir(d):
            return set()
        return {int(f[1:-3]) for f in os.listdir(d) if re.fullmatch(r"E\d+\.pt", f)}

    def fin_ok(self, L):
        r = jload(f"{self.root}/fin/L{L}.json")
        return bool(r and r.get("rc") == 0)

    def fails(self, kind, L):
        return len((jload(f"{self.root}/fail/{kind}_L{L}.json") or {}).get("fails", []))

    def add_fail(self, kind, L, info):
        p = f"{self.root}/fail/{kind}_L{L}.json"
        r = jload(p) or dict(fails=[])
        r["fails"].append(info); jdump(r, p)

    def stats_ready(self, L):
        ok = os.path.islink(f"{TXT}/stats/L{L}") and os.path.exists(f"{TXT}/stats/L{L}/meta.json")
        mm = self.cfg.get("stats_mm")
        return ok and (not mm or (os.path.islink(f"{mm}/stats/L{L}") and os.path.exists(f"{mm}/stats/L{L}/meta.json")))

    def enc_args(self, L):
        x = ["--layer", str(L), "--stats", TXT, "--out", self.root, "--source", "/tmp/nq_mini/model"]
        if os.path.exists(f"{TXT}/fixed_set.json"):
            x += ["--fixed-set", f"{TXT}/fixed_set.json"]
        if self.cfg.get("stats_mm"):
            x += ["--stats-mm", self.cfg["stats_mm"], "--mm-w", str(self.cfg["mm_w"])]
        return x + (["--res-k", self.cfg["res_k"]] if self.cfg["res_k"] else [])

    def enc_cmd(self, L, a, b):
        return [PY, f"{HERE}/ink_layer.py"] + self.enc_args(L) + ["--experts", f"{a}:{b}", "--no-finalize"]

    def fin_cmd(self, L):
        fin = [PY, f"{HERE}/ink_layer.py"] + self.enc_args(L) + ["--experts", "0:0"]
        return [PY, f"{HERE}/ink_fin.py", "--layer", str(L), "--out", self.root, "--fin-cmd", json.dumps(fin),
                "--n-decode", "8"]

    def launch(self, kind, L, gpu, cmd, n=0, rng=None):
        tag = time.strftime("%m%d%H%M%S", time.gmtime())
        lp = f"{self.root}/logs/{kind}_L{L}_{rng[0] if rng else ''}_{tag}.log"
        env = dict(self.env_base, CUDA_VISIBLE_DEVICES=str(gpu))
        p = subprocess.Popen(cmd, env=env, stdout=open(lp, "w"), stderr=subprocess.STDOUT, start_new_session=True,
                             preexec_fn=lambda: os.nice(10), close_fds=True)
        self.jobs.append(dict(kind=kind, L=L, gpu=gpu, p=p, t0=time.time(), n=n, rng=rng, log=lp))
        self.log(f"launch {kind} L{L} {rng or ''} gpu{gpu} pid {p.pid}")

    def mem_avail_gb(self):
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
        return 0

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
                p = f"{self.root}/timing/enc.json"
                r = jload(p) or dict(chunks=[])
                r["chunks"].append(dict(L=L, a=a, b=b, n=j["n"], got=got, s=round(dt, 1), gpu=j["gpu"], rc=rc, t=now()))
                jdump(r, p)
                self.log(f"enc L{L} {a}:{b} rc {rc} {got}/{j['n']} in {dt:.0f}s ({dt / max(got, 1):.1f} s/expert)")
                if rc != 0:
                    self.add_fail("enc", L, dict(rng=[a, b], rc=rc, t=now(), tail=tail[-600:]))
                    self.log(f"ALERT enc_fail L{L} {a}:{b}: {tail[-300:]}")
            elif j["kind"] == "fin":
                jdump(dict(rc=rc, s=round(dt), t=now(), tail=tail[-1200:]), f"{self.root}/fin/L{L}.json")
                if rc != 0:
                    self.add_fail("fin", L, dict(rc=rc, t=now(), tail=tail[-600:]))
                    self.log(f"ALERT fin_fail L{L}: {tail[-400:]}")
                else:
                    self.log(f"fin L{L} ok in {dt:.0f}s")

    def running(self, kind=None, L=None):
        return [j for j in self.jobs if (kind is None or j["kind"] == kind) and (L is None or j["L"] == L)]

    def next_chunk(self):
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
            while b < NEXP and b not in taken and n < self.a.chunk:
                n += b not in have; b += 1
            return L, a, b, n
        return None

    def schedule(self):
        if self.mem_avail_gb() < self.a.min_avail_gb:
            return
        for L in self.order:
            busy = [j["gpu"] for j in self.jobs]
            g = min(self.gpus, key=lambda i: busy.count(i))
            if (len(self.done_experts(L)) == NEXP and not self.fin_ok(L) and not self.running("fin", L)
                    and not self.running("enc", L) and self.fails("fin", L) < 2 and len(self.running("fin")) < self.a.max_fin):
                self.launch("fin", L, g, self.fin_cmd(L))
        for g in self.gpus:
            while len([j for j in self.running("enc") if j["gpu"] == g]) < self.a.wpg:
                rng = self.next_chunk()
                if rng is None:
                    return
                L, a, b, n = rng
                self.launch("enc", L, g, self.enc_cmd(L, a, b), n=n, rng=(a, b))

    def all_done(self):
        return all(self.fin_ok(L) for L in self.order)

    def write_status(self):
        lay = {L: dict(enc=len(self.done_experts(L)), fin=self.fin_ok(L)) for L in self.order}
        t = [r["s"] / r["got"] for r in (jload(f"{self.root}/timing/enc.json") or {}).get("chunks", []) if r["got"] >= 2]
        st = dict(t=now(), s_per_expert_per_worker=statistics.median(t[-48:]) if t else None,
                  jobs=[dict(kind=j["kind"], L=j["L"], gpu=j["gpu"], rng=j["rng"], age=round(time.time() - j["t0"]))
                        for j in self.jobs], layers=lay, n_enc=sum(v["enc"] for v in lay.values()),
                  n_fin=sum(v["fin"] for v in lay.values()))
        jdump(st, f"{self.root}/status.json")

    def run(self):
        last = 0
        while not self.all_done():
            self.reap()
            if os.path.exists(f"{self.root}/STOP"):
                if not self.jobs:
                    self.log("STOP: exit"); return
            else:
                self.schedule()
            if time.time() - last > 60:
                self.write_status(); last = time.time()
            time.sleep(5)
        self.write_status()
        self.log("ALL DONE")


def status(root):
    st = jload(f"{root}/status.json")
    print(json.dumps({k: v for k, v in st.items() if k != "layers"}, indent=1))
    print(" ".join(f"L{L}:{v['enc']}{'F' if v['fin'] else ''}" for L, v in st["layers"].items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "status"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--cfg", default="b175", choices=list(CFGS))
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--wpg", type=int, default=3)
    ap.add_argument("--omp", type=int, default=4)
    ap.add_argument("--order", default="0-23")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--max-fin", type=int, default=2)
    ap.add_argument("--min-avail-gb", type=float, default=200.0)
    a = ap.parse_args()
    if a.cmd == "status":
        return status(a.root)
    Campaign(a).run()


if __name__ == "__main__":
    main()
