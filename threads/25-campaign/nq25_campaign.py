"""Thread 25: full-model NestQuant encode campaign driver (GLM-5.3 routed experts, MoE layers 3..77).

  python nq25_campaign.py run    --out /tmp/nestquant/nq-encode --stats-root /tmp/nestquant/19-capture-glmfmt \
                                 --stats-version stats [--layers 3:78] [--experts 0:256] [--chunk 16] ...
  python nq25_campaign.py status --out ROOT
  python nq25_campaign.py upload --out ROOT [--go]      # dry run (listing + sizes) unless --go

The encoder is a black box: every encode worker is one subprocess
    nq_layer.py --layer L --experts a:b --no-finalize --stats SHIM --out ROOT --fixed-set F [ENC_ARGS]
(or --encoder-cmd TEMPLATE, e.g. T23's batched path, which must write ROOT/L{L}/experts/E{E}.pt exactly like
nq_layer). The only contract used: E{E}.pt appears atomically when expert E is done; finalize = nq_layer with an
empty expert range (+ --check-decode) writes L{L}/tp{0..7}.pt + manifest.json and prints the decode verdict.

Campaign config (stats root/version, layers, encoder command and flags) is frozen in ROOT/campaign.json at the
first run; later runs may change only scheduling knobs (refused otherwise unless --force-config).
Stats pinning: ROOT/_stats is a shim capture root whose stats/L{L} are absolute symlinks to the version directory
each layer resolved to when first scheduled (eval/, bnd_rows/ ... link to the real root), so every worker of a layer
reads the same stats version even if T19 merges more shards meanwhile. The pinned version is recorded per layer.

State: ROOT/state.json (driver is the only writer; rebuilt from the filesystem on restart: the E files are the truth)
  layer state: pending -> encoding -> encoded -> finalized -> checked -> [spotchecked] -> uploaded
Workers are started in their own session; a restarted driver adopts still-running workers (pid + create_time)
instead of launching duplicates.
Memory: a worker is launched on the GPU with the most effective free VRAM, only if
  nvidia-smi free - sum(our not-yet-materialised reservations on that GPU) >= reservation + margin,
and host MemAvailable (and the cgroup headroom) >= rss limit + host margin. Worker RSS (process tree) above
--rss-gb is killed (our own process only) and the chunk retried.
Spot checks (nq25_spot.py) run only on capacity beyond the per-GPU encode cap (or when no encode work is waiting).
"""
import os, sys, json, time, glob, shlex, signal, shutil, argparse, subprocess, collections, re
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
NQ = "/home/coder/git/nestquant"
T12 = f"{NQ}/threads/12-reference-encoder"
T23 = f"{NQ}/threads/23-encode-throughput"
PY = "/home/coder/git/glm52/.venv/bin/python"
FIXED = f"{NQ}/threads/22-boundary-experts/fixed_set.json"
NEXP = 256
MIN_EXPERT_FILE = 1 << 20          # sanity: a finished E{E}.pt is far larger

FROZEN = ("stats_root", "stats_version", "layers", "experts", "encoder", "encoder_cmd", "enc_args", "fixed_set", "source",
          "vision_root", "vision_version", "vision_weight", "vision_args", "spot_h_fn")
SCHED_DEFAULTS = dict(chunk=16, gpus="0,1,2,3,4,5,6,7", workers_per_gpu=3, vram_gb=14.0, margin_gb=6.0, rss_gb=48.0,
                      host_margin_gb=96.0, omp=8, max_attempts=3, spot=True, spot_workers=2, spot_vram_gb=14.0,
                      max_fin=4, tick=10.0, spot_l2=5.0, spot_l4=2.0, max_workers=64, min_shards=0, upload="dry", max_up=2,
                      nq_check="first", n_decode=8, refcheck=True, hf_reconcile=True, code_guard=True,
                      repo="jarrelscy/GLM-5.3-NestQuant-2-4bit", stats_gate="final", gate_backup=False, t23_go="",
                      t23_group=4, remanifest=True, mmw_go="")
ALERTS = "ALERTS.jsonl"


def now():
    return time.time()


def mel(t=None):
    """Melbourne wall time (UTC+10; the box's TZ may differ)."""
    return time.strftime("%a %d %b %H:%M", time.gmtime((t or now()) + 10 * 3600)) + " AEST"


def jload(p, d=None):
    try:
        with open(p) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return d


def jdump(obj, p):
    tmp = f"{p}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, p)


def rng(s):
    a, b = map(int, s.split(":"))
    return a, b


# ------------------------------------------------------------------------------------------------ environment
def worker_env(gpu, omp):
    env = dict(os.environ)
    # runtime/env.sh + threads/12-reference-encoder/env.sh, spelled out (no shell sourcing per launch)
    compat = "/home/coder/git/glm52/artifacts/shared-bit-graphs/runtime/cuda-compat/usr/local/cuda-13.0/compat"
    lib06 = f"{NQ}/threads/06-expert-objective/lib"
    env["LD_LIBRARY_PATH"] = ":".join(x for x in (lib06, compat, env.get("LD_LIBRARY_PATH", "")) if x)
    env["PYTHONPATH"] = ":".join(x for x in (f"{NQ}/threads/05-exl3-harness", "/home/coder/git/orbit-duet",
                                             env.get("PYTHONPATH", "")) if x)
    env.update(OMP_NUM_THREADS=str(omp), MKL_NUM_THREADS=str(omp), OPENBLAS_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", CUDA_DEVICE_ORDER="PCI_BUS_ID")
    for k in [k for k in env if k.startswith("HF_TOKEN") or k == "HUGGING_FACE_HUB_TOKEN"]:
        env.pop(k)                                   # workers never need the upload token
    return env


def gpu_info():
    """{idx: dict(free, total, uuid)} in MiB, and {pid: (uuid, used MiB)} for every compute process."""
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=60).stdout
    g = {}
    for line in q.strip().splitlines():
        i, u, f, t = [x.strip() for x in line.split(",")]
        g[int(i)] = dict(uuid=u, free=float(f), total=float(t))
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=60).stdout
    procs = {}
    for line in q.strip().splitlines():
        try:
            p, u, m = [x.strip() for x in line.split(",")]
            procs[int(p)] = (u, float(m))
        except ValueError:
            pass
    return g, procs


def host_headroom_gb():
    avail = psutil.virtual_memory().available
    try:
        mx = open("/sys/fs/cgroup/memory.max").read().strip()
        if mx != "max":
            cur = int(open("/sys/fs/cgroup/memory.current").read())
            # page cache is reclaimable, but be conservative: count anon+kernel only via memory.stat
            st = dict(l.split() for l in open("/sys/fs/cgroup/memory.stat"))
            used = int(st.get("anon", cur)) + int(st.get("kernel", 0)) + int(st.get("shmem", 0))
            avail = min(avail, int(mx) - used)
    except Exception:
        pass
    return avail / 2 ** 30


def tree_rss_gb(p):
    try:
        ps = [p] + p.children(recursive=True)
        return sum(x.memory_info().rss for x in ps if x.is_running()) / 2 ** 30
    except psutil.Error:
        return 0.0


def du_bytes(path):
    tot = 0
    for dp, _, fs in os.walk(path):
        for f in fs:
            try:
                tot += os.lstat(os.path.join(dp, f)).st_size
            except FileNotFoundError:
                pass
    return tot


# ------------------------------------------------------------------------------------------------ identity
CODE_FILES = {                                   # code that defines the encoded bytes (reference path)
    "threads/12-reference-encoder": ("nq_encode.py", "nq_decode.py", "nq_patvit.py", "nq_layer.py", "nq_bnd.py",
                                     "csrc/nq_fracvit.cu"),
    "threads/19-full-capture": ("nq19_load.py",),
    "threads/26-vision-calib": ("nq26_blend.py",),
    "threads/05-exl3-harness": ("harness.py",),
}
T23_FILES = {"threads/23-encode-throughput": ("nq_encode_batch.py", "nq_layer_batch.py", "common.py", "k2vit.py",
                                             "csrc/nq_k2vit.cu"),
             "threads/25-campaign": ("nq25_t23.py",)}
T25_FILES = {"threads/25-campaign": ("nq25_campaign.py", "nq25_finalize.py", "nq25_st.py", "nq25_spot.py", "nq25_upload.py")}
ID_EXCLUDE = ("layers", "experts", "encoder", "encoder_cmd", "spot_h_fn",      # don't change the encoded bytes
              "fixed_set")          # only feeds manifest default_allocation (T12 nq_layer.default_allocation), not the encode
T23_ENCODERS = ("t23", "t23b")      # t23 = nq25_t23.py adapter, t23b = T23's nq_layer_batch.py drop-in (nq_layer CLI)


def _sha16(p):
    import hashlib
    try:
        return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    except (FileNotFoundError, IsADirectoryError, TypeError):
        return None


def _group_id(files, groups):
    """hash over exactly the files listed in groups ({dir: (files...)}), not every file under the dir."""
    import hashlib
    ks = sorted(f"{d}/{f}" for d, fs in groups.items() for f in fs)
    return hashlib.sha256(json.dumps([(k, files.get(k)) for k in ks]).encode()).hexdigest()[:16]


def code_info():
    """sha16 of every file that defines the bytes; code_id = reference path (T12/T19/T26/T05), t23_id = batched path.
    The git head is recorded too, but other agents' working copies can be dirty: the file hashes are the identity."""
    files = {f"{d}/{f}": _sha16(f"{NQ}/{d}/{f}") for grp in (CODE_FILES, T23_FILES, T25_FILES)
             for d, fs in grp.items() for f in fs}
    git = subprocess.run(["git", "-C", NQ, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", NQ, "status", "--porcelain", "--"] + list(files), capture_output=True,
                           text=True).stdout.splitlines()
    return dict(code_id=_group_id(files, CODE_FILES), t23_id=_group_id(files, T23_FILES),
                t25_id=_group_id(files, T25_FILES), files=files, git_head=git, dirty=[x[3:] for x in dirty])


def config_id(cfg):
    """hash of the frozen config fields that change the encoded bytes. The fixed set is NOT part of it: it only sets the
    manifest's default_allocation, which is refreshed (and re-uploaded) whenever the fixed_set file changes."""
    import hashlib
    c = {k: v for k, v in cfg.items() if k not in ID_EXCLUDE}
    return hashlib.sha256(json.dumps(c, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ stats shim
SHIMS = (("_stats", "stats_root", "stats_version"), ("_stats_mm", "vision_root", "vision_version"))


def build_shim(root, cfg):
    """ROOT/_stats (text) and ROOT/_stats_mm (vision, if configured): capture roots linking everything but stats*;
    their stats/L{L} are pinned per layer by pin_layer."""
    for name, rk, _ in SHIMS:
        real = cfg.get(rk)
        if not real:
            continue
        shim = f"{root}/{name}"
        os.makedirs(f"{shim}/stats", exist_ok=True)
        for n in os.listdir(real):                   # eval, bnd_rows, fixed_set.json, MANIFEST ... (not stats*)
            if not n.startswith("stats") and not os.path.lexists(f"{shim}/{n}"):
                os.symlink(f"{real}/{n}", f"{shim}/{n}")


def pin_layer(root, cfg, L, min_shards=0, force=None):
    """Pin stats/L{L} of every configured shim to the version its real root resolves to now (once per layer), or to
    force[shim] (T19's final/L{L}.json version_dir). Returns {shim: pinned dir} or None while a root is not ready."""
    want = []
    force = force or {}
    for name, rk, vk in SHIMS:
        if not cfg.get(rk):
            continue
        link = f"{root}/{name}/stats/L{L}"
        if os.path.lexists(link):
            continue
        vd = force.get(name) or os.path.realpath(f"{cfg[rk]}/{cfg[vk]}/L{L}")
        m = jload(f"{vd}/meta.json")
        if m is None or (name == "_stats" and len(m.get("shards", [])) < min_shards):
            return None
        want.append((link, vd))
    for link, vd in want:                            # all ready -> pin together
        os.symlink(vd, link)
    return {name: os.readlink(f"{root}/{name}/stats/L{L}") for name, rk, _ in SHIMS if cfg.get(rk)}


def stats_desc(vd):
    m = jload(f"{vd}/meta.json", {}) or {}
    return dict(dir=vd, shards=[s.get("shard", s) if isinstance(s, dict) else s for s in m.get("shards", [])],
                T_fit=m.get("T_fit"), schema=m.get("schema"))


# ------------------------------------------------------------------------------------------------ campaign
class Campaign:
    def __init__(self, a):
        self.a = a
        self.root = a.out
        os.makedirs(f"{self.root}/logs", exist_ok=True)
        os.makedirs(f"{self.root}/spot", exist_ok=True)
        os.makedirs(f"{self.root}/uploads", exist_ok=True)
        self.cfg = self._config()
        l0, l1 = rng(self.cfg["layers"])
        self.layers = [L for L in range(l0, l1)]
        e0, e1 = rng(self.cfg["experts"])
        self.experts = list(range(e0, e1))
        self.st = jload(f"{self.root}/state.json") or dict(layers={}, workers={}, created=mel(), events=[])
        self.procs = {}                                # wid -> psutil.Process / Popen
        self._final = {}
        self._holds = {}                               # L -> current gate hold reason (status.json "holding")                               # L -> {"_stats": T19 final version_dir} (gate a pin)
        self.gpus_allowed = [int(x) for x in str(self.sched["gpus"]).split(",") if x != ""]
        self.last_du = (0, 0)
        self.log_f = open(f"{self.root}/campaign.log", "a")

    # -------- config
    def _config(self):
        a = self.a
        p = f"{self.root}/campaign.json"
        old = jload(p)
        new = dict(stats_root=a.stats_root, stats_version=a.stats_version, layers=a.layers, experts=a.experts,
                   encoder=a.encoder, encoder_cmd=a.encoder_cmd, enc_args=a.enc_args, fixed_set=a.fixed_set, source=a.source,
                   vision_root=a.vision_root, vision_version=a.vision_version, vision_weight=a.vision_weight,
                   vision_args=a.vision_args, spot_h_fn=a.spot_h_fn)
        sched = {k: getattr(a, k) for k in SCHED_DEFAULTS}
        if old:
            diff = {k: (old["frozen"].get(k), new[k]) for k in FROZEN if new[k] is not None and old["frozen"].get(k) != new[k]}
            if diff and not a.force_config:
                raise SystemExit(f"campaign.json frozen config differs {diff}; pass --force-config to override")
            frozen = {k: (new[k] if (new[k] is not None and a.force_config) else old["frozen"].get(k)) for k in FROZEN}
            sd = dict(old.get("sched", {}))
            sd.update({k: v for k, v in sched.items() if v is not None})
        else:
            frozen = {k: new[k] for k in FROZEN}
            for k, d in dict(layers="3:78", experts=f"0:{NEXP}", encoder="nq_layer", encoder_cmd="", enc_args="", fixed_set=FIXED,
                             source="/tmp/nestquant/src/glm53-fp8", vision_root="", vision_version="stats",
                             vision_weight=0.0, vision_args="", spot_h_fn="").items():
                if frozen[k] is None:
                    frozen[k] = d
            if not frozen["stats_root"] or not frozen["stats_version"]:
                raise SystemExit("first run needs --stats-root and --stats-version")
            sd = {k: (v if v is not None else SCHED_DEFAULTS[k]) for k, v in sched.items()}
        for k, d in SCHED_DEFAULTS.items():
            sd.setdefault(k, d)
        self.sched = sd
        code = (old or {}).get("code")
        if code is None or a.accept_code:
            code = code_info(); code["accepted"] = mel()
        cfg = dict(frozen=frozen, sched=sd, updated=mel(), config_id=config_id(frozen), code=code,
                   created=(old or {}).get("created", mel()))
        jdump(cfg, p)
        self.code = code
        return frozen

    def log(self, msg):
        line = f"[{mel()}] {msg}"
        print(line, flush=True)
        self.log_f.write(line + "\n"); self.log_f.flush()

    def save(self):
        self.st["updated"] = mel()
        jdump(self.st, f"{self.root}/state.json")

    def lay(self, L):
        return self.st["layers"].setdefault(str(L), dict(state="pending", experts={}, chunks={}, events=[]))

    def edir(self, L):
        return f"{self.root}/L{L}/experts"

    def done_experts(self, L):
        d = self.edir(L)
        if not os.path.isdir(d):
            return set()
        out = set()
        for f in os.listdir(d):
            if f.startswith("E") and f.endswith(".pt"):
                try:
                    if os.path.getsize(f"{d}/{f}") >= MIN_EXPERT_FILE:
                        out.add(int(f[1:-3]))
                except (ValueError, FileNotFoundError):
                    pass
        return out

    # -------- commands
    def enc_cmd(self, L, e0, e1, finalize=False):
        c = self.cfg
        shim = f"{self.root}/_stats"
        if c["encoder_cmd"] and not finalize and self.layer_encoder(L) != "nq_layer":
            t = c["encoder_cmd"].format(py=PY, layer=L, e0=e0, e1=e1, out=self.root, stats=shim, src=c["source"],
                                        fixed_set=c["fixed_set"], t12=T12)
            return shlex.split(t)
        extra = shlex.split(c["enc_args"] or "") + self.vision_flags()
        if self.layer_encoder(L) == "t23" and not finalize and not c["encoder_cmd"]:
            return [PY, f"{HERE}/nq25_t23.py", "--layer", str(L), "--experts", f"{e0}:{e1}", "--stats", shim,
                    "--out", self.root, "--source", c["source"]] + extra
        if self.layer_encoder(L) == "t23b" and not finalize and not c["encoder_cmd"]:     # T23: --group <= 4 (12 GB cap)
            return [PY, f"{T23}/nq_layer_batch.py", "--layer", str(L), "--experts", f"{e0}:{e1}", "--stats", shim,
                    "--out", self.root, "--source", c["source"], "--no-finalize", "--group", str(self.sched["t23_group"])] + extra
        cmd = [PY, f"{T12}/nq_layer.py", "--layer", str(L), "--stats", shim, "--out", self.root,
               "--source", c["source"], "--fixed-set", c["fixed_set"]]
        if finalize:                               # finalize reads only stats/fixed-set; encoder flags don't apply
            return cmd + ["--experts", "0:0", "--check-decode"]
        return cmd + ["--experts", f"{e0}:{e1}", "--no-finalize"] + extra

    def layer_encoder(self, L):
        """campaign encoder, unless the layer fell back to the reference path after a refcheck mismatch."""
        return self.lay(L).get("encoder") or (self.cfg["encoder"] if not self.cfg["encoder_cmd"] else "custom")

    def ref_cmd(self, L, E):
        """reference (T12 nq_layer) encode of one expert into ROOT/_ref + payload compare with the campaign file."""
        return [PY, f"{HERE}/nq25_refcheck.py", "--layer", str(L), "--expert", str(E), "--out", self.root, "--",
                ] + self.enc_cmd_ref(L, E)

    def vision_flags(self):
        """T12's flag spelling for the text/vision H blend (nq_layer --stats-mm ROOT --mm-w w; nq25_t23 mirrors it)."""
        c = self.cfg
        if not c.get("vision_root"):
            return []
        t = c.get("vision_args") or "--stats-mm {vstats} --mm-w {vw}"
        return shlex.split(t.format(vstats=f"{self.root}/_stats_mm", vw=c["vision_weight"], vroot=c["vision_root"],
                                    vversion=c["vision_version"]))

    def fin_cmd(self, L):
        """nq25_finalize.py: [refcheck] + nq_layer finalize [+ --check-decode] + safetensors convert + st check."""
        c = self.cfg
        fin = [PY, f"{T12}/nq_layer.py", "--layer", str(L), "--stats", f"{self.root}/_stats", "--out", self.root,
               "--source", c["source"], "--fixed-set", c["fixed_set"], "--experts", "0:0"]
        pol = self.sched["nq_check"]
        nq_chk = pol == "all" or (pol == "first" and not any((self.lay(x).get("check_decode") or {}).get("nq") is True
                                                             for x in self.layers))
        if nq_chk:
            fin.append("--check-decode")
        cmd = [PY, f"{HERE}/nq25_finalize.py", "--layer", str(L), "--out", self.root, "--fin-cmd", json.dumps(fin),
               "--n-decode", str(self.sched["n_decode"])]
        if self.sched["refcheck"] and self.layer_encoder(L) != "nq_layer":
            import random
            E = random.Random(L).choice(self.experts)
            cmd += ["--ref-expert", str(E), "--ref-cmd", json.dumps(self.enc_cmd_ref(L, E))]
        return cmd

    def enc_cmd_ref(self, L, E):
        c = self.cfg
        extra = shlex.split(c["enc_args"] or "") + self.vision_flags()
        return [PY, f"{T12}/nq_layer.py", "--layer", str(L), "--stats", f"{self.root}/_stats", "--out", f"{self.root}/_ref",
                "--source", c["source"], "--fixed-set", c["fixed_set"], "--experts", f"{E}:{E + 1}", "--no-finalize"] + extra

    def spot_cmd(self, L):
        c = self.cfg
        return [PY, f"{HERE}/nq25_spot.py", "--layer", str(L), "--stats", f"{self.root}/_stats", "--out", self.root,
                "--source", c["source"], "--fixed-set", c["fixed_set"],
                "--l2-thr", str(self.sched["spot_l2"]), "--l4-thr", str(self.sched["spot_l4"])] + self._spot_override(L) + \
               (["--h-fn", c["spot_h_fn"]] if c.get("spot_h_fn") else []) + \
               (["--vision-stats", f"{self.root}/_stats_mm", "--vision-weight", str(c["vision_weight"])] if c.get("vision_root") else [])

    def _spot_override(self, L):
        """partial-expert campaigns (dry runs): choose within the encoded range."""
        if len(self.experts) == NEXP:
            return []
        fs = set(int(e) for e in (jload(self.cfg["fixed_set"]) or {}).get("fixed_set", {}).get(str(L), []))
        f = [e for e in self.experts if e in fs]
        o = [e for e in self.experts if e not in fs]
        if not f or not o:
            return ["--experts", f"{self.experts[0]},{self.experts[-1]}"]
        return ["--experts", f"{f[0]},{o[len(o) // 2]}"]

    # -------- workers
    def launch(self, kind, L, gpu, cmd, extra=None, nice=0):
        wid = f"{kind}-L{L}" + (f"-{extra}" if extra else "") + f"-{int(now())}"
        logp = f"{self.root}/logs/{wid}.log"
        lf = open(logp, "w")
        lf.write(f"# {mel()} gpu {gpu}\n# {' '.join(shlex.quote(x) for x in cmd)}\n"); lf.flush()
        pre = ["nice", "-n", str(nice)] if nice else []
        env = worker_env(gpu if gpu is not None else "", self.sched["omp"])
        if self.cfg.get("vision_root"):
            env.update(NQ25_VISION_STATS=f"{self.root}/_stats_mm", NQ25_VISION_WEIGHT=str(self.cfg["vision_weight"]))
        if kind == "up":                                   # the upload job is the only one that needs the HF token
            env = dict(os.environ, PYTHONUNBUFFERED="1", HF_XET_HIGH_PERFORMANCE="1")   # 12.8 -> 81-94 MB/s measured
        p = subprocess.Popen(pre + cmd, stdout=lf, stderr=subprocess.STDOUT, env=env,
                             cwd=HERE, start_new_session=True)
        lf.close()
        pp = psutil.Process(p.pid)
        self.procs[wid] = p
        uu = self._g[gpu]["uuid"] if gpu is not None else None
        self._gpu_used_tick.add(gpu)
        self.st["workers"][wid] = dict(kind=kind, layer=L, gpu=gpu, pid=p.pid, ctime=pp.create_time(), t0=now(),
                                       pre_pids=sorted(q for q, (u, _) in self._procs.items() if u == uu), hpid=None,
                                       log=logp, extra=extra, reserve_mb=self.reserve_mb(kind), max_rss_gb=0.0,
                                       max_vram_mb=0.0)
        self.log(f"launch {wid} gpu{gpu} pid {p.pid}")
        return wid

    def reserve_mb(self, kind):
        return 1024 * (self.sched["spot_vram_gb"] if kind == "spot" else self.sched["vram_gb"])

    def alive(self, wid):
        w = self.st["workers"][wid]
        p = self.procs.get(wid)
        if isinstance(p, subprocess.Popen):
            return p.poll() is None
        try:                                            # adopted from a previous driver
            pp = psutil.Process(w["pid"])
            return abs(pp.create_time() - w["ctime"]) < 1 and pp.status() != psutil.STATUS_ZOMBIE
        except psutil.Error:
            return False

    def exit_code(self, wid):
        p = self.procs.get(wid)
        if isinstance(p, subprocess.Popen):
            return p.returncode
        return None                                     # adopted: unknown; outcome judged from files / log

    def kill(self, wid, why):
        w = self.st["workers"][wid]
        self.log(f"kill {wid} pid {w['pid']} ({why})")
        try:
            pp = psutil.Process(w["pid"])
            if abs(pp.create_time() - w["ctime"]) < 1:          # our own process only
                os.killpg(w["pid"], signal.SIGTERM)
                try:
                    pp.wait(20)
                except psutil.TimeoutExpired:
                    os.killpg(w["pid"], signal.SIGKILL)
        except (psutil.Error, ProcessLookupError, PermissionError):
            pass
        w["killed"] = why

    # -------- per-expert timing from worker logs
    PAT = re.compile(r"^\[L(\d+) E(\d+)\] (\d+)s")

    def parse_log(self, w):
        out = {}
        try:
            with open(w["log"]) as f:
                for line in f:
                    m = self.PAT.match(line)
                    if m:
                        out[int(m.group(2))] = int(m.group(3))
        except FileNotFoundError:
            pass
        return out

    # -------- reconcile
    def reconcile(self):
        """Rebuild layer states from the filesystem; adopt/forget previous workers."""
        for wid, w in list(self.st["workers"].items()):
            if w.get("end") is None and not self.alive(wid):
                w["end"] = now(); w["rc"] = "exited while driver was down"
                self.finish(wid)                     # judged from its files / log, like a live exit
        for L in self.layers:
            ly = self.lay(L)
            done = self.done_experts(L)
            ly["n_done"] = len(done & set(self.experts))
            man = jload(f"{self.root}/L{L}/manifest.json")
            if ly["state"] in ("pending", "encoding") and ly["n_done"] == len(self.experts):
                ly["state"] = "encoded"
            if ly["state"] in ("pending", "encoding") and ly["n_done"] and ly["state"] == "pending":
                ly["state"] = "encoding"
            if ly.get("remote_verified"):
                continue                                # HF is the truth for this layer; nothing local needed
            if man is None and ly["state"] in ("finalized", "checked", "spotchecked", "uploaded"):
                ly["state"] = "encoded" if ly["n_done"] == len(self.experts) else "encoding"
                ly["events"].append([mel(), "manifest missing -> re-finalize"])
            for k, c in ly["chunks"].items():
                if c["status"] == "running" and (c.get("wid") is None or self.st["workers"].get(c["wid"], {}).get("end")):
                    c["status"] = "pending"

    # -------- scheduling
    def chunks_for(self, L):
        ly = self.lay(L)
        if not ly["chunks"]:
            e0, e1 = self.experts[0], self.experts[-1] + 1
            ch = self.sched["chunk"]
            for a in range(e0, e1, ch):
                ly["chunks"][f"{a}:{min(a + ch, e1)}"] = dict(status="pending", attempts=0)
        return ly["chunks"]

    def running(self, kind=None):
        return [wid for wid, w in self.st["workers"].items() if w.get("end") is None and (kind is None or w["kind"] == kind)]

    def gpu_budget(self, g, procs):
        """effective free MiB per allowed GPU and our worker count per (gpu, kind)."""
        eff, cnt = {}, collections.Counter()
        for i in self.gpus_allowed:
            if i in g:
                eff[i] = g[i]["free"]
        # nvidia-smi reports host PIDs (not our container PIDs): attribute to each worker the host pid that appeared
        # on its GPU after its launch (one launch per GPU per tick keeps the attribution unique)
        uuid2i = {v["uuid"]: i for i, v in g.items()}
        taken = {w.get("hpid") for w in self.st["workers"].values() if w.get("end") is None and w.get("hpid")}
        for wid in sorted(self.running(), key=lambda x: self.st["workers"][x]["t0"]):
            w = self.st["workers"][wid]
            if w.get("hpid") is None and "pre_pids" in w:
                new = [p for p, (u, _) in procs.items() if uuid2i.get(u) == w["gpu"] and p not in w["pre_pids"] and p not in taken]
                if len(new) == 1:
                    w["hpid"] = new[0]; taken.add(new[0])
        for wid in self.running():
            w = self.st["workers"][wid]
            cnt[(w["gpu"], w["kind"])] += 1
            used = procs[w["hpid"]][1] if w.get("hpid") in procs else 0.0
            w["vram_mb"] = used
            w["max_vram_mb"] = max(w.get("max_vram_mb", 0), used)
            if w["gpu"] in eff:
                eff[w["gpu"]] -= max(0.0, w["reserve_mb"] - used)     # reserved but not yet allocated
        return eff, cnt

    @staticmethod
    def _children(pid):
        try:
            return psutil.Process(pid).children(recursive=True)
        except psutil.Error:
            return []

    def pick_gpu(self, eff, cnt, kind):
        need = self.reserve_mb(kind) + 1024 * self.sched["margin_gb"]
        cap = self.sched["workers_per_gpu"]
        cands = []
        for i, f in eff.items():
            if f < need or i in self._gpu_used_tick:
                continue
            if kind == "enc" and cnt[(i, "enc")] >= cap:
                continue
            if kind == "spot" and cnt[(i, "enc")] < cap and self._enc_waiting:
                continue                                     # spot only beyond the encode cap
            cands.append((f, -sum(v for (j, _), v in cnt.items() if j == i), i))
        if not cands:
            return None
        return max(cands)[2]

    def host_ok(self):
        return host_headroom_gb() >= self.sched["rss_gb"] + self.sched["host_margin_gb"]

    def next_chunk(self):
        """lowest layer first (finish layers, so finalize/upload can stream)."""
        for L in self.layers:
            ly = self.lay(L)
            if ly["state"] not in ("pending", "encoding"):
                continue
            done = self.done_experts(L)
            for k, c in self.chunks_for(L).items():
                if c["status"] != "pending":
                    continue
                a, b = rng(k)
                if all(e in done for e in range(a, b)):
                    c["status"] = "done"; continue
                if c["attempts"] >= self.sched["max_attempts"]:
                    c["status"] = "failed"; continue
                if c.get("retry_after", 0) > now():
                    continue
                why = self.layer_gate(L)
                if why:
                    if self._holds.get(L) != why:              # log each layer's hold reason once per change
                        self.log(f"L{L}: holding ({why})")
                    self._holds[L] = why
                    break
                self._holds.pop(L, None)
                if pin_layer(self.root, self.cfg, L, self.sched["min_shards"], force=self._final.get(L)) is None:
                    self._note(f"L{L}: stats not ready (< {self.sched['min_shards']} shards); holding layer")
                    break
                return L, k, c
        return None

    def layer_gate(self, L):
        """per-layer launch gate (coordinator 2026-09-29): None = go, else the reason to hold. Stats are checked only
        until the layer is pinned (a pinned layer is never re-gated); MMW / T23 gates apply to every new launch.
          text   : T19's LOCAL final marker ROOT/final/L{L}.json (atomic; all 25 plan shards) -> pin its version_dir
                   (+ optionally, gate_backup, the flashblade done_full state; off: backup is insurance, not a gate)
          vision : the vision root's stats/L{L} meta shard set == its plan.json
          mm_w   : the mmw_go file exists and holds the campaign's vision_weight (adopted if nothing encoded yet)
          t23    : a T23 encoder waits for the t23_go file (T23 gate pass) if one is configured"""
        why = self.mmw_gate()
        if why:
            return why
        enc = self.layer_encoder(L)
        go = self.sched.get("t23_go")
        if enc in T23_ENCODERS and go and not os.path.exists(go):
            return f"T23 gate: {go} not present"
        if os.path.lexists(f"{self.root}/_stats/stats/L{L}") or self.sched.get("stats_gate") != "final":
            return None
        c = self.cfg
        root = c["stats_root"]
        fm = jload(f"{root}/final/L{L}.json")
        if fm is None:
            return "text: no T19 final marker yet"
        vd = fm.get("version_dir")
        if not vd or not os.path.isdir(vd) or fm.get("layer") not in (L, str(L)):
            return f"text: final marker bad ({vd})"
        if self.sched.get("gate_backup"):
            ok = any((jload(p) or {}).get(str(L), {}).get("version") == os.path.basename(vd)
                     for p in glob.glob(f"{root}/logs/fb_backup_state_full*.json"))
            if not ok:
                return f"text {os.path.basename(vd)} final but no done_full backup marker yet"
        if c.get("vision_root"):
            vr = c["vision_root"]
            vvd = os.path.realpath(f"{vr}/{c['vision_version']}/L{L}")
            m, plan = jload(f"{vvd}/meta.json"), jload(f"{vr}/plan.json")
            if m is None or plan is None:
                return "vision stats/plan missing"
            have = {s_["shard"] if isinstance(s_, dict) else s_ for s_ in m.get("shards", [])}
            if have != {int(k) for k in plan["shards"]}:
                return f"vision {os.path.basename(vvd)} has {len(have)}/{len(plan['shards'])} shards"
        self._final[L] = {"_stats": vd}
        return None

    def mmw_gate(self):
        """MMW_GO (written by the lead after T12's w A/B) holds the chosen mm_w. vision_weight is in config_id, so a
        different value is adopted only while no layer has pinned stats / encoded anything; otherwise hold + alert."""
        p = self.sched.get("mmw_go")
        if not p:
            return None
        try:
            w = float(open(p).read().split()[0])
        except FileNotFoundError:
            return f"mm_w gate: {p} not present"
        except (ValueError, IndexError):
            return f"mm_w gate: {p} unparsable"
        if w == float(self.cfg.get("vision_weight") or 0.0):
            return None
        started = [L for L in self.layers if self.lay(L)["state"] != "pending" or os.path.lexists(f"{self.root}/_stats/stats/L{L}")
                   or os.path.isdir(f"{self.root}/L{L}/experts") and os.listdir(f"{self.root}/L{L}/experts")]
        if started:
            if not getattr(self, "_mmw_alerted", False):
                self.alert("mmw_mismatch", None, f"MMW_GO says {w} but layers {started[:8]} already use "
                           f"{self.cfg.get('vision_weight')}; holding new launches")
                self._mmw_alerted = True
            return f"mm_w gate: {p}={w} != campaign {self.cfg.get('vision_weight')} with layers started"
        old = self.cfg.get("vision_weight")
        self.cfg["vision_weight"] = w
        cj = jload(f"{self.root}/campaign.json")
        cj["frozen"]["vision_weight"] = w
        cj["config_id"] = config_id(cj["frozen"]); cj["updated"] = mel()
        jdump(cj, f"{self.root}/campaign.json")
        self.alert("mmw_adopted", None, f"vision_weight {old} -> {w} from {p} (nothing encoded yet); config_id {cj['config_id']}")
        return None

    def refresh_manifests(self):
        """fixed_set.json only feeds the manifest's default_allocation: when its content differs from what a finalized
        layer's manifest recorded, rewrite default_allocation (T12's own nq_layer.default_allocation) + the campaign
        block, then re-queue the upload (only manifest.json differs, so only it is sent). Every 60 s."""
        if not self.sched.get("remanifest") or now() - getattr(self, "_rm_t", 0) < 60:
            return
        self._rm_t = now()
        fs = self.cfg["fixed_set"]
        if not os.path.exists(fs):
            return
        import hashlib, types
        cur = hashlib.sha256(open(fs, "rb").read()).hexdigest()
        for L in self.layers:
            ly = self.lay(L)
            if ly["state"] not in ("checked", "spotchecked", "uploaded"):
                continue
            mp = f"{self.root}/L{L}/manifest.json"
            man = jload(mp)
            if man is None or (man.get("default_allocation") or {}).get("sha256") == cur:
                continue
            sys.path.insert(0, T12)
            import nq_layer as NL
            old = (man.get("default_allocation") or {}).get("sha256")
            man["default_allocation"] = NL.default_allocation(types.SimpleNamespace(fixed_set=fs, stats=f"{self.root}/_stats"), L)
            fsd = jload(fs) or {}
            if str(fsd.get("schema", "")).endswith("-v3"):         # T25 global rescore: variable count per layer
                da = man["default_allocation"]                     # (T12's FIXED_RULE text says "top 26 per layer")
                da.update(rule=fsd.get("rule"), n=len(da.get("level4_experts") or []), budget=fsd.get("budget"),
                          floor=fsd.get("floor"), cap=fsd.get("cap"), estimator=fsd.get("estimator"))
            man.setdefault("campaign", {})["fixed_set"] = dict(path=fs, sha16=cur[:16], refreshed=mel())
            jdump(man, mp)
            ly.pop("upload", None); ly.pop("remote_verified", None); ly["up_attempts"] = {}
            if ly["state"] == "uploaded":
                ly["state"] = "checked"
            ly["events"].append([mel(), f"manifest default_allocation refreshed ({str(old)[:8]} -> {cur[:8]}); re-upload"])
            self.log(f"L{L}: fixed_set changed ({str(old)[:8]} -> {cur[:8]}): manifest default_allocation refreshed, re-uploading")

    def tick(self):
        g, procs = gpu_info()
        self._g, self._procs, self._gpu_used_tick = g, procs, set()
        # ---- reap + monitor
        for wid in self.running():
            w = self.st["workers"][wid]
            if self.alive(wid):
                try:
                    rss = tree_rss_gb(psutil.Process(w["pid"]))
                except psutil.Error:
                    rss = 0.0
                w["max_rss_gb"] = round(max(w.get("max_rss_gb", 0), rss), 2)
                if rss > self.sched["rss_gb"]:
                    self.kill(wid, f"rss {rss:.1f} GB > {self.sched['rss_gb']}")
                continue
            w["end"] = now(); w["rc"] = self.exit_code(wid)
            self.finish(wid)
        # ---- layer transitions
        for L in self.layers:
            ly = self.lay(L)
            if ly["state"] in ("pending", "encoding"):
                n = len(self.done_experts(L) & set(self.experts))
                ly["n_done"] = n
                if n == len(self.experts):
                    ly["state"] = "encoded"; ly["t_encoded"] = now(); ly["events"].append([mel(), "encoded"])
                    self.log(f"L{L} encoded ({n} experts)")
        self.refresh_manifests()
        # ---- launches
        eff, cnt = self.gpu_budget(g, procs)
        if self.code_drift():
            return 0                                  # running workers continue; nothing new until resolved
        nxt = self.next_chunk()
        self._enc_waiting = nxt is not None
        launched = 0
        # finalize (+ check-decode) first: short, unblocks spot/upload
        for L in self.layers:
            ly = self.lay(L)
            if ly["state"] != "encoded" or ly.get("fin_wid") and self.st["workers"][ly["fin_wid"]].get("end") is None:
                continue
            if len(self.running("fin")) >= self.sched["max_fin"] or ly.get("fin_attempts", 0) >= self.sched["max_attempts"]:
                continue
            if not self.host_ok():
                break
            i = self.pick_gpu(eff, cnt, "fin")
            if i is None:
                break
            ly["fin_attempts"] = ly.get("fin_attempts", 0) + 1
            ly["fin_wid"] = self.launch("fin", L, i, self.fin_cmd(L))
            eff[i] -= self.reserve_mb("fin"); cnt[(i, "fin")] += 1; launched += 1
        # encode chunks
        while nxt is not None and len(self.running()) < self.sched["max_workers"]:
            if not self.host_ok():
                self._note("host headroom low; holding launches")
                break
            i = self.pick_gpu(eff, cnt, "enc")
            if i is None:
                break
            L, k, c = nxt
            a, b = rng(k)
            c["attempts"] += 1; c["status"] = "running"
            ly = self.lay(L)
            ly.setdefault("encoder", self.layer_encoder(L))
            c["wid"] = self.launch("enc", L, i, self.enc_cmd(L, a, b), extra=k.replace(":", "-"))
            if ly["state"] == "pending":
                ly["state"] = "encoding"; ly["t_start"] = now()
                ly["stats"] = self.calib_desc(L)
            eff[i] -= self.reserve_mb("enc"); cnt[(i, "enc")] += 1; launched += 1
            nxt = self.next_chunk()
        self._enc_waiting = nxt is not None
        # spot checks on spare capacity
        if self.sched["spot"]:
            for L in self.layers:
                ly = self.lay(L)
                if ly["state"] in ("pending", "encoding") or ly.get("spot") or ly.get("spot_attempts", 0) >= self.sched["max_attempts"]:
                    continue
                if ly.get("spot_wid") and self.st["workers"][ly["spot_wid"]].get("end") is None:
                    continue
                if len(self.running("spot")) >= self.sched["spot_workers"] or not self.host_ok():
                    break
                i = self.pick_gpu(eff, cnt, "spot")
                if i is None:
                    break
                ly["spot_attempts"] = ly.get("spot_attempts", 0) + 1
                ly["spot_wid"] = self.launch("spot", L, i, self.spot_cmd(L), nice=10)
                eff[i] -= self.reserve_mb("spot"); cnt[(i, "spot")] += 1
        # uploads (no GPU): per checked layer; "dry" writes the plan listing, "go" uploads
        mode = self.sched["upload"]
        if mode in ("dry", "go"):
            for L in self.layers:
                ly = self.lay(L)
                if ly["state"] not in ("checked", "spotchecked") or len(self.running("up")) >= self.sched["max_up"]:
                    continue
                if (ly.get("upload") or {}).get("mode") == mode or ly.get("up_attempts", {}).get(mode, 0) >= self.sched["max_attempts"]:
                    continue
                if ly.get("up_wid") and self.st["workers"][ly["up_wid"]].get("end") is None:
                    continue
                ly.setdefault("up_attempts", {})[mode] = ly.get("up_attempts", {}).get(mode, 0) + 1
                cmd = [PY, f"{HERE}/nq25_upload.py", "--root", self.root, "--layers", str(L), "--repo", self.sched["repo"]] + \
                      (["--go"] if mode == "go" else [])
                ly["up_wid"] = self.launch("up", L, None, cmd, extra=mode)
        return launched

    def alert(self, kind, L, msg, **kw):
        """ALERTS.jsonl (for main / the lead) + campaign.log; the per-layer record keeps it too."""
        rec = dict(time=mel(), kind=kind, layer=L, msg=msg, **kw)
        with open(f"{self.root}/{ALERTS}", "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        if L is not None:
            self.lay(L).setdefault("alerts", []).append(rec)
        self.log(f"ALERT {kind} L{L}: {msg}")

    def code_drift(self):
        """the files that define the bytes changed since the campaign accepted them -> hold new launches."""
        if not self.sched["code_guard"] or now() - getattr(self, "_code_t", 0) < 60:
            return getattr(self, "_drift", False)
        self._code_t = now()
        ci = code_info()
        keys = ["code_id"] + (["t23_id"] if any(self.layer_encoder(L) in T23_ENCODERS for L in self.layers) else [])
        bad = [k for k in keys if ci[k] != self.code.get(k)]
        if bad and not getattr(self, "_drift", False):
            grp = {"code_id": CODE_FILES, "t23_id": T23_FILES}
            mine = {f"{d}/{f}" for k in bad for d, fs in grp[k].items() for f in fs}
            ch = [f for f, h in ci["files"].items() if f in mine and self.code["files"].get(f) != h]
            self.alert("code_drift", None, f"{bad} changed ({ch}); holding new launches. Restart with --accept-code "
                                           f"to adopt the new code (layers keep their recorded code ids).", files=ch)
        self._drift = bool(bad)
        return self._drift

    def hf_reconcile(self):
        """B1: the HF repo is the truth. Layers whose remote manifest + every listed file verify (size + LFS sha256) and
        whose campaign config_id equals ours are done (nothing local needed); every other layer's local 'uploaded' claim
        is dropped (it is re-finalized / re-uploaded / re-encoded from whatever local files exist)."""
        sys.path.insert(0, HERE)
        import nq25_upload as U
        t0 = now()
        rem = U.remote_layers(self.sched["repo"], cache=f"{self.root}/_remote")
        cid = config_id(self.cfg)
        n_ok = 0
        for L in self.layers:
            ly = self.lay(L)
            r = rem.get(L)
            if r and r["verified"] and r["config_id"] == cid:
                n_ok += 1
                if not ly.get("remote_verified"):
                    ly["events"].append([mel(), "remote verified on HF -> done"])
                ly.update(state="uploaded", remote_verified=dict(r, time=mel()))
                lm, rm = f"{self.root}/L{L}/manifest.json", f"{self.root}/_remote/layers/L{L}/manifest.json"
                if not os.path.exists(lm) and os.path.exists(rm):      # wiped: keep the remote manifest locally so a
                    os.makedirs(os.path.dirname(lm), exist_ok=True)       # fixed_set change can still refresh it
                    shutil.copy(rm, lm)
                ly.setdefault("upload", {}).update(mode="go", status="uploaded (remote verified)")
                if r.get("code_id") != self.code.get("code_id"):
                    self.log(f"L{L}: remote layer code_id {r.get('code_id')} != campaign {self.code.get('code_id')} "
                             f"(kept: config matches)")
                continue
            if r and r["verified"] and r["config_id"] != cid:
                self.alert("remote_config", L, f"HF layer has config_id {r['config_id']} != campaign {cid}; "
                                               f"it will be REPLACED by this campaign's encode")
            if ly.pop("remote_verified", None) or ly["state"] == "uploaded":
                ly["events"].append([mel(), f"local 'uploaded' not confirmed on HF ({(r or {}).get('reason', 'absent')})"])
                ly["state"] = "encoded" if self.done_experts(L) >= set(self.experts) else "encoding"
                ly.pop("upload", None); ly["fin_attempts"] = 0
        self.log(f"hf reconcile: {n_ok}/{len(self.layers)} layers verified on {self.sched['repo']} "
                 f"({len(rem)} remote layer dirs, {now() - t0:.0f}s)")

    def _note(self, msg):
        if getattr(self, "_last_note", None) != msg:
            self.log(msg)
        self._last_note = msg

    def finish(self, wid):
        w = self.st["workers"][wid]
        L = w["layer"]; ly = self.lay(L)
        dt = w["end"] - w["t0"]
        tail = ""
        try:
            tail = "".join(open(w["log"]).readlines()[-3:]).strip().replace("\n", " | ")[-300:]
        except FileNotFoundError:
            pass
        if w["kind"] == "enc":
            k = w["extra"].replace("-", ":")
            c = ly["chunks"][k]
            times = self.parse_log(w)
            for E, s in times.items():
                ly["experts"][str(E)] = dict(s=s, gpu=w["gpu"], wid=wid, t=w["end"])
            a, b = rng(k)
            done = self.done_experts(L)
            miss = [e for e in range(a, b) if e not in done]
            if not times and not miss:                 # t23b logs no per-expert lines: wall / experts new in this chunk
                new = [e for e in range(a, b) if str(e) not in ly["experts"]]
                for E in new:
                    ly["experts"][str(E)] = dict(s=round(dt / len(new), 1), gpu=w["gpu"], wid=wid, t=w["end"], est=True)
                times = {E: None for E in new}
            c["status"] = "pending" if miss else "done"
            if miss:
                c["retry_after"] = now() + 60 * c["attempts"]
                self.log(f"{wid} rc {w['rc']} after {dt:.0f}s: {len(miss)} experts missing -> retry ({c['attempts']}/{self.sched['max_attempts']}) :: {tail}")
                if c["attempts"] >= self.sched["max_attempts"]:
                    self.alert("chunk_failed", L, f"chunk {k} failed {c['attempts']}x: {tail}")
            else:
                self.log(f"{wid} done {dt:.0f}s ({len(times)} encoded here, {dt/max(1,len(times)):.1f} s/expert wall) "
                         f"rss {w['max_rss_gb']} GB vram {w['max_vram_mb']:.0f} MiB")
        elif w["kind"] == "fin":
            out = open(w["log"]).read() if os.path.exists(w["log"]) else ""
            if "REFCHECK MISMATCH" in out:
                self.refcheck_fallback(L, out)
                return
            rc_ = jload(f"{self.root}/refcheck/L{L}.json")
            if rc_ and "refcheck" in out:
                ly["refcheck"] = {k: rc_.get(k) for k in ("expert", "encoder", "match", "n_tensors", "tensor_bytes", "s")}
            man = jload(f"{self.root}/L{L}/manifest.json")
            ok_fin = man is not None and man.get("n_experts") == len(self.experts) and f"finalized {len(self.experts)} experts" in out
            m = re.search(r"shard-file decode == artifact decode: (True|False) \((\d+) mismatches\)", out)
            nq_ok = None if m is None else m.group(1) == "True"
            st = {k: (re.search(pat + r": (True|False)", out) or [None, None])[1] == "True"
                  for k, pat in (("roundtrip", r"safetensors round-trip == tp\.pt \([^)]*\)"),
                                 ("artifact", r"safetensors artifact == E\.pt artifact"),
                                 ("decode", r"safetensors decode == artifact decode"))}
            ran_nq = "--check-decode" in self.fin_cmd_log(w)
            ok_chk = all(st.values()) and (nq_ok is True if ran_nq else nq_ok is not False)
            ok_st_files = man is not None and all(f.endswith(".safetensors") for f in man.get("files", {}))
            ver = self.verify_manifest(L, man) if ok_fin and ok_st_files else "no safetensors manifest"
            ly["finalize"] = dict(ok=ok_fin, s=round(dt), wid=wid, manifest_verify=ver, rc=w["rc"], rss_gb=w["max_rss_gb"])
            ly["check_decode"] = dict(ok=ok_chk, nq=nq_ok, nq_mismatches=int(m.group(2)) if m else None, st=st)
            if ok_fin and ver == "ok":
                self.annotate_manifest(L)
                ly["state"] = "checked" if ok_chk else "finalized"
                ly["events"].append([mel(), ly["state"]])
                if not ok_chk:
                    self.alert("check_decode", L, f"decode check failed: nq {nq_ok} st {st}; layer NOT uploaded")
            self.log(f"L{L} finalize ok={ok_fin} verify={ver} check nq={nq_ok} st={st} refcheck="
                     f"{(ly.get('refcheck') or {}).get('match')} ({dt:.0f}s) :: {tail if not ok_chk else ''}")
        elif w["kind"] == "up":
            mode = w["extra"]
            rec = jload(f"{self.root}/uploads/L{L}.json" if mode == "go" else f"{self.root}/uploads/L{L}.plan.json")
            if rec and rec.get("status") in ("uploaded", "dry-run") and (mode == "go") == (rec["status"] == "uploaded"):
                ly["upload"] = dict(mode=mode, status=rec["status"], commit=rec.get("commit"), bytes=rec.get("bytes_to_send"),
                                    MBps=rec.get("MBps"), s=round(dt))
                if mode == "go":
                    ly["state"] = "uploaded"; ly["events"].append([mel(), "uploaded"])
                self.log(f"L{L} upload {mode}: {rec['status']} {rec.get('bytes_to_send', 0) / 2**30:.2f} GiB {rec.get('MBps')} MB/s")
            else:
                self.log(f"L{L} upload {mode} failed rc {w['rc']} :: {tail}")
                if mode == "go" and ly.get("up_attempts", {}).get(mode, 0) >= self.sched["max_attempts"]:
                    self.alert("upload", L, f"upload failed {self.sched['max_attempts']}x: {tail}")
        elif w["kind"] == "spot":
            sp = jload(f"{self.root}/spot/L{L}.json")
            if sp:
                ly["spot"] = dict(flag=sp.get("flag"), summary=sp.get("summary"), s=round(dt))
                if ly["state"] == "checked":
                    ly["state"] = "spotchecked"
                self.log(f"L{L} spotcheck flag={sp.get('flag')} {sp.get('summary')}")
                if sp.get("flag"):
                    self.alert("spot_flag", L, f"nq worse than EXL3 beyond +{self.sched['spot_l2']}% L2 / "
                                               f"+{self.sched['spot_l4']}% L4: {sp.get('summary')}",
                               uploaded=ly["state"] == "uploaded")
            else:
                self.log(f"L{L} spotcheck failed rc {w['rc']} :: {tail}")

    @staticmethod
    def fin_cmd_log(w):
        """the command line the driver wrote at the top of the worker log."""
        try:
            with open(w["log"]) as f:
                f.readline()
                return f.readline()
        except FileNotFoundError:
            return ""

    def refcheck_fallback(self, L, out):
        """batched output != reference on the refcheck expert: set the layer's batched E files aside, re-encode the
        whole layer with the reference nq_layer path (reusing the reference expert), alert main."""
        ly = self.lay(L)
        rc_ = jload(f"{self.root}/refcheck/L{L}.json") or {}
        ts = int(now())
        d = f"{self.root}/L{L}"
        aside = f"{d}.{ly.get('encoder', 'batch')}-mismatch-{ts}"
        os.rename(d, aside)
        os.makedirs(self.edir(L))
        E = rc_.get("expert")
        rp = f"{self.root}/_ref/L{L}/experts/E{E}.pt"
        if E is not None and os.path.exists(rp):
            shutil.copy2(rp, f"{self.edir(L)}/E{E}.pt")
        prev = ly.get("encoder")
        ly.update(encoder="nq_layer", state="pending", chunks={}, fin_attempts=0, n_done=0, spot=None, spot_attempts=0,
                  refcheck=dict(rc_, action=f"layer re-encoded with nq_layer; batched files moved to {aside}"))
        ly.pop("fin_wid", None); ly.pop("spot_wid", None)
        ly["events"].append([mel(), f"refcheck mismatch ({prev}) -> nq_layer re-encode"])
        self.alert("refcheck_mismatch", L, f"{prev} != nq_layer on E{E} ({rc_.get('n_mismatch')} paths, e.g. "
                                           f"{(rc_.get('mismatches') or [])[:3]}); re-encoding L{L} with nq_layer",
                   aside=aside)

    def calib_desc(self, L):
        c = self.cfg
        d = dict(text=dict(root=c["stats_root"], version=c["stats_version"],
                           **stats_desc(os.readlink(f"{self.root}/_stats/stats/L{L}"))))
        if c.get("vision_root"):
            d["vision"] = dict(root=c["vision_root"], version=c["vision_version"],
                               **stats_desc(os.readlink(f"{self.root}/_stats_mm/stats/L{L}")))
            d["blend"] = dict(rule="H = (1-w) H_text/tr + w H_vision/tr (per expert)", w_vision=c["vision_weight"])
        return d

    def annotate_manifest(self, L):
        """add the campaign's calibration record to L{L}/manifest.json (tp shard bytes/shas untouched)."""
        p = f"{self.root}/L{L}/manifest.json"
        man = jload(p)
        ly = self.lay(L)
        ci = code_info()
        man["campaign"] = dict(config_id=config_id(self.cfg), calibration=self.calib_desc(L),
                               encoder=self.layer_encoder(L), encoder_cmd=self.cfg["encoder_cmd"] or None,
                               enc_args=self.cfg["enc_args"], vision_flags=" ".join(self.vision_flags()),
                               refcheck=ly.get("refcheck"), check_decode=ly.get("check_decode"),
                               fixed_set=dict(path=self.cfg["fixed_set"], sha16=_sha16(self.cfg["fixed_set"])),
                               source=self.cfg["source"],
                               code=dict(code_id=ci["code_id"], t23_id=ci["t23_id"], t25_id=ci["t25_id"],
                                         git_head=ci["git_head"], dirty=ci["dirty"], files=ci["files"],
                                         campaign_code_id=self.code.get("code_id")),
                               repo_config="threads/25-campaign/campaign.json",
                               driver="threads/25-campaign/nq25_campaign.py", time=mel())
        jdump(man, p)

    def verify_manifest(self, L, man):
        """re-hash the shard files against the manifest (nq_layer computed them; this catches later corruption)."""
        import hashlib
        for f, info in man["files"].items():
            p = f"{self.root}/L{L}/{f}"
            if not os.path.exists(p) or os.path.getsize(p) != info["bytes"]:
                return f"size mismatch {f}"
            hh = hashlib.sha256()
            with open(p, "rb") as fh:
                for b in iter(lambda: fh.read(1 << 24), b""):
                    hh.update(b)
            if hh.hexdigest() != info["sha256"]:
                return f"sha mismatch {f}"
        return "ok"

    # -------- throughput / status
    def status(self, write=True):
        tot = len(self.layers) * len(self.experts)
        done = 0; recent = []
        per_state = collections.Counter()
        for L in self.layers:
            ly = self.lay(L)
            per_state[ly["state"]] += 1
            done += ly.get("n_done", 0)
            for E, x in ly["experts"].items():
                recent.append((x["t"], x["s"]))
        t = now()
        encw = [w for w in self.st["workers"].values() if w["kind"] == "enc"]
        t_first = min((w["t0"] for w in encw), default=t)
        last_h = [r for r in recent if r[0] > t - 3600]
        # rate: experts finished in the last hour (or since start if younger)
        span = min(3600.0, max(1.0, t - t_first))
        rate_h = len(last_h) / span * 3600
        s_exp = [s for _, s in recent]
        remaining = tot - done
        eta = t + remaining / rate_h * 3600 if rate_h > 0 else None
        if t - self.last_du[0] > 300 or not write:
            self.last_du = (t, du_bytes(self.root))
        free = shutil.disk_usage(self.root).free
        n_enc = len(self.running("enc"))
        st = dict(time=mel(t), experts_done=done, experts_total=tot, layers=dict(per_state),
                  experts_per_hour=round(rate_h, 1), running=dict(enc=n_enc, fin=len(self.running("fin")),
                                                                  spot=len(self.running("spot"))),
                  worker_s_per_expert=dict(mean=round(sum(s_exp) / len(s_exp), 1) if s_exp else None,
                                           n=len(s_exp)),
                  eta_all_layers=mel(eta) if eta else None,
                  disk_out_gb=round(self.last_du[1] / 2 ** 30, 2), disk_free_tb=round(free / 2 ** 40, 2),
                  flags=[L for L in self.layers if (self.lay(L).get("spot") or {}).get("flag")],
                  failed_chunks=[f"L{L}:{k}" for L in self.layers for k, c in self.lay(L)["chunks"].items() if c["status"] == "failed"])
        ts = [self.lay(L).get("t_start") for L in self.layers if self.lay(L).get("t_start")]
        te = sorted(self.lay(L)["t_encoded"] for L in self.layers if self.lay(L).get("t_encoded"))
        st["first_layer_start"] = mel(min(ts)) if ts else None
        win = [x for x in te if x > t - 6 * 3600]                   # layers fully encoded, last 6 h (or since start)
        hspan = min(6 * 3600.0, max(1.0, t - min(ts))) / 3600 if ts else None
        st["layers_encoded"] = len(te)
        st["layers_per_hour"] = round(len(win) / hspan, 2) if hspan else None
        holding = collections.defaultdict(list)
        for L in self.layers:                          # fresh (cheap file checks), not the last next_chunk verdicts
            ly = self.lay(L)
            if ly["state"] == "pending" and not ly.get("n_done"):
                why = self.layer_gate(L)
                if why:
                    holding[why].append(L)
        st["holding"] = {w: ",".join(map(str, v)) if len(v) < 6 else f"{len(v)} layers L{v[0]}..L{v[-1]}" for w, v in holding.items()}
        if write:
            jdump(st, f"{self.root}/status.json")
            with open(f"{self.root}/throughput.jsonl", "a") as f:
                f.write(json.dumps(st) + "\n")
        return st

    def all_done(self):
        up = self.sched["upload"]
        return all(self.lay(L)["state"] in ("checked", "spotchecked", "uploaded", "finalized") and
                   (up not in ("dry", "go") or (self.lay(L).get("upload") or {}).get("mode") == up or self.lay(L)["state"] == "finalized"
                    or self.lay(L).get("up_attempts", {}).get(up, 0) >= self.sched["max_attempts"]) and
                   (not self.sched["spot"] or self.lay(L).get("spot") or self.lay(L).get("spot_attempts", 0) >= self.sched["max_attempts"])
                   for L in self.layers) and not self.running()

    def blocked(self):
        """nothing running and nothing launchable (all remaining work failed)."""
        if self.running():
            return False
        for L in self.layers:
            ly = self.lay(L)
            if ly["state"] in ("pending", "encoding") and any(c["status"] in ("pending", "running") for c in self.chunks_for(L).values()):
                return False
            if ly["state"] == "encoded" and ly.get("fin_attempts", 0) < self.sched["max_attempts"]:
                return False
        return True


def cmd_run(a):
    c = Campaign(a)
    pf = f"{c.root}/driver.pid"
    old = jload(pf)
    if old:
        try:
            if abs(psutil.Process(old["pid"]).create_time() - old["ctime"]) < 1:
                raise SystemExit(f"driver already running (pid {old['pid']})")
        except psutil.Error:
            pass
    jdump(dict(pid=os.getpid(), ctime=psutil.Process().create_time(), started=mel()), pf)
    if c.cfg.get("vision_root") and c.cfg.get("encoder_cmd") and not c.cfg.get("vision_args"):
        raise SystemExit("--vision-root with a custom --encoder-cmd needs --vision-args (refusing a silently text-only encode)")
    build_shim(c.root, c.cfg)
    if c.sched["hf_reconcile"] and c.sched["upload"] == "go":
        c.hf_reconcile()
    c.log(f"campaign {c.root}: layers {c.cfg['layers']} experts {c.cfg['experts']} stats {c.cfg['stats_root']}/"
          f"{c.cfg['stats_version']} sched {c.sched}")
    c.reconcile(); c.save()
    stop = {"flag": False}
    def on_sig(*_):
        stop["flag"] = True
    signal.signal(signal.SIGTERM, on_sig); signal.signal(signal.SIGINT, on_sig)
    last_st = 0
    while not stop["flag"]:
        c.tick(); c.save()
        if now() - last_st > 300:
            st = c.status(); last_st = now()
            c.log(f"status {json.dumps(st)}")
        if c.all_done():
            c.log("all layers done"); break
        if c.blocked():
            c.log("blocked: remaining work failed max_attempts; stopping"); break
        time.sleep(c.sched["tick"])
    st = c.status(); c.save()
    c.log(f"driver exit; status {json.dumps(st)}  (workers keep running; restart adopts them)")


def cmd_stop(a):
    """SIGTERM the driver of ROOT (workers keep running and are adopted by the next run)."""
    open(f"{a.out}/STOPPED", "w").write(mel() + "\n")     # the watchdog leaves a STOPPED campaign alone
    d = jload(f"{a.out}/driver.pid")
    try:
        p = psutil.Process(d["pid"])
        if abs(p.create_time() - d["ctime"]) < 1:
            p.terminate(); print(f"sent SIGTERM to driver {d['pid']}")
    except (psutil.Error, TypeError):
        print("no running driver")


def cmd_status(a):
    c = Campaign(a)
    st = c.status(write=False)
    print(json.dumps(st, indent=1))
    for L in c.layers:
        ly = c.lay(L)
        if ly["state"] != "pending":
            print(f"L{L:<3} {ly['state']:<12} {ly.get('n_done', 0):>3}/{len(c.experts)}  "
                  f"fin {ly.get('finalize', {}).get('ok')} chk {ly.get('check_decode', {}).get('ok')} "
                  f"spot {(ly.get('spot') or {}).get('flag')} up {ly.get('upload', {}).get('status')}")


def cmd_upload(a):
    """standalone listing (never writes state.json; real uploads go through `run --upload go`)."""
    sys.path.insert(0, HERE)
    import nq25_upload as U
    st = jload(f"{a.out}/state.json")
    todo = sorted(int(L) for L, ly in st["layers"].items() if ly["state"] in ("checked", "spotchecked", "uploaded"))
    if a.upload_layers:
        l0, l1 = rng(a.upload_layers)
        todo = [L for L in todo if l0 <= L < l1]
    U.run(a.out, todo, go=False, repo=a.repo or SCHED_DEFAULTS["repo"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "status", "upload", "stop"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-root"); ap.add_argument("--stats-version")
    ap.add_argument("--layers", help="a:b (default 3:78)"); ap.add_argument("--experts", help="a:b (default 0:256)")
    ap.add_argument("--encoder", choices=("nq_layer", "t23", "t23b"), help="per-worker encoder (default nq_layer; t23 = "
                    "nq25_t23.py adapter; t23b = T23's nq_layer_batch.py drop-in)")
    ap.add_argument("--encoder-cmd", help="template, placeholders {py} {layer} {e0} {e1} {out} {stats} {src} {fixed_set} {t12};"
                                          " must write {out}/L{layer}/experts/E{E}.pt")
    ap.add_argument("--enc-args", help="extra flags passed through to nq_layer (encode and finalize)")
    ap.add_argument("--fixed-set"); ap.add_argument("--source")
    ap.add_argument("--vision-root", help="T26 vision capture root (T19 format); empty = text-only calibration")
    ap.add_argument("--vision-version", help="stats dir inside the vision root (default stats)")
    ap.add_argument("--vision-weight", type=float, help="w in H = (1-w) H_text/tr + w H_vision/tr (user: 0.25)")
    ap.add_argument("--vision-args", help="encoder flag template for the blend, placeholders {vstats} {vw} {vroot} {vversion},"
                                          " e.g. '--stats-mm {vstats} --mm-w {vw}' once T12/T26 fix the spelling")
    ap.add_argument("--spot-h-fn", help="module:function(cap, L, E, vision_stats, w) -> HG used by the spot check "
                                        "(must equal the encode H; required when --vision-root is set)")
    ap.add_argument("--force-config", action="store_true")
    for k, d in SCHED_DEFAULTS.items():
        if isinstance(d, bool):
            ap.add_argument(f"--{k.replace('_', '-')}", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        else:
            ap.add_argument(f"--{k.replace('_', '-')}", type=type(d), default=None)
    ap.add_argument("--accept-code", action="store_true", help="adopt the current code hashes as the campaign's")
    ap.add_argument("--upload-layers", help="a:b subset for upload")
    a = ap.parse_args()
    {"run": cmd_run, "status": cmd_status, "upload": cmd_upload, "stop": cmd_stop}[a.cmd](a)


if __name__ == "__main__":
    main()
