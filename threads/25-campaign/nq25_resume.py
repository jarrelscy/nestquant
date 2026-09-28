"""Thread 25 (B2): bring the campaign back after box death / a /tmp wipe with one idempotent command.

  resume.sh --check        report every precondition (+ download estimates); changes nothing
  resume.sh                fix what is missing (downloads, links) then (re)start the driver with --upload go
  nq25_resume.py pin       at launch: record code hashes + fixed_set sha16 into campaign.json (commit it)

HF (jarrelscy/GLM-5.3-NestQuant-2-4bit) is the source of truth for finished layers: the driver's hf_reconcile marks
every remotely verified layer with our config_id as done, so after a wipe only unfinished layers are re-encoded.
Steps (each is check -> fix, skip when already satisfied):
  1 env      venv python, LD_LIBRARY_PATH dirs, /tmp/nestquant/12-reference-encoder/bin/ninja -> venv ninja
  2 code     /home/coder/git/nestquant contains the pinned commit; file hashes == campaign.json code.files
             (after a wipe the code comes from GitHub: git clone/pull jarrelscy/nestquant -- needs the lead's creds)
  3 hf token present (never printed)
  4 fp8      zai-org/GLM-5.3 @ pinned revision into fp8_source.local_dir, sizes + LFS sha256 verified
  5 stats    text + vision stats restored from flashblade (fb_restore.py --set full), per-layer links present
  6 fixed    fixed_set.json present (manifest default_allocation only; not part of config_id)
  7 vision   vision graft files at the pinned revision, sha256 verified
  8 driver   nq25_campaign.py run (existing ROOT: plain resume; new ROOT: full config from campaign.json)
The token is looked up by huggingface_hub (HF_TOKEN / ~/.cache/huggingface/token) and never printed.
"""
import os, sys, json, glob, time, hashlib, argparse, subprocess
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
NQ = "/home/coder/git/nestquant"
CJ = f"{HERE}/campaign.json"
T19 = f"{NQ}/threads/19-full-capture"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def gb(n):
    return f"{n / 1e9:.1f} GB"


class R:
    def __init__(self):
        self.rows = []

    def add(self, step, ok, msg, est_s=0):
        self.rows.append(dict(step=step, ok=ok, msg=msg, est_s=est_s))
        eta = f"  (~{est_s / 60:.0f} min)" if est_s else ""
        print(f"[{'OK ' if ok else 'TODO' if ok is None else 'FAIL'}] {step:<7} {msg}{eta}", flush=True)
        return ok


# ------------------------------------------------------------------------------------------------ steps
def step_env(c, fix, r):
    e = c["env"]
    bad = [p for p in [e["python"]] + e["LD_LIBRARY_PATH"].split(":") if not os.path.exists(p)]
    if bad:
        return r.add("env", False, f"missing {bad} (venv/compat libs live on NFS home: restore them first)")
    link, tgt = e["ninja_link"]
    if os.path.realpath(link) != os.path.realpath(tgt) or not os.path.exists(link):
        if not fix:
            return r.add("env", None, f"ninja link {link} -> {tgt} missing/other")
        os.makedirs(os.path.dirname(link), exist_ok=True)
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(tgt, link)
    return r.add("env", True, f"python, LD_LIBRARY_PATH, ninja -> {os.readlink(link)}")


def step_code(c, fix, r, accept=False):
    want = c["code"]["nestquant_commit"]
    if not os.path.isdir(f"{NQ}/.git"):
        return r.add("code", False, f"{NQ} missing: git clone https://github.com/jarrelscy/nestquant {NQ} (lead's creds)")
    anc = subprocess.run(["git", "-C", NQ, "merge-base", "--is-ancestor", want, "HEAD"], capture_output=True)
    if anc.returncode:
        return r.add("code", False, f"HEAD does not contain pinned commit {want}: git -C {NQ} pull (lead's creds)")
    if not c["code"].get("files"):
        return r.add("code", None if not accept else True, f"contains {want}; no file hashes pinned yet (run 'pin' at launch)")
    sys.path.insert(0, HERE)
    import nq25_campaign as K
    ci = K.code_info()
    diff = [f for f, h in c["code"]["files"].items() if ci["files"].get(f) != h]
    if diff and not accept:
        return r.add("code", False, f"file hashes differ from the pinned campaign code: {diff} (re-pin or --accept-code)")
    return r.add("code", True, f"contains {want}; code_id {ci['code_id']} t23_id {ci['t23_id']}"
                 + (f" (accepted drift {diff})" if diff else ""))


def step_token(c, fix, r):
    from huggingface_hub import get_token
    return r.add("token", bool(get_token()), "HF token present" if get_token() else
                 "no HF token: put it in ~/.cache/huggingface/token (huggingface-cli login)")


def _hf_files(repo, rev, repo_type="model"):
    from huggingface_hub import HfApi
    out = {}
    for it in HfApi().list_repo_tree(repo, revision=rev, recursive=True, expand=True, repo_type=repo_type):
        if getattr(it, "size", None) is None:
            continue
        lfs = getattr(it, "lfs", None)
        out[it.path] = dict(size=it.size, sha256=lfs.sha256 if lfs else None)
    return out


def step_fp8(c, fix, r):
    s = c["fp8_source"]
    d, rev = s["local_dir"], s["revision"]
    rem = _hf_files(s["repo"], rev)
    mk = f"{d}/.nq25_verified.json"
    ver = json.load(open(mk)) if os.path.exists(mk) else {}
    miss = {p: v for p, v in rem.items() if not os.path.exists(f"{d}/{p}") or os.path.getsize(f"{d}/{p}") != v["size"]}
    unver = [p for p, v in rem.items() if p not in miss and v["sha256"] and
             ver.get(p) != [v["sha256"], os.path.getmtime(f"{d}/{p}")]]
    nmiss = sum(v["size"] for v in miss.values())
    nunv = sum(rem[p]["size"] for p in unver)
    if not fix:
        if not miss and not unver:
            return r.add("fp8", True, f"{len(rem)} files @ {rev[:8]} present + sha256 verified")
        return r.add("fp8", None, f"{len(miss)} files missing ({gb(nmiss)} @ ~{s['measured_MBps']} MB/s), "
                     f"{len(unver)} unverified ({gb(nunv)} to sha256)",
                     est_s=nmiss / 1e6 / s["measured_MBps"] + nunv / 1e9)
    if miss:
        from huggingface_hub import snapshot_download
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
        t0 = time.time()
        snapshot_download(s["repo"], revision=rev, local_dir=d, allow_patterns=list(miss), max_workers=16)
        print(f"  fp8: fetched {gb(nmiss)} in {time.time() - t0:.0f}s", flush=True)
    todo = [p for p, v in rem.items() if v["sha256"] and ver.get(p) != [v["sha256"], os.path.getmtime(f"{d}/{p}")]]
    with ThreadPoolExecutor(16) as ex:
        hs = dict(zip(todo, ex.map(lambda p: sha256(f"{d}/{p}"), todo)))
    bad = [p for p, h in hs.items() if h != rem[p]["sha256"]]
    for p, h in hs.items():
        if h == rem[p]["sha256"]:
            ver[p] = [h, os.path.getmtime(f"{d}/{p}")]
    json.dump(ver, open(mk + ".tmp", "w")); os.replace(mk + ".tmp", mk)
    if bad:
        for p in bad:
            os.remove(f"{d}/{p}")
        return r.add("fp8", False, f"sha256 mismatch {bad} (deleted; rerun)")
    return r.add("fp8", True, f"{len(rem)} files @ {rev[:8]} present + sha256 verified")


def _fb_latest(prefix, which):
    sys.path.insert(0, T19)
    from fb_restore import fetch_json
    from fb_backup19 import latest_name
    try:
        return fetch_json(f"{prefix}/{latest_name(which)}.json")
    except Exception:
        return None


def _stats_missing(root, layers):
    bad = []
    for L in layers:
        p = f"{root}/stats/L{L}"
        if not os.path.isdir(p) or not os.listdir(p):
            bad.append(L)
    return bad


def step_stats(c, fix, r):
    f = c["frozen"]
    lo, hi = map(int, f["layers"].split(":"))
    layers = range(lo, hi)
    ok = True
    for kind, root in (("text", f["stats_root"]), ("vision", f["vision_root"])):
        if not root:
            continue
        b = c["stats_backup"][kind]
        miss = _stats_missing(root, layers)
        if not miss:
            r.add("stats", True, f"{kind} {root}: stats/L{lo}..L{hi - 1} present"); continue
        lt = _fb_latest(b["prefix"], b["set"])
        if lt is None:
            r.add("stats", None, f"{kind}: {len(miss)} layers missing and no {b['set']} backup at {b['prefix']} (still being "
                  f"produced? the driver's per-layer gate waits for them)")
            ok = None if ok else ok; continue
        have = set(lt.get("layers_done", []))
        nb = lt.get("bytes") or b.get("bytes_approx", 0)
        est = nb * len(miss) / max(1, len(have)) / 1e6 / b.get("measured_MBps", 290)
        if not fix:
            r.add("stats", None, f"{kind}: {len(miss)} layers missing; backup has {len(have)} layers "
                  f"({gb(nb)}; missing from backup: {sorted(set(miss) - have)[:8]}...)", est_s=est)
            ok = None if ok else ok; continue
        rc = subprocess.run([sys.executable, f"{T19}/fb_restore.py", "--prefix", b["prefix"], "--root", root,
                             "--set", b["set"], "--layers", f"{lo}-{hi - 1}"], cwd=T19).returncode
        miss = _stats_missing(root, layers)
        res = True if rc == 0 and not miss else None if rc == 0 and not set(miss) & have else False
        r.add("stats", res, f"{kind}: fb_restore rc {rc}, still missing {miss}"
              + (" (not in the backup yet: the driver's per-layer gate waits)" if res is None else ""))
        ok = res if ok is True else (ok if res is not False else False)
    return ok


def step_fixed(c, fix, r):
    """fixed_set.json feeds only the manifests' default_allocation (not config_id); the driver refreshes manifests when
    it changes. It is a T19 global file, so the stats step's fb_restore brings it back."""
    p = c["frozen"]["fixed_set"]
    if not os.path.exists(p):
        return r.add("fixed", None, f"{p} missing (restored with the text stats' globals; layers still encode, manifests "
                     f"get default_allocation 'pending' until it appears)")
    return r.add("fixed", True, f"{p} sha16 {sha256(p)[:16]}")


def step_vision(c, fix, r):
    v = c["vision_graft"]
    bad = [n for n, i in v["files"].items() if not os.path.exists(f"{v['local_dir']}/{n}") or
           ("bytes" in i and os.path.getsize(f"{v['local_dir']}/{n}") != i["bytes"])]
    bad += [n for n, i in v["files"].items() if n not in bad and sha256(f"{v['local_dir']}/{n}") != i["sha256"]]
    if bad and fix:
        from huggingface_hub import hf_hub_download
        for n in bad:
            hf_hub_download(v["repo"], n, revision=v["revision"], local_dir=v["local_dir"])
        bad = [n for n, i in v["files"].items() if sha256(f"{v['local_dir']}/{n}") != i["sha256"]]
        return r.add("vision", not bad, f"graft files @ {v['revision'][:8]} " + (f"sha mismatch {bad}" if bad else "restored + verified"))
    if bad:
        return r.add("vision", None, f"graft files to fetch @ {v['revision'][:8]}: {bad} (~0.95 GB)", est_s=60)
    return r.add("vision", True, f"{len(v['files'])} graft files sha256 verified @ {v['revision'][:8]}")


def driver_cmd(c, accept_code=False):
    py = c["env"]["python"]
    cmd = [py, f"{HERE}/nq25_campaign.py", "run", "--out", c["root"]]
    if not os.path.exists(f"{c['root']}/campaign.json"):          # new ROOT (wiped): the full config from campaign.json
        f = c["frozen"]
        for k in ("stats_root", "stats_version", "layers", "experts", "encoder", "encoder_cmd", "enc_args", "fixed_set",
                  "source", "vision_root", "vision_version", "vision_weight", "vision_args", "spot_h_fn"):
            if f.get(k) not in (None, ""):
                cmd += [f"--{k.replace('_', '-')}", str(f[k])]
        cmd += ["--accept-code"]                                  # step 2 verified the hashes == the pinned ones
    for k, v in c["sched"].items():
        cmd += [f"--{k.replace('_', '-')}", str(v).lower() if isinstance(v, bool) else str(v)]
    cmd += ["--repo", c["repo"]]
    if accept_code and "--accept-code" not in cmd:
        cmd.append("--accept-code")
    return cmd


def driver_running(root):
    try:
        import psutil
        d = json.load(open(f"{root}/driver.pid"))
        return abs(psutil.Process(d["pid"]).create_time() - d["ctime"]) < 1 and d["pid"]
    except Exception:
        return False


def step_driver(c, fix, r, accept_code=False, ready=True):
    pid = driver_running(c["root"])
    if pid:
        return r.add("driver", True, f"already running (pid {pid}); status: nq25_campaign.py status --out {c['root']}")
    cmd = driver_cmd(c, accept_code)
    if not fix or not ready:
        return r.add("driver", None, ("would run: " if ready else "blocked by the steps above; would run: ") + " ".join(cmd))
    os.makedirs(c["root"], exist_ok=True)
    if os.path.exists(f"{c['root']}/STOPPED"):
        if os.environ.get("NQ25_WATCHDOG"):
            return r.add("driver", None, "STOPPED (nq25_campaign.py stop); watchdog does not restart it")
        os.remove(f"{c['root']}/STOPPED")                          # a manual resume.sh clears a manual stop
    env = dict(os.environ, LD_LIBRARY_PATH=c["env"]["LD_LIBRARY_PATH"] + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
    log = open(f"{c['root']}/driver.out", "a")
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True, cwd=HERE)
    time.sleep(15)
    return r.add("driver", p.poll() is None, f"started pid {p.pid} (log {c['root']}/campaign.log)"
                 if p.poll() is None else f"driver exited rc {p.returncode}: see {c['root']}/driver.out")


def cmd_pin(c):
    sys.path.insert(0, HERE)
    import nq25_campaign as K
    ci = K.code_info()
    head = subprocess.run(["git", "-C", NQ, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    c["code"].update(files=ci["files"], code_id=ci["code_id"], t23_id=ci["t23_id"], t25_id=ci["t25_id"],
                     git_head=head, dirty=ci["dirty"], pinned=time.strftime("%Y-%m-%d %H:%M:%S"))
    c["config_id"] = K.config_id({k: v for k, v in c["frozen"].items() if k in K.FROZEN})
    json.dump(c, open(CJ + ".tmp", "w"), indent=1); os.replace(CJ + ".tmp", CJ)
    print(f"pinned code_id {ci['code_id']} t23_id {ci['t23_id']} t25_id {ci['t25_id']} head {head} dirty {ci['dirty']} "
          f"config_id {c['config_id']} -> {CJ} (commit it)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="resume", choices=("resume", "pin"))
    ap.add_argument("--check", action="store_true", help="report only, change nothing")
    ap.add_argument("--accept-code", action="store_true", help="launch although code hashes differ from the pin")
    ap.add_argument("--skip", default="", help="comma list of steps to skip (e.g. fp8 while it is known good)")
    ap.add_argument("--campaign", default=CJ)
    a = ap.parse_args()
    os.environ["TZ"] = "Australia/Melbourne"; time.tzset()
    c = json.load(open(a.campaign))
    c["frozen"] = {k: v for k, v in c["frozen"].items() if k != "encoder_fallback"}
    if a.cmd == "pin":
        return cmd_pin(c)
    fix = not a.check
    print(f"nq25_resume {'CHECK' if a.check else 'RESUME'} {time.strftime('%a %d %b %H:%M %Z')} campaign "
          f"{c['status']} root {c['root']}", flush=True)
    if c["status"] != "launched" and fix:
        print("  campaign.json status is not 'launched' (launch gates not passed): resume only prepares inputs, "
              "the driver step is skipped", flush=True)
    r = R()
    skip = set(filter(None, a.skip.split(",")))
    steps = [("env", lambda: step_env(c, fix, r)), ("code", lambda: step_code(c, fix, r, a.accept_code)),
             ("token", lambda: step_token(c, fix, r)), ("fp8", lambda: step_fp8(c, fix, r)),
             ("stats", lambda: step_stats(c, fix, r)), ("fixed", lambda: step_fixed(c, fix, r)),
             ("vision", lambda: step_vision(c, fix, r))]
    ok = True
    for n, f in steps:
        if n in skip:
            r.add(n, None, "skipped"); continue
        try:
            res = f()
        except Exception as e:
            res = r.add(n, False, f"{type(e).__name__}: {str(e)[:300]}")
        ok = ok and res is not False          # TODO (None) does not block: the driver gates per layer
    step_driver(c, fix and c["status"] == "launched", r, a.accept_code, ready=ok or a.check)
    est = sum(x["est_s"] for x in r.rows)
    print(f"summary: {sum(x['ok'] is True for x in r.rows)}/{len(r.rows)} ok"
          + (f"; estimated restore time ~{est / 3600:.1f} h (downloads run sequentially)" if est else ""), flush=True)
    sys.exit(0 if all(x["ok"] is True for x in r.rows) else 2 if a.check else 1)


if __name__ == "__main__":
    main()
