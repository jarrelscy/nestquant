"""Thread 25: adopt the global-rescore fixed_set (schema v3) once every layer has uploaded (lead/user 2026-09-29).

    nq25_adopt_v3.py [--dry] [--no-wait]

1. wait until all 75 layers are 'uploaded' (no layer finalizes mid-switch);
2. build it: nq25_rescore.py (full, no partial) -> RS/fixed_set_v3.json; gates
   (a) exactly 1950, every layer in [16, 128], fixed & floating_default disjoint
   (b) d_e estimator vs spot numbers: Spearman > 0.5
   (c) predicted error removed >= the current set's;  any failure -> RS/ADOPT_FAILED, nothing changes;
3. stop the driver (STOPPED keeps the watchdog off);
4. back up the current 19-capture-glmfmt/fixed_set.json as fixed_set_v2_perlayer.json (local + flashblade global/,
   budget scope total < 4.8 TB), then write the v3 file to 19-capture-glmfmt/fixed_set.json (+ flashblade global/); resume.sh restarts the driver;
5. the driver's remanifest refreshes + re-uploads every manifest; wait, then verify all 75 REMOTE manifests carry the
   new sha (fresh hf_hub_download);
6. upload README_hf.md as README.md.  Log: RS/adopt.log, result RS/ADOPT_DONE.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/tmp/nestquant/nq-encode-v1"
RS = "/tmp/nestquant/25-campaign/rescore"
CUR = "/tmp/nestquant/19-capture-glmfmt/fixed_set.json"
BAK = "/tmp/nestquant/19-capture-glmfmt/fixed_set_v2_perlayer.json"
S3G = "s3://annalise-shared-prod/jarrel/nestquant/19-capture-glmfmt/global"
SCOPE = "s3://annalise-shared-prod/jarrel/"
EP = "https://fb.harrisonai.io"
REPO = "jarrelscy/GLM-5.3-NestQuant-2-4bit"
PY = "/home/coder/git/glm52/.venv/bin/python"
LAYERS = range(3, 78)


def log(*m):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    s = datetime.now(ZoneInfo("Australia/Melbourne")).strftime("%a %d %b %H:%M:%S %Z") + "  " + " ".join(str(x) for x in m)
    print(s, flush=True)
    open(f"{RS}/adopt.log", "a").write(s + "\n")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def aws(*args, timeout=900):
    env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"], AWS_PROFILE="flashblade",
               AWS_REQUEST_CHECKSUM_CALCULATION="when_required", AWS_RESPONSE_CHECKSUM_VALIDATION="when_required")
    r = subprocess.run(["aws", "--endpoint-url", EP, *args], env=env, capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError(f"aws {args[:2]}: {r.stderr[-400:]}")
    return r.stdout


def states():
    s = json.load(open(f"{ROOT}/state.json"))
    return {int(L): v["state"] for L, v in s["layers"].items()}


def local_shas():
    out = {}
    for L in LAYERS:
        try:
            out[L] = (json.load(open(f"{ROOT}/L{L}/manifest.json")).get("default_allocation") or {}).get("sha256")
        except Exception:
            out[L] = None
    return out


def driver_pid():
    try:
        return json.load(open(f"{ROOT}/driver.pid"))["pid"]
    except Exception:
        return None


def alive(pid):
    try:
        os.kill(pid, 0); return True
    except Exception:
        return False


def remote_verify(new):
    from huggingface_hub import hf_hub_download
    import tempfile
    bad = {}
    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        for L in LAYERS:
            for k in range(5):
                try:
                    p = hf_hub_download(REPO, f"layers/L{L}/manifest.json", local_dir=td, force_download=True)
                    break
                except Exception as e:
                    time.sleep(10 * (k + 1)); err = e
            else:
                bad[L] = f"download failed: {err}"; continue
            da = json.load(open(p)).get("default_allocation") or {}
            if da.get("sha256") != new:
                bad[L] = str(da.get("sha256"))[:16]
            elif da.get("n") != len(da.get("level4_experts") or []):
                bad[L] = f"n {da.get('n')} != {len(da.get('level4_experts') or [])}"
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="build + gate only")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--partial", action="store_true", help="test only (implies --dry --no-wait): rescore --allow-partial")
    a = ap.parse_args()
    if a.partial:
        a.dry = a.no_wait = True
    os.makedirs(RS, exist_ok=True)
    # 1. wait for all uploads
    while not a.no_wait:
        st = states()
        if len(st) == 75 and all(v == "uploaded" for v in st.values()):
            break
        time.sleep(120)
    log("all 75 layers uploaded" if not a.no_wait else "no-wait")
    # 2. build + gates
    out = f"{RS}/fixed_set_v3.json" if not a.partial else f"{RS}/partial_v3.json"
    r = subprocess.run([PY, f"{HERE}/nq25_rescore.py", "--out", out] + (["--allow-partial"] if a.partial else []),
                       capture_output=True, text=True)
    if r.returncode:
        log("rescore FAILED", r.stderr[-800:]); open(f"{RS}/ADOPT_FAILED", "w").write(r.stderr[-2000:]); sys.exit(1)
    rep = json.load(open(out.replace(".json", ".report.json")))
    doc = json.load(open(out))
    n = {int(L): len(v) for L, v in doc["fixed_set"].items()}
    ga = (rep["n_total"] == 1950 and len(n) == 75 and all(16 <= v <= 128 for v in n.values()) and rep["disjoint"]
          and not rep["missing_layers"])
    gb = rep.get("xcheck", {}).get("spearman_d", -1) > 0.5
    gc = rep["err_removed"]["new"] >= rep["err_removed"]["cur"]
    gates = dict(a=ga, b=gb, c=gc, n_total=rep["n_total"], min=rep["min"], max=rep["max"], disjoint=rep["disjoint"],
                 spearman=rep.get("xcheck", {}).get("spearman_d"), xn=rep.get("xcheck", {}).get("n"),
                 err_ratio=rep["err_removed"]["ratio"])
    log("gates", json.dumps(gates))
    if a.partial:
        log("partial test (missing layers allowed by the test only):", rep["missing_layers"]); return
    if not (ga and gb and gc):
        open(f"{RS}/ADOPT_FAILED", "w").write(json.dumps(gates)); log("gate FAILED: nothing adopted"); sys.exit(1)
    if a.dry:
        log("dry: stop here"); return
    new = sha(out)
    # 3. stop the driver (it has usually exited already: COMPLETE); STOPPED keeps the watchdog off during the switch
    pid = driver_pid()
    subprocess.run([PY, f"{HERE}/nq25_campaign.py", "stop", "--out", ROOT], capture_output=True, text=True)
    for _ in range(120):
        if not pid or not alive(pid):
            break
        time.sleep(2)
    else:
        open(f"{RS}/ADOPT_FAILED", "w").write("driver did not stop"); log("driver did not stop"); sys.exit(1)
    log("driver stopped (pid", pid, ")")
    # 4. backup + write (driver down)
    if not os.path.exists(BAK):
        shutil.copy2(CUR, BAK)
    assert sha(BAK) == sha(CUR) or json.load(open(BAK)).get("schema", "").endswith("-v2"), "backup is not the v2 file"
    lst = aws("s3", "ls", SCOPE, "--recursive", "--summarize", timeout=3600)
    tot = int([l for l in lst.splitlines() if "Total Size" in l][0].split(":")[1])
    need = os.path.getsize(BAK) + os.path.getsize(out)
    log(f"flashblade scope total {tot / 1e12:.4f} TB + {need / 1e6:.2f} MB (budget 4.8 TB)")
    if tot + need >= 4.8e12:
        open(f"{RS}/ADOPT_FAILED", "w").write("flashblade budget"); log("flashblade budget FAILED"); sys.exit(1)
    aws("s3", "cp", "--only-show-errors", BAK, f"{S3G}/fixed_set_v2_perlayer.json")
    tmp = CUR + ".v3tmp"
    shutil.copy2(out, tmp); os.replace(tmp, CUR)
    assert sha(CUR) == new
    aws("s3", "cp", "--only-show-errors", CUR, f"{S3G}/fixed_set.json")
    s3 = aws("s3", "ls", f"{S3G}/")
    log("written", CUR, new[:16], "| flashblade:", [l.split()[-2:] for l in s3.splitlines() if "fixed_set" in l])
    # 4b. start the driver on the committed v3-aware code: resume.sh clears STOPPED; the driver removes COMPLETE, its
    #     refresh_manifests rewrites every manifest's default_allocation and re-queues the (manifest-only) uploads
    r = subprocess.run([f"{HERE}/resume.sh"], capture_output=True, text=True, cwd=HERE)
    p2 = driver_pid()
    log("driver restart: resume rc", r.returncode, "driver pid", p2, r.stdout.strip().splitlines()[-1:] if r.stdout else "")
    # 5. wait for remanifest + re-upload, then verify remote
    t0 = time.time()
    while True:
        st, ls = states(), local_shas()
        if all(ls[L] == new for L in LAYERS) and all(v == "uploaded" for v in st.values()):
            break
        if time.time() - t0 > 4 * 3600:
            open(f"{RS}/ADOPT_FAILED", "w").write("re-upload timeout"); log("re-upload timeout"); sys.exit(1)
        time.sleep(60)
    log("all 75 local manifests refreshed + re-uploaded in", round((time.time() - t0) / 60, 1), "min")
    bad = remote_verify(new)
    log("remote verify:", "all 75 OK" if not bad else f"BAD {bad}")
    # 6. README
    from huggingface_hub import HfApi
    HfApi().upload_file(path_or_fileobj=f"{HERE}/README_hf.md", path_in_repo="README.md", repo_id=REPO,
                        commit_message="README: global default 4-bit set (1950 experts, 16-128 per layer)")
    log("README.md uploaded")
    open(f"{RS}/ADOPT_DONE" if not bad else f"{RS}/ADOPT_FAILED", "w").write(json.dumps(dict(sha=new, gates=gates, remote_bad=bad)))


if __name__ == "__main__":
    main()
