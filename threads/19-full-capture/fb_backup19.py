"""Thread 19: flashblade S3 backup of the capture outputs, per final layer (restart insurance; /tmp is wiped on
container restart).

    fb_backup19.py --root /tmp/nestquant/19-capture-glmfmt --prefix s3://annalise-shared-prod/jarrel/nestquant/19-capture-glmfmt [--loop]
    fb_backup19.py --root /tmp/nestquant/19-capture --prefix .../19-capture --small-only     (no A0/A2/D0/D2/Dc grams)

A layer L is uploaded once it is final: ROOT/stats0/L{L} exists (the immutable chunk-0 version) and ROOT/eval/VAL_READY
exists (val rows merged + boundary-flagged).  Per layer: the stats version dir, the boundary rows of its shards,
eval/val/layer_L.pt and eval/matched/layer_L.pt.  Every object carries x-amz-meta sha256 + size, is HEAD-verified
after upload, then PREFIX/done/L{L}.json (file list with sha256/size/key) and PREFIX/latest.json are written.
Global small files (plan.json, corpus shas, shard protocol/progress, flags, markers) go to PREFIX/global/.
Raw x activations and stage-1 hidden-state checkpoints are not uploaded (recomputable from corpus + model).
--set stats1 (chunk 0 + traces snapshot) / full (every planned shard): markers done_<set>/, latest_<set>.json.
--bnd-max-shard N skips boundary rows of later shards; --budget-tb (default 4.8, scope s3://.../jarrel/) aborts a
pass whose listed total + still-to-upload bytes would reach the budget.  Restore: fb_restore.py.
"""
import argparse
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time

AWS = os.path.expanduser("~/.local/bin/aws")
ENDPOINT = "https://fb.harrisonai.io"
ENV = dict(os.environ, AWS_PROFILE="flashblade", AWS_REQUEST_CHECKSUM_CALCULATION="when_required",
           AWS_RESPONSE_CHECKSUM_VALIDATION="when_required")
GRAMS = {"A0.f32", "A2.f32", "D0.f32", "D2.f32", "Dc.f32"}


def aws(*args, capture=True):
    r = subprocess.run([AWS, "--endpoint-url", ENDPOINT, *args], env=ENV, capture_output=capture, text=True)
    if r.returncode:
        raise RuntimeError(f"aws {' '.join(args[:3])}: {r.stderr[-500:]}")
    return r.stdout


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(64 << 20)
            if not b:
                return h.hexdigest()
            h.update(b)


def split(uri):
    b, k = uri[5:].split("/", 1)
    return b, k


def head(uri):
    b, k = split(uri)
    try:
        return json.loads(aws("s3api", "head-object", "--bucket", b, "--key", k))
    except RuntimeError as e:
        if "404" in str(e) or "Not Found" in str(e):
            return None
        raise


def put(path, uri, sha=None):
    """Upload with sha256/size metadata; skip if an identical verified object exists; HEAD-verify."""
    size = os.path.getsize(path)
    sha = sha or sha256(path)
    h = head(uri)
    if not (h and h["ContentLength"] == size and h.get("Metadata", {}).get("sha256") == sha):
        aws("s3", "cp", path, uri, "--only-show-errors", "--metadata", f"sha256={sha},size={size}")
        h = head(uri)
    if not h or h["ContentLength"] != size or h.get("Metadata", {}).get("sha256") != sha:
        raise RuntimeError(f"verify failed for {uri}")
    return dict(size=size, sha256=sha, key=uri)


def put_snapshot(path, uri, tmpdir):
    """Upload a copy (global files such as progress.json may be rewritten while uploading)."""
    p = f"{tmpdir}/.fb_snap_{os.getpid()}_{os.path.basename(path)}"
    shutil.copyfile(path, p)
    try:
        return put(p, uri)
    finally:
        os.remove(p)


def put_json(obj, uri, tmpdir):
    p = f"{tmpdir}/.fb_{os.getpid()}_{os.path.basename(uri)}"
    with open(p, "w") as f:
        json.dump(obj, f, indent=1)
    r = put(p, uri)
    os.remove(p)
    return r


def final_ids(root):
    """Shard set of the 'full' stats (chunk 0 + every other planned group, e.g. traces)."""
    return {int(k) for k in json.load(open(f"{root}/plan.json"))["shards"]}


def layer_files(root, L, small_only, which="stats0", bnd_max_shard=None):
    """[(local path, relative key)] of final layer L, or None if not final yet.
    which = stats0 (frozen chunk-0 version) or full (current stats once every planned shard is merged)."""
    s0 = f"{root}/stats/L{L}" if which == "full" else f"{root}/{which}/L{L}"
    if not (os.path.exists(s0) and os.path.exists(f"{root}/eval/VAL_READY")):
        return None
    vd = os.path.realpath(s0)
    if which == "full" and {s_["shard"] for s_ in json.load(open(f"{vd}/meta.json"))["shards"]} != final_ids(root):
        return None
    vname = os.path.basename(vd)
    out = [(f"{vd}/{f}", f"stats/{vname}/{f}") for f in sorted(os.listdir(vd))
           if not (small_only and f in GRAMS) and not f.endswith(".tmp")]
    m = json.load(open(f"{vd}/meta.json"))
    for sh in m["shards"]:
        if bnd_max_shard is not None and sh["shard"] > bnd_max_shard:
            continue                               # boundary rows not needed by the encode (weight 1); recomputable
        d = sh.get("bnd_rows") or f"{root}/bnd_rows/s{sh['shard']:02d}/L{L}"
        if not os.path.exists(f"{d}/rows.npz"):
            return None
        out += [(f"{d}/{f}", f"bnd_rows/s{sh['shard']:02d}/L{L}/{f}") for f in sorted(os.listdir(d))]
    for kind in ("val", "matched"):
        p = f"{root}/eval/{kind}/layer_{L}.pt"
        if os.path.exists(p):
            out.append((os.path.realpath(p), f"eval/{kind}/layer_{L}.pt"))
    return out, vname, m


def global_files(root):
    pats = ["plan.json", "corpus_sha256.txt", "SHARD0_READY", "HOLD_SHARDS_ABOVE", "eval/VAL_READY",
            "bnd/*.npz", "shards/s*/protocol.json", "shards/s*/state/progress.json", "MANIFEST.json", "fixed_set_text.json", "latest_final.json", "FINAL_STATS_READY", "BACKUP_FINAL_DONE",
            "fixed_set.json", "eval/VAL_TRACES_READY", "eval/val_traces/layer_*.pt"]
    out = []
    for pat in pats:
        for p in sorted(glob.glob(f"{root}/{pat}")):
            out.append((p, "global/" + os.path.relpath(p, root)))
    return out


def marker_dir(which):
    return "done" if which == "stats0" else f"done_{which}"


def latest_name(which):
    return "latest" if which == "stats0" else f"latest_{which}"


def backup_layer(root, prefix, L, small_only, tmpdir, which="stats0", bnd_max_shard=None, gate=None):
    lf = layer_files(root, L, small_only, which, bnd_max_shard)
    if lf is None:
        return None
    files, vname, m = lf
    if gate is not None:
        gate(L, files)                             # per-layer budget gate (raises BudgetExceeded)
    t0 = time.time()
    rec = []
    for path, key in files:
        r = put(path, f"{prefix}/{key}")
        rec.append(dict(r, rel=key))
    done = dict(layer=L, version=vname, set=which, shards=[s["shard"] for s in m["shards"]], small_only=small_only,
                bnd_rows_shards=[s["shard"] for s in m["shards"] if bnd_max_shard is None or s["shard"] <= bnd_max_shard],
                files=rec, bytes=sum(r["size"] for r in rec), uploaded_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                seconds=round(time.time() - t0, 1))
    put_json(done, f"{prefix}/{marker_dir(which)}/L{L}.json", tmpdir)
    return done


def s3_listing(scope):
    """{uri: size} of every object under scope (s3://bucket/prefix/)."""
    b, k = split(scope)
    out = {}
    for line in aws("s3", "ls", scope, "--recursive").splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4:
            out[f"s3://{b}/{parts[3]}"] = int(parts[2])
    return out


class BudgetExceeded(RuntimeError):
    pass


def make_layer_gate(a, prefix):
    """Per-layer gate: before each layer upload, list the budget scope and require
    total + (this layer's bytes not yet on S3) + (other in-flight layers' reservations) < budget.
    The pass-level budget_check only sees layers that are final at pass start; this also covers layers that
    become final mid-pass."""
    lock = threading.Lock()
    reserved = {}

    def gate(L, files):
        with lock:
            lst = s3_listing(a.budget_scope)
            total = sum(lst.values())
            need = sum(os.path.getsize(p) for p, k in files if lst.get(f"{prefix}/{k}") != os.path.getsize(p))
            other = sum(v for k, v in reserved.items() if k != L)
            rec = dict(gate_layer=L, total_tb=round(total / 1e12, 4), need_gb=round(need / 1e9, 2),
                       inflight_gb=round(other / 1e9, 2), budget_tb=a.budget_tb,
                       utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            print(json.dumps(rec), flush=True)
            if total + need + other >= a.budget_tb * 1e12:
                raise BudgetExceeded(f"BUDGET: {rec} -- not uploading L{L}")
            reserved[L] = need

    def release(L):
        with lock:
            reserved.pop(L, None)

    gate.release = release
    return gate


def budget_check(a, prefix, todo):
    """Abort unless (bytes under the budget scope) + (bytes this pass would still upload) < budget."""
    lst = s3_listing(a.budget_scope)
    total = sum(lst.values())
    plan = 0
    for L in todo:
        lf = layer_files(a.root, L, a.small_only, a.set, a.bnd_max_shard)
        if lf:
            plan += sum(os.path.getsize(p) for p, k in lf[0] if lst.get(f"{prefix}/{k}") != os.path.getsize(p))
    rec = dict(budget_scope=a.budget_scope, total_tb=round(total / 1e12, 4), planned_tb=round(plan / 1e12, 4),
               budget_tb=a.budget_tb, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print(json.dumps(rec), flush=True)
    if total + plan >= a.budget_tb * 1e12:
        raise SystemExit(f"BUDGET: {rec} -- not uploading")


def final_verify(a, prefix, tmpdir):
    """All 75 done_full markers: every listed object present on S3 with its size -> latest_final.json (S3 + local)
    and ROOT/BACKUP_FINAL_DONE."""
    lst = s3_listing(prefix + "/")
    n = nb = 0
    layers = {}
    for L in range(3, 78):
        p = f"{tmpdir}/.fb_verify_L{L}.json"
        aws("s3", "cp", f"{prefix}/{marker_dir('full')}/L{L}.json", p, "--only-show-errors")
        d = json.load(open(p)); os.remove(p)
        bad = [r["key"] for r in d["files"] if lst.get(r["key"]) != r["size"]]
        if bad or d["small_only"] or not {os.path.basename(r["rel"]) for r in d["files"]} >= GRAMS:
            raise SystemExit(f"final verify failed at L{L}: {len(bad)} bad, small_only={d['small_only']}")
        n += len(d["files"]); nb += d["bytes"]
        layers[L] = dict(version=d["version"], objects=len(d["files"]), bytes=d["bytes"])
    doc = dict(root=a.root, prefix=prefix, set="full", verified_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               layers=layers, objects=n, bytes=nb, bnd_rows_max_shard=a.bnd_max_shard,
               restore=f"fb_restore.py --prefix {prefix} --root <root> --set full")
    with open(f"{a.root}/latest_final.json.tmp", "w") as f:
        json.dump(doc, f, indent=1)
    os.replace(f"{a.root}/latest_final.json.tmp", f"{a.root}/latest_final.json")
    put(f"{a.root}/latest_final.json", f"{prefix}/latest_final.json")
    open(f"{a.root}/BACKUP_FINAL_DONE", "w").write(json.dumps(dict(objects=n, bytes=nb, utc=doc["verified_utc"])))
    put(f"{a.root}/BACKUP_FINAL_DONE", f"{prefix}/BACKUP_FINAL_DONE")
    print(json.dumps(dict(final_verified=True, objects=n, bytes=nb)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--small-only", action="store_true", help="skip the per-expert gram files (A0/A2/D0/D2/Dc)")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--workers", type=int, default=2, help="layers uploaded concurrently")
    ap.add_argument("--set", default="stats0",
                    help="stats0 / stats1 / ...: a frozen snapshot (plan.json); full: stats after every planned shard")
    ap.add_argument("--bnd-max-shard", type=int, help="upload boundary rows only of shards <= N")
    ap.add_argument("--budget-tb", type=float, default=4.8, help="abort if scope total + planned upload >= this")
    ap.add_argument("--budget-scope", default="s3://annalise-shared-prod/jarrel/")
    ap.add_argument("--state-tag", default="", help="suffix for the local state file (separate passes of one set)")
    a = ap.parse_args()
    prefix = a.prefix.rstrip("/")
    tmpdir = f"{a.root}/logs"
    state_p = f"{a.root}/logs/fb_backup_state{'' if a.set == 'stats0' else '_' + a.set}{a.state_tag}.json"
    latest = f"{prefix}/{latest_name(a.set)}.json"
    state = json.load(open(state_p)) if os.path.exists(state_p) else {}
    gate = make_layer_gate(a, prefix)

    def job(L):
        try:
            return backup_layer(a.root, prefix, L, a.small_only, tmpdir, a.set, a.bnd_max_shard, gate=gate)
        finally:
            gate.release(L)

    while True:
        todo = [L for L in range(3, 78) if str(L) not in state]
        budget_check(a, prefix, todo)
        with cf.ThreadPoolExecutor(a.workers) as ex:
            futs = {ex.submit(job, L): L for L in todo}
            for f in cf.as_completed(futs):
                L = futs[f]
                try:
                    d = f.result()
                except BudgetExceeded as e:
                    for g_ in futs:
                        g_.cancel()
                    raise SystemExit(str(e))
                if d is None:
                    continue
                state[str(L)] = dict(version=d["version"], bytes=d["bytes"], seconds=d["seconds"])
                with open(state_p + ".tmp", "w") as fh:
                    json.dump(state, fh)
                os.replace(state_p + ".tmp", state_p)
                print(json.dumps(dict(layer=L, bytes=d["bytes"], seconds=d["seconds"])), flush=True)
        g = [dict(put_snapshot(p, f"{prefix}/{k}", tmpdir), rel=k) for p, k in global_files(a.root)]
        put_json(dict(root=a.root, prefix=prefix, set=a.set, small_only=a.small_only, layers_done=sorted(int(k) for k in state),
                      bytes=sum(v["bytes"] for v in state.values()), global_files=g,
                      updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), latest, tmpdir)
        if len(state) == 75 and a.set == "full" and not a.small_only:
            final_verify(a, prefix, tmpdir)
        if not a.loop or len(state) == 75:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
