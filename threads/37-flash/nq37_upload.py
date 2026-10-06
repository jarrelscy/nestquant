"""T37 upload (GLM-5.3-Flash NestQuant 1.5/4): T25 nq25_upload adapted for Flash (42 MoE layers L3-44, 288 experts).
DRY RUN unless --go.  Nothing here creates a repo unless --create-repo is given.

  python nq37_upload.py layers   --layers 3,4 [--go]        # one create_commit per layer, then re-list + sha verify
  python nq37_upload.py top      [--top-dir REL] [--go]     # every file under REL (recursive, rel paths kept) -> root
  python nq37_upload.py complete [--go]                     # refuses unless 42 layers + top + serving/predictor verified
  python nq37_upload.py status                              # remote_layers() (HF truth) as json
  python nq37_upload.py watch    [--go] [--sleep 120] [--once] [--create-repo public|private]

Layout (same as the GLM-5.3 2-4bit layers/ tree):
  layers/L{L}/tp{s}.safetensors  s = 0..7 (nq25_st convert output; the .pt stay local)
  layers/L{L}/manifest.json      nq_layer manifest (files[] = safetensors sha256/bytes)
  <top-level files>              from /tmp/nestquant/37-flash/release (config/tokenizer/non-expert/vision/README/
                                 serving/predictor/...); COMPLETE (written LAST, by `complete`)
Diffs vs nq25_upload: REPO; per-layer records go to ROOT/upload/L{L}.json (not ROOT/uploads); a pre-upload guard
(manifest n_experts 288, layer in 3..44, tp0..7 safetensors present with manifest sizes); top mode is recursive over
the release dir; plus `complete` and `watch`.
watch: every --sleep s, each layer with ROOT/fin/L{L}.json rc 0 whose ROOT/upload/L{L}.json is not status "uploaded"
for the CURRENT local manifest.json sha256 is uploaded + verified (lowest layer first).  Any exception (network, 5xx,
403 quota, ...) is logged, the layer backs off (10 min doubling to 2 h) and the loop goes on; it never exits on
errors.  A 403 also writes ROOT/upload/ALERT_403 (HF storage quota: see memory b175-release-hf-quota).  It exits only
when all 42 layers are uploaded (unless --forever) or with --once.  One watcher at a time (flock ROOT/upload/watch.lock).
The token comes from the huggingface_hub default lookup (HF_TOKEN env or ~/.cache/huggingface/token); never printed.
"""
import os, sys, json, time, hashlib, fcntl, traceback

REPO = "jarrelscy/GLM-5.3-Flash-NestQuant-1.5-4bit"
ROOT = "/tmp/nestquant/37-flash/enc_b15"
REL = "/tmp/nestquant/37-flash/release"
LAYERS = list(range(3, 45))
NEXP, NSH = 288, 8


def mel():
    """Melbourne wall time (AEDT from 2026-10-04)."""
    try:
        from zoneinfo import ZoneInfo
        import datetime
        return datetime.datetime.now(ZoneInfo("Australia/Melbourne")).strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime())


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def git_blob_sha1(p):
    data = open(p, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def is_404(e):
    return "404" in str(e) or "not found" in str(e).lower() or "NotFound" in type(e).__name__


def remote_tree(api, repo, prefix=None):
    """{path: dict(size, lfs_sha256, oid)} under prefix (whole repo if None; empty if repo/path absent)."""
    out = {}
    try:
        for it in api.list_repo_tree(repo, path_in_repo=prefix, recursive=True, expand=True):
            if getattr(it, "size", None) is None:          # folder
                continue
            lfs = getattr(it, "lfs", None)
            out[it.path] = dict(size=it.size, lfs_sha256=(lfs.sha256 if lfs else None), oid=getattr(it, "blob_id", None))
    except Exception as e:                                 # EntryNotFound / RepositoryNotFound for a fresh path
        if not is_404(e):
            raise
    return out


def same_remote(r, lp, sh, nb):
    return r is not None and r["size"] == nb and (
        (r["lfs_sha256"] == sh) if r["lfs_sha256"] else (os.path.exists(lp) and r["oid"] == git_blob_sha1(lp)))


def with_retry(fn, tries=6, what=""):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1 or "403" in str(e):          # quota 403s do not heal by retrying for minutes
                raise
            w = min(300, 10 * 2 ** i)
            print(f"  retry {what} in {w}s after {type(e).__name__}: {str(e)[:200]}", flush=True)
            time.sleep(w)


def remote_layers(repo=REPO, cache=None):
    """HF is the source of truth: {L: dict(verified, ...)} for every layers/L{L}/ on the Hub (nq25_upload.remote_layers
    + n_experts must be 288 and all 8 tp files listed)."""
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    tree = with_retry(lambda: remote_tree(api, repo, "layers"), what="list layers")
    out = {}
    Ls = sorted({int(p.split("/")[1][1:]) for p in tree if p.count("/") >= 2 and p.split("/")[1][1:].isdigit()})
    for L in Ls:
        mp = f"layers/L{L}/manifest.json"
        if mp not in tree:
            out[L] = dict(verified=False, reason="no manifest.json"); continue
        try:
            f = with_retry(lambda: hf_hub_download(repo, mp, local_dir=cache), what=f"manifest L{L}")
            man = json.load(open(f))
        except Exception as e:
            out[L] = dict(verified=False, reason=f"manifest unreadable: {type(e).__name__}"); continue
        bad = []
        for fn, info in man.get("files", {}).items():
            r = tree.get(f"layers/L{L}/{fn}")
            if r is None or r["size"] != info["bytes"] or r["lfs_sha256"] != info["sha256"]:
                bad.append(fn)
        shapes_ok = man.get("n_experts") == NEXP and sorted(man.get("files", {})) == [f"tp{s}.safetensors" for s in range(NSH)]
        c = man.get("campaign") or {}
        rm = tree[mp]
        out[L] = dict(verified=not bad and shapes_ok, bad=bad, config_id=c.get("config_id"),
                      code_id=(c.get("code") or {}).get("code_id"), encoder=c.get("encoder"),
                      n_experts=man.get("n_experts"), bytes=sum(v["bytes"] for v in man.get("files", {}).values()),
                      manifest_oid=rm.get("oid"), manifest_lfs=rm.get("lfs_sha256"), manifest_sha256=sha256(f),
                      reason="ok" if not bad and shapes_ok else (f"{len(bad)} files missing/mismatched" if bad else
                                                                 f"n_experts {man.get('n_experts')} / files {sorted(man.get('files', {}))}"))
    return out


def layer_files(root, L):
    d = f"{root}/L{L}"
    man = json.load(open(f"{d}/manifest.json"))
    # guard: GLM-5.3 hardcodes (256 experts, L3-77) must never reach this repo
    if L not in LAYERS:
        raise SystemExit(f"L{L}: not a Flash MoE layer (3..44)")
    if man.get("n_experts") != NEXP or man.get("layer", L) != L:
        raise SystemExit(f"L{L}: manifest n_experts {man.get('n_experts')} layer {man.get('layer')} (want {NEXP}, {L})")
    if sorted(man["files"]) != [f"tp{s}.safetensors" for s in range(NSH)]:
        raise SystemExit(f"L{L}: manifest files {sorted(man['files'])}")
    files = [(f"{d}/{f}", f"layers/L{L}/{f}", info["sha256"], info["bytes"]) for f, info in sorted(man["files"].items())]
    for lp, _, _, nb in files:
        if os.path.exists(lp) and os.path.getsize(lp) != nb:
            raise SystemExit(f"L{L}: {lp} is {os.path.getsize(lp)} B, manifest says {nb}")
    mp = f"{d}/manifest.json"
    files.append((mp, f"layers/L{L}/manifest.json", sha256(mp), os.path.getsize(mp)))
    return files


def run(root, layers, go=False, repo=REPO):
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi()
    res = {}
    tot_new = 0
    rd = f"{root}/upload"
    os.makedirs(rd, exist_ok=True)
    for L in layers:
        files = layer_files(root, L)
        rt = with_retry(lambda: remote_tree(api, repo, f"layers/L{L}"), what=f"list L{L}")
        plan = []
        for lp, rp, sh, nb in files:
            r = rt.get(rp)
            if not os.path.exists(lp):        # remote-only (local layer dir pruned/wiped, manifest refreshed): must match
                if not (r is not None and r["size"] == nb and r["lfs_sha256"] == sh):
                    raise SystemExit(f"L{L}: {rp} missing locally and not on the Hub with sha {sh[:16]}")
                plan.append(dict(local=lp, path=rp, bytes=nb, sha256=sh, action="skip (same, remote-only)"))
                continue
            same = same_remote(r, lp, sh, nb)
            plan.append(dict(local=lp, path=rp, bytes=nb, sha256=sh, action="skip (same)" if same else
                             ("replace" if r is not None else "add")))
        new = [p for p in plan if not p["action"].startswith("skip")]
        nb_new = sum(p["bytes"] for p in new)
        tot_new += nb_new
        print(f"L{L}: {len(new)}/{len(plan)} files to send, {nb_new / 2**30:.2f} GiB"
              f"{'' if go else '  [DRY RUN]'}", flush=True)
        for p in plan:
            print(f"   {p['action']:<12} {p['path']:<28} {p['bytes'] / 2**20:10.1f} MiB  sha256 {p['sha256'][:16]}", flush=True)
        man_sha = files[-1][2]
        rec = dict(layer=L, repo=repo, plan=plan, bytes_to_send=nb_new, manifest_sha256=man_sha, time=mel())
        if not go:
            rec["status"] = "dry-run"
            json.dump(rec, open(f"{rd}/L{L}.plan.json", "w"), indent=1)
            res[L] = dict(status="dry-run", files=len(plan), to_send=len(new), bytes=nb_new)
            continue
        if new:
            ops = [CommitOperationAdd(path_in_repo=p["path"], path_or_fileobj=p["local"]) for p in new]
            t0 = time.time()
            ci = with_retry(lambda: api.create_commit(repo, operations=ops, commit_message=f"NestQuant layer {L}: "
                                                      f"{len(new)} files"), what=f"commit L{L}")
            dt = time.time() - t0
            rec.update(commit=ci.oid, upload_s=round(dt, 1), MBps=round(nb_new / 1e6 / max(dt, 1e-3), 1))
        rt = with_retry(lambda: remote_tree(api, repo, f"layers/L{L}"), what=f"relist L{L}")
        bad = []
        for p in plan:
            r = rt.get(p["path"])
            p["remote"] = r
            if not same_remote(r, p["local"], p["sha256"], p["bytes"]):
                bad.append(p["path"])
        rec["status"] = "uploaded" if not bad else f"verify failed {bad}"
        tmp = f"{rd}/L{L}.json.tmp"
        json.dump(rec, open(tmp, "w"), indent=1); os.replace(tmp, f"{rd}/L{L}.json")
        res[L] = dict(status=rec["status"], commit=rec.get("commit"), bytes=nb_new, MBps=rec.get("MBps"))
        print(f"L{L}: {rec['status']} commit {rec.get('commit')} {rec.get('MBps')} MB/s", flush=True)
    print(f"total to send: {tot_new / 2**30:.2f} GiB over {len(layers)} layers{'' if go else ' [DRY RUN]'}", flush=True)
    return res


def top_files(top, names=None):
    """(local, path_in_repo, sha256, bytes) for every file under `top` (recursive; skips COMPLETE, dot-dirs, *.tmp).
    sha256 from nonexpert_manifest.json where listed (nq25 convention), else computed."""
    man = {f["file"]: f for f in json.load(open(f"{top}/nonexpert_manifest.json"))["files"]} \
        if os.path.exists(f"{top}/nonexpert_manifest.json") else {}
    if not names:
        names = []
        for dp, dns, fns in os.walk(top):
            dns[:] = sorted(d for d in dns if not d.startswith("."))
            for fn in sorted(fns):
                rp = os.path.relpath(f"{dp}/{fn}", top)
                if rp != "COMPLETE" and not fn.endswith(".tmp") and not fn.startswith("."):
                    names.append(rp)
    out = []
    for n in names:
        lp = f"{top}/{n}"
        nb = os.path.getsize(lp)
        sh = man[n]["sha256"] if n in man and man[n].get("bytes") == nb else sha256(lp)
        out.append((lp, n, sh, nb))
    return out


def run_top(top, names=None, go=False, repo=REPO, rec_dir=None):
    """top-level (non-layer) files from the staging dir `top` -> repo (rel paths kept). Same compare / commit /
    re-verify as the layers; one commit."""
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi()
    files = top_files(top, names)
    bad_paths = [f[1] for f in files if f[1].startswith("layers/") or f[1] == "COMPLETE"]
    if bad_paths:
        raise SystemExit(f"top: refusing layer/COMPLETE paths {bad_paths[:5]}")
    rt = with_retry(lambda: remote_tree(api, repo), what="list root")
    new = [f for f in files if not same_remote(rt.get(f[1]), *[f[0], f[2], f[3]])]
    nb_new = sum(f[3] for f in new)
    print(f"top: {len(new)}/{len(files)} files to send, {nb_new / 2**30:.2f} GiB{'' if go else '  [DRY RUN]'}", flush=True)
    for f in files:
        print(f"   {'send' if f in new else 'skip (same)':<12} {f[1]:<48} {f[3] / 2**20:10.1f} MiB  sha256 {f[2][:16]}", flush=True)
    rec = dict(repo=repo, files=[dict(path=f[1], bytes=f[3], sha256=f[2], action="send" if f in new else "skip (same)")
                                 for f in files], time=mel())
    if go and new:
        ops = [CommitOperationAdd(path_in_repo=rp, path_or_fileobj=lp) for lp, rp, _, _ in new]
        t0 = time.time()
        ci = with_retry(lambda: api.create_commit(repo, operations=ops, commit_message=f"NestQuant top-level: {len(new)} files"),
                        what="commit top")
        dt = time.time() - t0
        rec.update(commit=ci.oid, upload_s=round(dt, 1), MBps=round(nb_new / 1e6 / max(dt, 1e-3), 1))
        rt = with_retry(lambda: remote_tree(api, repo), what="relist root")
    bad = [f[1] for f in files if not same_remote(rt.get(f[1]), f[0], f[2], f[3])] if go else []
    rec["status"] = ("uploaded" if not bad else f"verify failed {bad}") if go else "dry-run"
    if rec_dir:
        os.makedirs(rec_dir, exist_ok=True)
        json.dump(rec, open(f"{rec_dir}/{'top' if go else 'top.dry'}.json", "w"), indent=1)   # a dry plan never hides the last real upload
        if go:
            with open(f"{rec_dir}/top_history.jsonl", "a") as fh:
                fh.write(json.dumps({k: rec.get(k) for k in ("time", "commit", "status", "MBps")} | dict(files=[x["path"] for x in rec.get("files", []) if x.get("action") == "send"])) + "\n")
    print(f"top: {rec['status']} commit {rec.get('commit')} {rec.get('MBps')} MB/s", flush=True)
    return rec


def run_complete(root, top, go=False, repo=REPO, names=None):
    """COMPLETE marker, LAST.  Refuses (exit 2) unless, on the Hub: all 42 layers verified (manifest + 8 tp files, LFS
    sha256 == manifest, n_experts 288), every local top-level file (release dir) present with the same sha, README.md
    + config.json present, and >= 1 serving/predictor/* file present (and equal to the local copy if there is one).
    COMPLETE = json {format, repo, artifact key, per-layer manifest sha256 + file sha256s, top-level sha256s}."""
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi()
    errs = []
    rl = remote_layers(repo, cache=f"{root}/upload/_remote")
    miss = [L for L in LAYERS if not (rl.get(L) or {}).get("verified")]
    if miss:
        errs.append(f"layers not verified on the Hub: {miss} " + "; ".join(f"L{L}: {(rl.get(L) or {}).get('reason', 'absent')}" for L in miss[:5]))
    extra = sorted(set(rl) - set(LAYERS))
    if extra:
        errs.append(f"unexpected layers on the Hub: {extra}")
    rt = with_retry(lambda: remote_tree(api, repo), what="list root")
    tf = top_files(top, names) if os.path.isdir(top) else []
    if not tf:
        errs.append(f"no local top-level files in {top}")
    tbad = [f[1] for f in tf if not same_remote(rt.get(f[1]), f[0], f[2], f[3])]
    if tbad:
        errs.append(f"top-level files missing/mismatched on the Hub: {tbad[:10]}")
    for need in ("README.md", "config.json"):
        if need not in rt:
            errs.append(f"{need} not on the Hub")
    pred = sorted(p for p in rt if p.startswith("serving/predictor/"))
    if not pred:
        errs.append("serving/predictor/ empty on the Hub")
    if errs:
        print("COMPLETE REFUSED:\n  " + "\n  ".join(errs), flush=True)
        return 2
    layers = {}
    for L in LAYERS:
        man_p = f"layers/L{L}/manifest.json"
        layers[str(L)] = dict(manifest_sha256=rl[L]["manifest_sha256"],
                              files={p.split("/")[-1]: rt[p]["lfs_sha256"] for p in rt if p.startswith(f"layers/L{L}/") and rt[p]["lfs_sha256"]},
                              bytes=rl[L]["bytes"])
        assert man_p in rt
    key = hashlib.sha256("".join(f"L{L} {layers[str(L)]['manifest_sha256']}\n" for L in LAYERS).encode()).hexdigest()[:16]
    top_sh = {f[1]: f[2] for f in tf}
    for p in pred:
        top_sh.setdefault(p, rt[p]["lfs_sha256"] or rt[p]["oid"])
    comp = dict(format="nq-complete-v1", repo=repo, artifact="nq-glm53flash-b15", key=key, n_layers=len(LAYERS),
                layers_range=[LAYERS[0], LAYERS[-1]], n_experts=NEXP, layers=layers, top=top_sh,
                total_layer_bytes=sum(v["bytes"] for v in layers.values()), time=mel())
    os.makedirs(f"{root}/upload", exist_ok=True)
    cp = f"{root}/upload/COMPLETE"
    json.dump(comp, open(cp, "w"), indent=1)
    print(f"COMPLETE ok: 42 layers + {len(tf)} top + {len(pred)} predictor files verified; key {key}; "
          f"{comp['total_layer_bytes'] / 1e9:.1f} GB of layers{'' if go else '  [DRY RUN: written to ' + cp + ' only]'}", flush=True)
    if not go:
        return 0
    ci = with_retry(lambda: api.create_commit(repo, operations=[CommitOperationAdd(path_in_repo="COMPLETE", path_or_fileobj=cp)],
                                              commit_message=f"COMPLETE: 42 layers, key {key}"), what="commit COMPLETE")
    rt = with_retry(lambda: remote_tree(api, repo), what="relist root")
    ok = same_remote(rt.get("COMPLETE"), cp, sha256(cp), os.path.getsize(cp))
    json.dump(dict(commit=ci.oid, key=key, verified=ok, time=mel()), open(f"{root}/upload/COMPLETE.rec.json", "w"), indent=1)
    print(f"COMPLETE commit {ci.oid} verified {ok}", flush=True)
    return 0 if ok else 3


# ---------------------------------------------------------------- watch
def fin_ok(root, L):
    try:
        return json.load(open(f"{root}/fin/L{L}.json")).get("rc") == 0
    except Exception:
        return False


def uploaded_for_current(root, L):
    """upload/L{L}.json says uploaded AND it was for the current local manifest (a re-finalized layer re-uploads)."""
    try:
        rec = json.load(open(f"{root}/upload/L{L}.json"))
    except Exception:
        return False
    if rec.get("status") != "uploaded":
        return False
    mp = f"{root}/L{L}/manifest.json"
    return not os.path.exists(mp) or rec.get("manifest_sha256") == sha256(mp)


def ensure_repo(repo, create):
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        api.repo_info(repo)
        return True
    except Exception as e:
        if not is_404(e):
            raise
    if not create:
        print(f"{mel()} repo {repo} does not exist; pass --create-repo public|private to create it", flush=True)
        return False
    api.create_repo(repo, repo_type="model", private=(create == "private"), exist_ok=True)
    print(f"{mel()} created repo {repo} ({create})", flush=True)
    return True


def watch(root, go=False, repo=REPO, sleep_s=120, once=False, forever=False, create=None):
    rd = f"{root}/upload"
    os.makedirs(rd, exist_ok=True)
    lk = open(f"{rd}/watch.lock", "w")
    try:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(f"another watcher holds {rd}/watch.lock")
    lk.write(f"{os.getpid()}\n"); lk.flush()
    nxt, fails, dry_done = {}, {}, {}
    print(f"{mel()} watch {'GO' if go else 'DRY RUN'} repo {repo} root {root} sleep {sleep_s}s pid {os.getpid()}", flush=True)
    repo_ok = not go
    while True:
        try:
            if go and not repo_ok:
                repo_ok = ensure_repo(repo, create)
            done = [L for L in LAYERS if uploaded_for_current(root, L)]
            todo = [L for L in LAYERS if fin_ok(root, L) and L not in done and time.time() >= nxt.get(L, 0)]
            if not go:                            # dry: plan each (layer, manifest) once
                todo = [L for L in todo if dry_done.get(L) != (os.path.getmtime(f"{root}/L{L}/manifest.json")
                                                               if os.path.exists(f"{root}/L{L}/manifest.json") else None)]
            if go and not repo_ok:
                todo = []
            print(f"{mel()} uploaded {len(done)}/42, finalized {sum(fin_ok(root, L) for L in LAYERS)}/42, this pass {todo}", flush=True)
            for L in todo:
                try:
                    r = run(root, [L], go=go, repo=repo)[L]
                    if go and r["status"] != "uploaded":
                        raise RuntimeError(r["status"])
                    fails.pop(L, None)
                    if not go:
                        dry_done[L] = os.path.getmtime(f"{root}/L{L}/manifest.json")
                except BaseException as e:            # SystemExit from the guards too: never leave the loop on a layer
                    if isinstance(e, KeyboardInterrupt):
                        raise
                    n = fails[L] = fails.get(L, 0) + 1
                    w = min(7200, 600 * 2 ** (n - 1))
                    nxt[L] = time.time() + w
                    msg = f"{type(e).__name__}: {str(e)[:400]}"
                    print(f"{mel()} L{L} upload FAILED (#{n}), retry in {w}s :: {msg}", flush=True)
                    with open(f"{rd}/failures.jsonl", "a") as fh:
                        fh.write(json.dumps(dict(layer=L, n=n, time=mel(), err=msg)) + "\n")
                    if "403" in str(e):
                        open(f"{rd}/ALERT_403", "w").write(f"{mel()} L{L}: {msg}\n")
                        print(f"{mel()} ALERT: HF 403 (storage quota?) -- free space (delete whole repos; squash frees nothing)", flush=True)
                        break                         # same quota for every layer: wait for the next pass
            if once:
                return
            if len(done) == len(LAYERS) and not forever:
                print(f"{mel()} all 42 layers uploaded; next: top + complete", flush=True)
                return
        except Exception:
            print(f"{mel()} watch pass error (continuing):\n{traceback.format_exc()[-1500:]}", flush=True)
            if once:
                return
        time.sleep(sleep_s)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("layers", "top", "complete", "status", "watch"))
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--layers", default="", help="comma list (layers mode)")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--go", action="store_true", help="really upload (dry run otherwise)")
    ap.add_argument("--top-dir", default=REL, help="release staging dir of top-level files")
    ap.add_argument("--top-files", default="", help="comma list of rel paths in --top-dir (default: every file)")
    ap.add_argument("--sleep", type=int, default=120)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--forever", action="store_true", help="watch: keep polling after 42/42 (refits)")
    ap.add_argument("--create-repo", choices=("public", "private"), help="watch --go: create the repo if absent")
    a = ap.parse_args()
    names = [x for x in a.top_files.split(",") if x] or None
    if a.cmd == "top":
        run_top(a.top_dir, names, go=a.go, repo=a.repo, rec_dir=f"{a.root}/upload")
    elif a.cmd == "complete":
        sys.exit(run_complete(a.root, a.top_dir, go=a.go, repo=a.repo, names=names))
    elif a.cmd == "status":
        print(json.dumps(remote_layers(a.repo, cache=f"{a.root}/upload/_remote"), indent=1))
    elif a.cmd == "watch":
        watch(a.root, go=a.go, repo=a.repo, sleep_s=a.sleep, once=a.once, forever=a.forever, create=a.create_repo)
    else:
        run(a.root, [int(x) for x in a.layers.split(",")], go=a.go, repo=a.repo)
