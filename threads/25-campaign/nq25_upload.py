"""Thread 25 upload step: finalized + checked layers -> HF repo (DRY RUN unless go=True).

Layout (proposal):
  layers/L{L}/tp{s}.safetensors  s = 0..7, nq_layer's TP8 shard dicts in safetensors (nq25_st.py; the .pt stay local)
  layers/L{L}/manifest.json      nq_layer manifest (files[] = the safetensors sha256/bytes, config, default allocation,
                                 + the campaign block: config_id, calibration, encoder, refcheck, code hashes)
  (later, top level: nestquant_index.json + README.md + config/tokenizer/vision/non-expert weights)
Only new/changed files are sent: local sha256 (manifest for tp files, computed for manifest.json) vs the Hub's LFS
sha256 (or git blob sha1 for small non-LFS files). One commit per layer, retried with backoff. After a commit the
remote tree is re-listed and the remote sha per file + commit oid are recorded in ROOT/uploads/L{L}.json.
The token comes from the huggingface_hub default lookup (HF_TOKEN env or ~/.cache/huggingface/token); never printed.
"""
import os, json, time, hashlib

REPO = "jarrelscy/GLM-5.3-NestQuant-2-4bit"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def git_blob_sha1(p):
    data = open(p, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def remote_tree(api, repo, prefix):
    """{path: dict(size, lfs_sha256, oid)} under prefix (empty if absent)."""
    out = {}
    try:
        for it in api.list_repo_tree(repo, path_in_repo=prefix, recursive=True, expand=True):
            if getattr(it, "size", None) is None:          # folder
                continue
            lfs = getattr(it, "lfs", None)
            out[it.path] = dict(size=it.size, lfs_sha256=(lfs.sha256 if lfs else None), oid=getattr(it, "blob_id", None))
    except Exception as e:                                 # EntryNotFound / RepositoryNotFound for a fresh path
        if "404" not in str(e) and "not found" not in str(e).lower() and "EntryNotFound" not in type(e).__name__:
            raise
    return out


def remote_layers(repo=REPO, cache=None):
    """HF is the source of truth: {L: dict(verified, config_id, code_id, commit, reason)} for every layers/L{L}/ on the
    Hub. verified = the remote manifest.json parses and every file it lists is on the Hub with that exact size and LFS
    sha256 (the manifest's own sha256 is the one nq_layer/nq25_st computed locally before upload)."""
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
        c = man.get("campaign") or {}
        out[L] = dict(verified=not bad and bool(man.get("files")), bad=bad, config_id=c.get("config_id"),
                      code_id=(c.get("code") or {}).get("code_id"), encoder=c.get("encoder"),
                      n_experts=man.get("n_experts"), bytes=sum(v["bytes"] for v in man.get("files", {}).values()),
                      reason="ok" if not bad else f"{len(bad)} files missing/mismatched")
    return out


def layer_files(root, L):
    d = f"{root}/L{L}"
    man = json.load(open(f"{d}/manifest.json"))
    files = [(f"{d}/{f}", f"layers/L{L}/{f}", info["sha256"], info["bytes"]) for f, info in sorted(man["files"].items())]
    mp = f"{d}/manifest.json"
    files.append((mp, f"layers/L{L}/manifest.json", sha256(mp), os.path.getsize(mp)))
    return files


def with_retry(fn, tries=6, what=""):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1:
                raise
            w = min(300, 10 * 2 ** i)
            print(f"  retry {what} in {w}s after {type(e).__name__}: {str(e)[:200]}", flush=True)
            time.sleep(w)


def run(root, layers, go=False, repo=REPO, allow_unchecked=False):
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi()
    res = {}
    tot_new = 0
    os.makedirs(f"{root}/uploads", exist_ok=True)
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
            same = r is not None and r["size"] == nb and (
                (r["lfs_sha256"] == sh) if r["lfs_sha256"] else (r["oid"] == git_blob_sha1(lp)))
            plan.append(dict(local=lp, path=rp, bytes=nb, sha256=sh, action="skip (same)" if same else
                             ("replace" if r is not None else "add")))
        new = [p for p in plan if not p["action"].startswith("skip")]
        nb_new = sum(p["bytes"] for p in new)
        tot_new += nb_new
        print(f"L{L}: {len(new)}/{len(plan)} files to send, {nb_new / 2**30:.2f} GiB"
              f"{'' if go else '  [DRY RUN]'}", flush=True)
        for p in plan:
            print(f"   {p['action']:<12} {p['path']:<28} {p['bytes'] / 2**20:10.1f} MiB  sha256 {p['sha256'][:16]}", flush=True)
        rec = dict(layer=L, repo=repo, plan=plan, bytes_to_send=nb_new, time=time.strftime("%Y-%m-%d %H:%M:%S"))
        if not go:
            rec["status"] = "dry-run"
            json.dump(rec, open(f"{root}/uploads/L{L}.plan.json", "w"), indent=1)
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
            ok = r is not None and r["size"] == p["bytes"] and (
                r["lfs_sha256"] == p["sha256"] if r["lfs_sha256"] else
                (os.path.exists(p["local"]) and r["oid"] == git_blob_sha1(p["local"])))
            if not ok:
                bad.append(p["path"])
        rec["status"] = "uploaded" if not bad else f"verify failed {bad}"
        json.dump(rec, open(f"{root}/uploads/L{L}.json", "w"), indent=1)
        res[L] = dict(status=rec["status"], commit=rec.get("commit"), bytes=nb_new, MBps=rec.get("MBps"))
        print(f"L{L}: {rec['status']} commit {rec.get('commit')} {rec.get('MBps')} MB/s", flush=True)
    print(f"total to send: {tot_new / 2**30:.2f} GiB over {len(layers)} layers{'' if go else ' [DRY RUN]'}", flush=True)
    return res


def run_top(top, names, go=False, repo=REPO, rec_dir=None):
    """top-level (non-layer) files from the staging dir `top` (nq25_nonexpert.py output) -> repo root. sha256 from
    nonexpert_manifest.json where listed, else computed. Same compare / commit / re-verify as the layers; one commit."""
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi()
    man = {f["file"]: f for f in json.load(open(f"{top}/nonexpert_manifest.json"))["files"]} \
        if os.path.exists(f"{top}/nonexpert_manifest.json") else {}
    files = []
    for n in names:
        lp = f"{top}/{n}"
        nb = os.path.getsize(lp)
        sh = man[n]["sha256"] if n in man and man[n]["bytes"] == nb else sha256(lp)
        files.append((lp, n, sh, nb))
    rt = {}
    rt = with_retry(lambda: {it.path: dict(size=it.size, lfs_sha256=(it.lfs.sha256 if getattr(it, "lfs", None) else None),
                                           oid=getattr(it, "blob_id", None))
                             for it in api.list_repo_tree(repo, expand=True) if getattr(it, "size", None) is not None},
                    what="list root")
    def same(lp, rp, sh, nb):
        r = rt.get(rp)
        return r is not None and r["size"] == nb and ((r["lfs_sha256"] == sh) if r["lfs_sha256"] else r["oid"] == git_blob_sha1(lp))
    new = [f for f in files if not same(*f)]
    nb_new = sum(f[3] for f in new)
    print(f"top: {len(new)}/{len(files)} files to send, {nb_new / 2**30:.2f} GiB{'' if go else '  [DRY RUN]'}", flush=True)
    rec = dict(repo=repo, files=[dict(path=f[1], bytes=f[3], sha256=f[2], action="send" if f in new else "skip (same)")
                                 for f in files], time=time.strftime("%Y-%m-%d %H:%M:%S"))
    if go and new:
        ops = [CommitOperationAdd(path_in_repo=rp, path_or_fileobj=lp) for lp, rp, _, _ in new]
        t0 = time.time()
        ci = with_retry(lambda: api.create_commit(repo, operations=ops, commit_message=f"NestQuant top-level: {len(new)} files"),
                        what="commit top")
        dt = time.time() - t0
        rec.update(commit=ci.oid, upload_s=round(dt, 1), MBps=round(nb_new / 1e6 / max(dt, 1e-3), 1))
        rt = with_retry(lambda: {it.path: dict(size=it.size, lfs_sha256=(it.lfs.sha256 if getattr(it, "lfs", None) else None),
                                               oid=getattr(it, "blob_id", None))
                                 for it in api.list_repo_tree(repo, expand=True) if getattr(it, "size", None) is not None},
                        what="relist root")
    bad = [f[1] for f in files if not same(*f)] if go else []
    rec["status"] = ("uploaded" if not bad else f"verify failed {bad}") if go else "dry-run"
    if rec_dir:
        os.makedirs(rec_dir, exist_ok=True)
        json.dump(rec, open(f"{rec_dir}/{'top' if go else 'top.dry'}.json", "w"), indent=1)   # a dry plan never hides the last real upload
        if go:
            with open(f"{rec_dir}/top_history.jsonl", "a") as fh:
                fh.write(json.dumps({k: rec.get(k) for k in ("time", "commit", "status", "MBps")} | dict(files=[x["path"] for x in rec.get("files", []) if x.get("action") == "send"])) + "\n")
    print(f"top: {rec['status']} commit {rec.get('commit')} {rec.get('MBps')} MB/s", flush=True)
    return rec


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--layers", default="", help="comma list")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--go", action="store_true", help="really upload (only after the lead's 'go upload')")
    ap.add_argument("--remote-status", action="store_true", help="just print remote_layers() (HF truth) as json")
    ap.add_argument("--top", help="staging dir of top-level files (nq25_nonexpert.py output)")
    ap.add_argument("--top-files", default="", help="comma list of file names in --top (default: nonexpert-* + manifest + index)")
    a = ap.parse_args()
    if a.top:
        import glob
        names = [x for x in a.top_files.split(",") if x] or sorted(os.path.basename(p) for p in glob.glob(f"{a.top}/nonexpert-*.safetensors")) \
            + ["nonexpert_manifest.json", "model.safetensors.index.json"]
        run_top(a.top, names, go=a.go, repo=a.repo, rec_dir=f"{a.root}/uploads")
    elif a.remote_status:
        print(json.dumps(remote_layers(a.repo, cache=f"{a.root}/_remote"), indent=1))
    else:
        run(a.root, [int(x) for x in a.layers.split(",")], go=a.go, repo=a.repo)
