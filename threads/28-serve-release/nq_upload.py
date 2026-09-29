"""Thread 28: upload a serving release (nq_release.py output) to the HF repo under serving/tp{N}/, layer by layer, in order,
COMPLETE last. DRY RUN unless --go.

  python nq_upload.py OUT --tp 4 [--repo jarrelscy/GLM-5.3-NestQuant-2-4bit] [--layers 3-77] [--go]

Per layer L (ascending): the files of L whose local sha256 differs from the Hub's (LFS sha256 for .bin/.pt, git blob sha1
for json) -- rank{r}/L{L}.bin, res/rank{r}/L{L}.pt, layers/L{L}.json -- plus a partial index (rank{r}.json +
manifest.json covering the layers verified on the Hub so far, + L) go in ONE commit; then the remote tree is re-listed and
every file of L must match (size + sha). Unchanged layers are skipped, so a refit re-uploads only the layers whose content
hash changed. Once every layer is verified: the full index + manifest (== the local ones), serving/nq_assemble.py, then
COMPLETE in its own last commit, then a final verify of the whole serving/tp{N}/ tree against the local build.
Refit: if the Hub has a COMPLETE and some layer must change, COMPLETE is deleted first (the release is incomplete while
blocks are replaced) and rewritten at the end. Only paths under serving/ are ever added or deleted (asserted); nothing in
layers/, README.md or the configs is touched. Upload records: OUT/serving/tp{N}/_uploads/L{L}.json (local only).
Token: huggingface_hub default lookup (HF_TOKEN or the cached token); never printed.
"""
import os, sys, json, time, shutil, argparse, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "25-campaign"))
import nq_release as NR
from nq25_upload import remote_tree, with_retry, git_blob_sha1, sha256

REPO = "jarrelscy/GLM-5.3-NestQuant-2-4bit"
LFS_EXT = (".bin", ".pt")


def local_sha(p, rel, known=None):
    """(kind, digest): LFS sha256 for .bin/.pt (from the layer block when known), git blob sha1 otherwise"""
    if rel.endswith(LFS_EXT):
        return "lfs", known or sha256(p)
    return "blob", git_blob_sha1(p)


def matches(r, kind, dig, nb):
    if r is None or r["size"] != nb:
        return False
    return (r["lfs_sha256"] == dig) if kind == "lfs" else (r["oid"] == dig and r["lfs_sha256"] is None)


def layer_files(T, tp, L):
    """[(local path, repo-relative path under serving/tp{N}/, known sha256 or None)]"""
    b = json.load(open(f"{T}/layers/L{L}.json")); out = []
    for r in range(tp):
        e = b["ranks"][str(r)]
        out += [(f"{T}/{e['rec']['file']}", e["rec"]["file"], e["rec"]["sha256"]),
                (f"{T}/{e['res']['file']}", e["res"]["file"], e["res"]["sha256"])]
    out.append((f"{T}/layers/L{L}.json", f"layers/L{L}.json", None))
    return out, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--tp", type=int, default=4); ap.add_argument("--repo", default=REPO)
    ap.add_argument("--layers", default=None); ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
    api = HfApi(); T = NR.tpdir(a.out, a.tp); P = f"serving/tp{a.tp}"
    man = json.load(open(f"{T}/manifest.json"))
    if not os.path.exists(f"{T}/COMPLETE"):
        sys.exit(f"{T}/COMPLETE missing: build (nq_release.py) all layers first")
    comp = json.load(open(f"{T}/COMPLETE"))
    Ls = sorted(int(L) for L in comp["layers"]) if a.layers is None else NR.parse_layers(a.layers)
    os.makedirs(f"{T}/_uploads", exist_ok=True)

    def add(ops, lp, rel):
        path = f"{P}/{rel}"; assert path.startswith("serving/") and ".." not in path, path
        ops.append(CommitOperationAdd(path_in_repo=path, path_or_fileobj=lp))

    def commit(ops, msg):
        for o in ops:
            assert o.path_in_repo.startswith("serving/"), o.path_in_repo
        return with_retry(lambda: api.create_commit(a.repo, operations=ops, commit_message=msg), what=msg)

    rt = with_retry(lambda: remote_tree(api, a.repo, P), what="list")
    rel_tree = lambda t: {k[len(P) + 1:]: v for k, v in t.items()}
    rt = rel_tree(rt)
    plan = {}
    for L in Ls:
        files, _ = layer_files(T, a.tp, L); todo = []
        for lp, rel, known in files:
            kind, dig = local_sha(lp, rel, known)
            if not matches(rt.get(rel), kind, dig, os.path.getsize(lp)):
                todo.append((lp, rel, kind, dig))
        plan[L] = todo
    changed = [L for L in Ls if plan[L]]
    nbytes = sum(os.path.getsize(x[0]) for L in changed for x in plan[L])
    print(f"{a.repo}:{P}: {len(Ls)} layers, {len(changed)} to upload ({nbytes/1e9:.1f} GB): {changed[:10]}{'...' if len(changed) > 10 else ''}", flush=True)
    if not a.go:
        print("dry run (--go to upload)"); return
    if changed and "COMPLETE" in rt:
        commit([CommitOperationDelete(path_in_repo=f"{P}/COMPLETE")], f"NestQuant serving tp{a.tp}: refit of {len(changed)} layers (release incomplete until COMPLETE)")
        print("deleted remote COMPLETE (refit in progress)", flush=True)
    done = {int(L) for L in man["layers_present"]} - set(changed)          # verified on the Hub already (unchanged)
    stage = f"{T}/_uploads/stage"
    t0 = time.time(); sent = 0
    for L in changed:
        t = time.time(); ops = []
        for lp, rel, kind, dig in plan[L]:
            add(ops, lp, rel)
        shutil.rmtree(stage, ignore_errors=True)
        NR.build_index(a.out, a.tp, only=done | {L}, dest=stage)            # index of what the Hub holds after this commit
        for r in range(a.tp):
            add(ops, f"{stage}/rank{r}.json", f"rank{r}.json")
        add(ops, f"{stage}/manifest.json", "manifest.json")
        if os.path.exists(f"{stage}/COMPLETE"):
            os.remove(f"{stage}/COMPLETE")
        nb = sum(os.path.getsize(x[0]) for x in plan[L])
        ci = commit(ops, f"NestQuant serving tp{a.tp}: layer {L} ({len(plan[L])} files, {nb/1e9:.2f} GB)")
        rl = rel_tree(with_retry(lambda: remote_tree(api, a.repo, P), what=f"verify L{L}"))
        files, blk = layer_files(T, a.tp, L); bad = []
        for lp, rel, known in files:
            kind, dig = local_sha(lp, rel, known)
            if not matches(rl.get(rel), kind, dig, os.path.getsize(lp)):
                bad.append(rel)
        if bad:
            sys.exit(f"L{L}: remote verify failed for {bad}")
        done.add(L); sent += nb
        json.dump(dict(L=L, commit=getattr(ci, "oid", None), layer_hash=blk["layer_hash"], files=[x[1] for x in plan[L]],
                       bytes=nb, secs=round(time.time() - t, 1), time=time.strftime("%Y-%m-%d %H:%M:%S")),
                  open(f"{T}/_uploads/L{L}.json", "w"), indent=1)
        el = time.time() - t0
        print(f"L{L}: uploaded + verified {nb/1e9:.2f} GB in {time.time()-t:.0f}s ({sent/el/1e6:.0f} MB/s avg), commit {getattr(ci, 'oid', '?')[:10]}", flush=True)
    # full index + manifest (== local) + the assemble script, then COMPLETE alone, last
    rt = rel_tree(with_retry(lambda: remote_tree(api, a.repo, P), what="list final"))
    ops = []
    for rel in [f"rank{r}.json" for r in range(a.tp)] + ["manifest.json"]:
        if not matches(rt.get(rel), "blob", git_blob_sha1(f"{T}/{rel}"), os.path.getsize(f"{T}/{rel}")):
            add(ops, f"{T}/{rel}", rel)
    asm = with_retry(lambda: remote_tree(api, a.repo, "serving"), what="list serving").get("serving/nq_assemble.py")
    if not matches(asm, "blob", git_blob_sha1(f"{HERE}/nq_assemble.py"), os.path.getsize(f"{HERE}/nq_assemble.py")):
        ops.append(CommitOperationAdd(path_in_repo="serving/nq_assemble.py", path_or_fileobj=f"{HERE}/nq_assemble.py"))
    if ops:
        commit(ops, f"NestQuant serving tp{a.tp}: full index + manifest")
    rt = rel_tree(with_retry(lambda: remote_tree(api, a.repo, P), what="verify all"))
    bad = []
    for L in sorted(int(x) for x in comp["layers"]):
        for lp, rel, known in layer_files(T, a.tp, L)[0]:
            kind, dig = local_sha(lp, rel, known)
            if not matches(rt.get(rel), kind, dig, os.path.getsize(lp)):
                bad.append(rel)
    for rel in [f"rank{r}.json" for r in range(a.tp)] + ["manifest.json"]:
        if not matches(rt.get(rel), "blob", git_blob_sha1(f"{T}/{rel}"), os.path.getsize(f"{T}/{rel}")):
            bad.append(rel)
    if bad:
        sys.exit(f"final verify failed ({len(bad)}): {bad[:8]} -- COMPLETE not written")
    if not matches(rt.get("COMPLETE"), "blob", git_blob_sha1(f"{T}/COMPLETE"), os.path.getsize(f"{T}/COMPLETE")):
        ci = commit([CommitOperationAdd(path_in_repo=f"{P}/COMPLETE", path_or_fileobj=f"{T}/COMPLETE")],
                    f"NestQuant serving tp{a.tp}: COMPLETE ({len(comp['layers'])} layers)")
        print(f"COMPLETE written, commit {getattr(ci, 'oid', '?')[:10]}", flush=True)
    rt = rel_tree(with_retry(lambda: remote_tree(api, a.repo, P), what="verify COMPLETE"))
    assert matches(rt.get("COMPLETE"), "blob", git_blob_sha1(f"{T}/COMPLETE"), os.path.getsize(f"{T}/COMPLETE"))
    print(f"serving tp{a.tp}: release COMPLETE on {a.repo} ({len(comp['layers'])} layers, {sent/1e9:.1f} GB sent)", flush=True)


if __name__ == "__main__":
    main()
