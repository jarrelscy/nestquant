"""Thread 25: HF upload throughput test into a scratch path of the campaign repo (deleted afterwards).

  python nq25_hfspeed.py MODE [--gb 5] [--files 8] [--procs 4]
MODE: one    = one create_commit with all files (hf_xet parallelizes internally)
      multi  = --procs concurrent create_commit calls, files split among them
      large  = upload_large_folder(num_workers=--procs)
Random (non-dedupable) bytes, like the real payload. Token from the hub default lookup; never printed.
"""
import os, sys, time, json, argparse, shutil, numpy as np
from concurrent.futures import ThreadPoolExecutor
REPO = "jarrelscy/GLM-5.3-NestQuant-2-4bit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode"); ap.add_argument("--gb", type=float, default=5); ap.add_argument("--files", type=int, default=8)
    ap.add_argument("--procs", type=int, default=4); ap.add_argument("--dir", default="/tmp/nestquant/25-campaign/hfspeed")
    a = ap.parse_args()
    from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
    api = HfApi()
    tag = f"{a.mode}-{int(time.time())}"
    d = f"{a.dir}/{tag}"; os.makedirs(d)
    nb = int(a.gb * 1e9 / a.files)
    g = np.random.default_rng(int(time.time()))
    for i in range(a.files):
        g.integers(0, 256, nb, dtype=np.uint8).tofile(f"{d}/f{i}.bin")
    files = sorted(os.listdir(d)); tot = nb * a.files
    pre = f"_t25_scratch/{tag}"
    t0 = time.time()
    if a.mode == "one":
        api.create_commit(REPO, operations=[CommitOperationAdd(f"{pre}/{f}", f"{d}/{f}") for f in files],
                          commit_message="t25 speed test (scratch, deleted after)")
    elif a.mode == "multi":
        parts = [files[i::a.procs] for i in range(a.procs)]
        def one(p):
            return api.create_commit(REPO, operations=[CommitOperationAdd(f"{pre}/{f}", f"{d}/{f}") for f in p],
                                     commit_message="t25 speed test (scratch, deleted after)")
        with ThreadPoolExecutor(a.procs) as ex:
            list(ex.map(one, parts))
    elif a.mode == "large":
        api.upload_large_folder(repo_id=REPO, folder_path=a.dir, allow_patterns=[f"{tag}/*"], num_workers=a.procs,
                                print_report=False)
        pre = f"{tag}"
    dt = time.time() - t0
    r = dict(mode=a.mode, procs=a.procs, files=a.files, bytes=tot, s=round(dt, 1), MBps=round(tot / 1e6 / dt, 1),
             xet_hp=os.environ.get("HF_XET_HIGH_PERFORMANCE"), time=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(json.dumps(r), flush=True)
    open(f"{a.dir}/results.jsonl", "a").write(json.dumps(r) + "\n")
    api.delete_folder(pre, repo_id=REPO, commit_message="t25 speed test cleanup")
    shutil.rmtree(d)
    shutil.rmtree(f"{a.dir}/.cache", ignore_errors=True)


if __name__ == "__main__":
    main()
