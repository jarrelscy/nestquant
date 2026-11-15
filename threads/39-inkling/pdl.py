import sys, os, subprocess, concurrent.futures as cf, urllib.request
name, out = sys.argv[1], sys.argv[2]
url = f"https://huggingface.co/thinkingmachines/Inkling-2-Small/resolve/main/{name}"
req = urllib.request.Request(url, method="HEAD")
size = int(urllib.request.urlopen(req).headers["Content-Length"])
P = 256 << 20
parts = [(i, o, min(size, o + P) - 1) for i, o in enumerate(range(0, size, P))]
tmp = f"{out}.parts"; os.makedirs(tmp, exist_ok=True)
def get(p):
    i, a, b = p; f = f"{tmp}/{i:05d}"
    if os.path.exists(f) and os.path.getsize(f) == b - a + 1:
        return
    for _ in range(8):
        r = subprocess.run(["curl", "-sL", "--max-time", "600", "-r", f"{a}-{b}", "-o", f + ".t", url])
        if r.returncode == 0 and os.path.getsize(f + ".t") == b - a + 1:
            os.replace(f + ".t", f); return
    raise RuntimeError(f"part {i} failed")
with cf.ThreadPoolExecutor(24) as ex:
    list(ex.map(get, parts))
with open(out + ".tmp", "wb") as o:
    for i, _, _ in parts:
        o.write(open(f"{tmp}/{i:05d}", "rb").read())
assert os.path.getsize(out + ".tmp") == size
os.replace(out + ".tmp", out); subprocess.run(["rm", "-rf", tmp]); print("done", name, size)
