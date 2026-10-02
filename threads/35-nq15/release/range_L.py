import json, requests, os
from huggingface_hub import hf_hub_url, get_token
R = "jarrelscy/GLM-5.3-NestQuant-2-4bit"; D = "/tmp/nestquant/35-nq15/gate_a2/shipped"; L = int(__import__("os").environ["L"])
for r in range(4):
    idx = json.load(open(f"{D}/rank{r}.json")); rb = idx["rec_bytes"]; L0 = idx.get("L0", 3)
    off, n = (L - L0) * 256 * rb, 256 * rb
    u = hf_hub_url(R, f"rank{r}.bin", revision="3c6b26979ccad101b60b7253aa243f7e297d02d3")
    with requests.get(u, headers={"Authorization": f"Bearer {get_token()}", "Range": f"bytes={off}-{off+n-1}"}, stream=True, timeout=600) as q:
        q.raise_for_status(); b = b"".join(q.iter_content(1 << 22))
    assert len(b) == n, (len(b), n)
    open(f"{D}/rank{r}.L{L}.rec", "wb").write(b); print(r, rb, n, flush=True)
