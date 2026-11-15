"""Upload /tmp/nq_inkling/release to the private HF repo; COMPLETE (all layer file sha256) last."""
import json, os, hashlib
from huggingface_hub import HfApi
REPO = "jarrelscy/Inkling-2-Small-NestQuant-1.75-4bit"
R = "/tmp/nq_inkling/release"
api = HfApi()
api.upload_large_folder(repo_id=REPO, repo_type="model", folder_path=R, allow_patterns=["layers/**", "README.md"], num_workers=16)
done = {}
for L in range(48):
    m = json.load(open(f"{R}/layers/L{L}/manifest.json"))
    done[f"L{L}"] = {f"layers/L{L}/{k}": v["sha256"] for k, v in m["files"].items()}
    done[f"L{L}"][f"layers/L{L}/manifest.json"] = hashlib.sha256(open(f"{R}/layers/L{L}/manifest.json", "rb").read()).hexdigest()
json.dump(dict(format="nestquant-v1-tp-safetensors", tp=2, layers=done), open(f"{R}/COMPLETE", "w"), indent=1)
api.upload_file(path_or_fileobj=f"{R}/COMPLETE", path_in_repo="COMPLETE", repo_id=REPO, commit_message="COMPLETE: all 48 layers")
print("UPLOAD_DONE", flush=True)
