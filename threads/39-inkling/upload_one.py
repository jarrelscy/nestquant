"""Single-commit upload of the release folder (data mostly already in HF storage from the first attempt), then COMPLETE."""
import json, hashlib, time
from huggingface_hub import HfApi
REPO = "jarrelscy/Inkling-2-Small-NestQuant-1.75-4bit"; R = "/tmp/nq_inkling/release"
api = HfApi()
time.sleep(int(__import__("os").environ.get("UP_DELAY", "0")))     # let HF's hourly commit window clear first
for i in range(20):
    try:
        c = api.upload_folder(repo_id=REPO, folder_path=R, allow_patterns=["layers/**", "README.md"],
                              commit_message="Inkling-2-Small NestQuant 1.75/4: all 48 layers (TP2) + README")
        print("committed", c.oid, flush=True); break
    except Exception as e:
        print("retry", i, str(e)[:300], flush=True); time.sleep(3900)
else:
    raise SystemExit("upload_folder failed 20 times")
done = {}
for L in range(48):
    m = json.load(open(f"{R}/layers/L{L}/manifest.json"))
    done[f"L{L}"] = {f"layers/L{L}/{k}": v["sha256"] for k, v in m["files"].items()}
    done[f"L{L}"][f"layers/L{L}/manifest.json"] = hashlib.sha256(open(f"{R}/layers/L{L}/manifest.json", "rb").read()).hexdigest()
json.dump(dict(format="nestquant-v1-tp-safetensors", tp=2, layers=done), open(f"{R}/COMPLETE", "w"), indent=1)
api.upload_file(path_or_fileobj=f"{R}/COMPLETE", path_in_repo="COMPLETE", repo_id=REPO, commit_message="COMPLETE: all 48 layers")
print("UPLOAD_DONE", flush=True)
