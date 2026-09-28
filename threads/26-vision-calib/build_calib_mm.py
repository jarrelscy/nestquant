#!/usr/bin/env python3
"""Build the multimodal (image+text) calibration side-dataset for GLM-5.3.

CPU-ONLY, streaming, idempotent. Produces /data/glm53-calib-mm/:
  samples.jsonl                    {domain, id, caption, image_file,
                                    tensor_file, tensor_key, grid_thw,
                                    source, license}
  pixels_<domain>.safetensors      fp16 [1024, 3, 14, 14] per sample
                                   (key = sample id), 256 vision tokens each
  images/<domain>/<id>.jpg         the exact 448x448 RGB input, for repro
  README.md

Four domains, ~300 samples each, all from UNGATED HF datasets verified to
stream: medical (ROCOv2 radiology), screenshots/UI (WebSight), natural
(COCO-Caption2017), OCR/document (SynthDoG-en).

Every image is bicubic-resized to 448x448 RGB, then run through the grafted
checkpoint's own KimiK25VisionProcessor (loaded from /data/glm53-vision):
448/14 = 32x32 patches, 2x2 merge -> EXACTLY 256 vision tokens per image
(grid_thw = [1, 32, 32]), matching tools/capture53/vision_probe53.py's
splice protocol (IMG_BEGIN 154830 + 256 x IMG_TOK 154854 + IMG_END 154831
at the embedding layer).

Memory notes (shared box, ~5 GB anon budget):
  - NO IterableDataset.shuffle(): its shard-order randomization opens
    several remote parquet files at once and the per-file readahead blew
    anon RSS past 17 GB on synthdog-en. Sample diversity comes from seeded
    rejection sampling (keep each streamed row with prob KEEP_P, seed 42)
    over a strictly sequential stream instead.
  - Each domain is built in its own subprocess with RLIMIT_DATA = 5 GB, so
    arrow/fsspec allocations are returned to the OS between domains and a
    regression can never OOM the box.

Run (orchestrates all four domains, then merges + verifies):
  CUDA_VISIBLE_DEVICES= HF_HOME=/data/hf-cache-mm \
    /tmp/venv-glm53/bin/python tools/capture53/build_calib_mm.py
Single domain (what the orchestrator calls internally):
  ... build_calib_mm.py --domain medical
"""
import argparse
import importlib.util
import io
import json
import os
import random
import resource
import subprocess
import sys
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # never touch GPUs
os.environ.setdefault("HF_HOME", "/tmp/nestquant/hf-cache-mm")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

OUT = os.environ.get("GLM53_MM_CALIB", "/tmp/nestquant/calib-mm")
VISION = os.environ.get("GLM_VISION_DIR", "/tmp/nestquant/vision-graft")          # grafted kimi_k25 processor files
SIZE = 448                             # -> 32x32 patches -> 256 vision tokens
N_PER_DOMAIN = 300
N_HELDOUT = 50                         # T26: extra samples streamed AFTER the 300
                                       # (ids _0300.._0349) -> heldout/ split; the
                                       # first 300 are byte-identical to the original recipe
N_TOTAL = N_PER_DOMAIN + N_HELDOUT
SEED = 42
KEEP_P = 1 / 3                         # rejection-sampling keep probability
MIN_CAPTION = 20
MAX_CAPTION = 1500                     # cap OCR/HTML-derived text
EXPECT_GRID = [1, 32, 32]              # 1*32*32/(2*2) = 256 tokens
MEM_CAP = 5 * 2**30                    # RLIMIT_DATA per domain worker

# domain -> ordered candidate list; first one that streams wins.
# extract(example) -> caption str or None (None = reject sample)
DOMAINS = {
    "medical": [
        dict(source="eltorio/ROCOv2-radiology", config=None, split="train",
             license="CC BY-NC-SA 4.0",
             extract=lambda ex: ex.get("caption")),
        dict(source="mdwiratathya/ROCO-radiology", config=None, split="train",
             license="see dataset card (ROCO)",
             extract=lambda ex: ex.get("caption")),
    ],
    "screenshots": [
        dict(source="HuggingFaceM4/WebSight", config="v0.2", split="train",
             license="CC BY 4.0",
             extract=lambda ex: ex.get("llm_generated_idea")),
        dict(source="rootsautomation/ScreenSpot", config=None, split="test",
             license="Apache-2.0",
             extract=lambda ex: ex.get("instruction")),
    ],
    "natural": [
        dict(source="lmms-lab/COCO-Caption2017", config=None, split="val",
             license="CC BY 4.0 (annotations); COCO/Flickr terms (images)",
             extract=lambda ex: (ex.get("answer") or [None])[0]),
    ],
    "ocr": [
        dict(source="naver-clova-ix/synthdog-en", config=None, split="train",
             license="synthetic (SynthDoG / Donut toolkit, MIT)",
             extract=lambda ex: json.loads(ex["ground_truth"])
                 .get("gt_parse", {}).get("text_sequence")),
    ],
}


def log(msg):
    print(msg, flush=True)


def load_processor():
    """Import the checkpoint's own processor from /data/glm53-vision.

    (AutoImageProcessor needs torchvision under transformers 5.x, which the
    shared venv lacks; importing the trust_remote_code modules directly is
    byte-identical preprocessing.)
    """
    pkg = types.ModuleType("k25pkg")
    pkg.__path__ = [VISION]
    sys.modules["k25pkg"] = pkg
    for mod in ("media_utils", "kimi_k25_vision_processing"):
        spec = importlib.util.spec_from_file_location(
            f"k25pkg.{mod}", os.path.join(VISION, f"{mod}.py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules[f"k25pkg.{mod}"] = m
        spec.loader.exec_module(m)
    cfg = json.load(open(os.path.join(VISION, "preprocessor_config.json")))
    vp = sys.modules["k25pkg.kimi_k25_vision_processing"]
    return vp.KimiK25VisionProcessor(media_proc_cfg=cfg["media_proc_cfg"])


def mostly_english(text):
    if not text:
        return False
    ascii_frac = sum(c.isascii() for c in text) / len(text)
    return ascii_frac >= 0.9


def domain_done(domain):
    jl = os.path.join(OUT, f"samples_{domain}.jsonl")
    st = os.path.join(OUT, f"pixels_{domain}.safetensors")
    if not (os.path.exists(jl) and os.path.exists(st)):
        return False
    return sum(1 for _ in open(jl)) >= N_PER_DOMAIN


def build_domain(domain):
    import torch
    from datasets import load_dataset
    from PIL import Image
    from safetensors.torch import save_file

    proc = load_processor()
    img_dir = os.path.join(OUT, "images", domain)
    os.makedirs(img_dir, exist_ok=True)

    last_err = None
    for cand in DOMAINS[domain]:
        src = cand["source"]
        try:
            ds = load_dataset(src, cand["config"], split=cand["split"],
                              streaming=True)
            it = iter(ds)
            first = next(it)          # verifies real downloadability
        except Exception as e:
            log(f"[{domain}] {src} unusable ({type(e).__name__}: "
                f"{str(e)[:120]}) -> next candidate")
            last_err = e
            continue

        log(f"[{domain}] streaming {src} split={cand['split']} "
            f"(sequential, keep_p={KEEP_P:.3f}, seed {SEED})")
        rng = random.Random(SEED)
        records, tensors = [], {}
        seen = kept = 0
        import itertools
        for ex in itertools.chain([first], it):
            if kept >= N_TOTAL:
                break
            seen += 1
            if seen > 50 * N_TOTAL:        # safety valve
                break
            if rng.random() >= KEEP_P:     # seeded thinning, no shuffle buffer
                continue
            try:
                cap = cand["extract"](ex)
            except Exception:
                continue
            if not cap or len(cap.strip()) < MIN_CAPTION:
                continue
            cap = " ".join(cap.split())[:MAX_CAPTION]
            if not mostly_english(cap):
                continue
            img = ex.get("image")
            try:
                if img is None:
                    continue
                if not isinstance(img, Image.Image):
                    img = Image.open(io.BytesIO(img["bytes"]))
                img = img.convert("RGB").resize(
                    (SIZE, SIZE), Image.Resampling.BICUBIC)
            except Exception:
                continue
            out = proc.preprocess([{"type": "image", "image": img}],
                                  return_tensors="pt")
            grid = out["grid_thws"][0].tolist()
            assert grid == EXPECT_GRID, f"unexpected grid {grid}"
            sid = f"{domain}_{kept:04d}"
            img_file = f"images/{domain}/{sid}.jpg"
            img.save(os.path.join(OUT, img_file), quality=90)
            tensors[sid] = out["pixel_values"].to(torch.float16).contiguous()
            records.append({
                "domain": domain,
                "id": sid,
                "caption": cap,
                "image_file": img_file,
                "tensor_file": f"pixels_{domain}.safetensors",
                "tensor_key": sid,
                "grid_thw": grid,
                "n_vision_tokens": 256,
                "source": src,
                "source_split": cand["split"],
                "license": cand["license"],
            })
            kept += 1
            if kept % 50 == 0:
                anon = resource.getrusage(
                    resource.RUSAGE_SELF).ru_maxrss / 1048576
                log(f"[{domain}] {kept}/{N_TOTAL} "
                    f"(scanned {seen}, peak rss {anon:.2f} GB)")

        if kept < 100:
            log(f"[{domain}] {src} too sparse ({kept}) -> next candidate")
            continue
        if kept < N_TOTAL:
            log(f"[{domain}] only {kept} usable samples from {src}; keeping")
        n_fit = min(kept, N_PER_DOMAIN)
        fit_ids = [r["id"] for r in records[:n_fit]]
        ho_recs = records[n_fit:]
        ho_tensors = {r["id"]: tensors.pop(r["id"]) for r in ho_recs}
        records = records[:n_fit]
        hdir = os.path.join(OUT, "heldout")
        os.makedirs(hdir, exist_ok=True)
        for r in ho_recs:
            r["tensor_file"] = f"heldout/pixels_{domain}.safetensors"
            r["split"] = "heldout"
        for r in records:
            r["split"] = "calib"
        if ho_tensors:
            save_file(ho_tensors, os.path.join(hdir, f"pixels_{domain}.safetensors"),
                      metadata={"dtype": "float16",
                                "shape_per_key": "[1024, 3, 14, 14]",
                                "grid_thw": json.dumps(EXPECT_GRID),
                                "vision_tokens_per_image": "256",
                                "source": src, "split": "heldout"})
        with open(os.path.join(hdir, f"samples_{domain}.jsonl"), "w") as f:
            for r in ho_recs:
                f.write(json.dumps(r) + "\n")

        save_file(tensors, os.path.join(OUT, f"pixels_{domain}.safetensors"),
                  metadata={"dtype": "float16",
                            "shape_per_key": "[1024, 3, 14, 14]",
                            "grid_thw": json.dumps(EXPECT_GRID),
                            "vision_tokens_per_image": "256",
                            "source": src})
        with open(os.path.join(OUT, f"samples_{domain}.jsonl"), "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        log(f"[{domain}] done: {kept} samples from {src} (scanned {seen})")
        return kept
    raise RuntimeError(f"no usable dataset for domain {domain}: {last_err}")


README = """# glm53-calib-mm: multimodal calibration side-dataset

Additive image+text calibration set for the GLM-5.3 hybrid quant project.
Built CPU-only by `tools/capture53/build_calib_mm.py` (seed {seed}); the
text-only calib at /data/glm52-calib-v3 is untouched.

## Purpose

The shipped hybrid's calib was text-only while the checkpoint now carries a
grafted GLM-5V vision stack (vision_tower/mm_projector + kimi_k25
processor). Vision tokens flow through the same MoE layers, so the planned
GPU activation-capture RERUN should include image+text sequences so REAP
expert salience stats and AQLM Hessians see vision-token distributions
(REAP re-tier + AQLM converge redo).

## Domains ({n} samples each target)

| domain      | source dataset                | license |
|-------------|-------------------------------|---------|
{rows}

Filters: decodable image, caption >= {minc} chars after whitespace
normalization, >= 90% ASCII (English heuristic), text capped at {maxc}
chars. Sampling: strictly sequential stream thinned by seeded rejection
sampling (keep_p = 1/3, seed {seed}) — IterableDataset.shuffle is
deliberately avoided (its shard randomization + parquet readahead blew
anon RSS >17 GB on this box).

## Preprocessing

Each image: RGB, bicubic resize to {size}x{size}, then the grafted
checkpoint's own `KimiK25VisionProcessor` (modules imported from
/data/glm53-vision, byte-identical to trust_remote_code loading):
normalize mean=std=0.5, patchify 14x14 -> pixel tensor **[1024, 3, 14, 14]
fp16** per image, `grid_thw = [1, 32, 32]` -> with 2x2 merge exactly
**256 vision tokens per image**.

## Files

- `samples.jsonl` — one record per sample: domain, id, caption,
  image_file, tensor_file, tensor_key, grid_thw, source, license.
  (`samples_<domain>.jsonl` are the per-domain parts; `samples.jsonl` is
  their concatenation.)
- `pixels_<domain>.safetensors` — key = sample id, value fp16
  [1024, 3, 14, 14] processor-ready pixel patches.
- `images/<domain>/<id>.jpg` — the exact 448x448 input, for reproduction.

## Downstream use (future GPU capture run)

Follow `tools/capture53/vision_probe53.py` / `graft_vision53.py`:
run vision_tower + mm_projector on `pixel_values` (or feed the saved
patches directly), build token ids
`[gMASK]<sop><|user|>\\n` + `<|media_begin|>`(154830) + 256 x
`<|media_pad|>`(154854) + `<|media_end|>`(154831) + prompt + caption,
embed, then overwrite the 256 media-pad positions with the projected
image features before the layer-streamed forward. The existing capture
(`stream_capture53.py`) consumes token-id shards
(`shard_*.npy` reshaped to [N, 2048]); these multimodal samples are NOT in
that format — the capture script needs a small extension to splice
embeddings, which is why captions + pixel tensors are stored separately
here rather than pre-tokenized.
"""


def orchestrate():
    os.makedirs(OUT, exist_ok=True)
    counts, meta = {}, {}
    for domain in DOMAINS:
        if domain_done(domain):
            counts[domain] = sum(1 for _ in open(
                os.path.join(OUT, f"samples_{domain}.jsonl")))
            log(f"[{domain}] already built ({counts[domain]} samples) -> skip")
        else:
            r = subprocess.run(
                [sys.executable, os.path.abspath(__file__),
                 "--domain", domain])
            if not domain_done(domain):
                raise RuntimeError(f"domain {domain} worker failed "
                                   f"(rc={r.returncode})")
            if r.returncode != 0:
                log(f"[{domain}] worker rc={r.returncode} but artifacts "
                    f"complete (known benign shutdown crash) -> continue")
            counts[domain] = sum(1 for _ in open(
                os.path.join(OUT, f"samples_{domain}.jsonl")))
        first = json.loads(open(os.path.join(
            OUT, f"samples_{domain}.jsonl")).readline())
        meta[domain] = (first["source"], first["license"])

    # merge per-domain jsonl -> samples.jsonl
    with open(os.path.join(OUT, "samples.jsonl"), "w") as out:
        for domain in DOMAINS:
            with open(os.path.join(OUT, f"samples_{domain}.jsonl")) as f:
                out.write(f.read())

    # T26: held-out split merge
    with open(os.path.join(OUT, "heldout", "samples.jsonl"), "w") as out:
        for domain in DOMAINS:
            with open(os.path.join(OUT, "heldout", f"samples_{domain}.jsonl")) as f:
                out.write(f.read())

    rows = "".join(f"| {d:<11} | {meta[d][0]:<29} | {meta[d][1]} |\n"
                   for d in DOMAINS)
    with open(os.path.join(OUT, "README.md"), "w") as f:
        f.write(README.format(seed=SEED, n=N_PER_DOMAIN, rows=rows,
                              minc=MIN_CAPTION, maxc=MAX_CAPTION, size=SIZE))

    # verification pass: safetensors headers + counts
    import torch
    from safetensors import safe_open
    total = 0
    for domain in DOMAINS:
        p = os.path.join(OUT, f"pixels_{domain}.safetensors")
        with safe_open(p, "pt") as f:
            keys = list(f.keys())
            t = f.get_tensor(keys[0])
            assert t.shape == (1024, 3, 14, 14) and t.dtype == torch.float16
        n_jl = sum(1 for _ in open(os.path.join(
            OUT, f"samples_{domain}.jsonl")))
        assert n_jl == len(keys), f"{domain}: jsonl {n_jl} != st {len(keys)}"
        log(f"[verify] {domain}: {len(keys)} tensors [1024,3,14,14] fp16, "
            f"jsonl matches")
        total += len(keys)
    log(f"[verify] total {total} samples across {len(DOMAINS)} domains")
    ho = [json.loads(l) for l in open(os.path.join(OUT, "heldout", "samples.jsonl"))]
    cal = [json.loads(l) for l in open(os.path.join(OUT, "samples.jsonl"))]
    ids_c, ids_h = {r["id"] for r in cal}, {r["id"] for r in ho}
    assert len(ids_c) == len(cal) and len(ids_h) == len(ho) and not (ids_c & ids_h)
    caps_c = {r["caption"] for r in cal}
    log(f"[verify] heldout {len(ho)} samples, ids disjoint; "
        f"caption overlap with calib: {sum(r['caption'] in caps_c for r in ho)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=list(DOMAINS), default=None)
    args = ap.parse_args()
    if args.domain:
        # worker mode: hard cap anon memory so a leak can't OOM the box
        resource.setrlimit(resource.RLIMIT_DATA, (MEM_CAP, MEM_CAP))
        build_domain(args.domain)
        # datasets' streaming threads abort in interpreter finalization
        # (PyGILState_Release crash); artifacts are fully written, so skip
        # finalization entirely.
        sys.stdout.flush()
        os._exit(0)
    else:
        orchestrate()


if __name__ == "__main__":
    main()
