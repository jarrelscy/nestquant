#!/usr/bin/env python3
"""Inkling-2-Small multimodal calibration samples (step 1 of 2; step 2 = pack_mm.py).

Port of nestquant T26 / glm52 build_calib_mm.py with the SAME sample selection:
  sources + fallbacks, split, seed 42, sequential stream, seeded rejection sampling keep_p = 1/3
  (one rng.random() per streamed row, before any filter), caption >= 20 chars after whitespace
  normalisation, capped at 1500 chars, >= 90% ASCII, decodable image; first 300 kept = calib,
  next 50 (ids <domain>_0300..0349) = heldout.
Changed vs T26 (model-specific only):
  - resize target 640x640 (bicubic) instead of 448: Inkling patch 20 x merge 2 = 40 px per LM image token,
    640/40 = 16x16 = 256 image tokens per image (same token count as T26's 256), no -1 padding, and
    640/20 = 32 patches/side = the vision tower's native posemb_grid_size (no posemb interpolation).
  - preprocessing by Inkling2ImageProcessor (AutoProcessor from /tmp/nq_inkling/model); per-image output
    saved to OUT/pix/<id>.pt = {"pixel_values": fp32 [256, 40, 40, 3], "image_grid_thw": int64 [1, 3] = [[1, 16, 16]]}.
  - each sample written immediately (no in-RAM tensor dict).

CPU only. Each domain runs in its own subprocess with RLIMIT_DATA = 8 GiB.
  CUDA_VISIBLE_DEVICES= .venv/bin/python build_mm.py              # all domains (parallel workers), then merge
  ... build_mm.py --domain medical                                # one domain
"""
import argparse, io, itertools, json, os, random, resource, subprocess, sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HOME", "/tmp/nq_inkling/corpus_mm/hf-cache")
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "2")
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

OUT = os.environ.get("INK_MM_OUT", "/tmp/nq_inkling/corpus_mm")
MODEL = "/tmp/nq_inkling/model"
SIZE = 640
N_PER_DOMAIN, N_HELDOUT = 300, 50
N_TOTAL = N_PER_DOMAIN + N_HELDOUT
SEED = 42
KEEP_P = 1 / 3
MIN_CAPTION, MAX_CAPTION = 20, 1500
EXPECT_GRID = [1, 16, 16]
N_VIS = 256
MEM_CAP = 8 * 2**30

DOMAINS = {   # identical to T26 build_calib_mm.py
    "medical": [
        dict(source="eltorio/ROCOv2-radiology", config=None, split="train", license="CC BY-NC-SA 4.0",
             extract=lambda ex: ex.get("caption")),
        dict(source="mdwiratathya/ROCO-radiology", config=None, split="train", license="see dataset card (ROCO)",
             extract=lambda ex: ex.get("caption")),
    ],
    "screenshots": [
        dict(source="HuggingFaceM4/WebSight", config="v0.2", split="train", license="CC BY 4.0",
             extract=lambda ex: ex.get("llm_generated_idea")),
        dict(source="rootsautomation/ScreenSpot", config=None, split="test", license="Apache-2.0",
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
             extract=lambda ex: json.loads(ex["ground_truth"]).get("gt_parse", {}).get("text_sequence")),
    ],
}


def log(m):
    print(m, flush=True)


def mostly_english(text):
    return bool(text) and sum(c.isascii() for c in text) / len(text) >= 0.9


def jl_path(domain):
    return os.path.join(OUT, "samples", f"samples_{domain}.jsonl")


def domain_done(domain):
    p = jl_path(domain)
    return os.path.exists(p) and sum(1 for _ in open(p)) >= N_PER_DOMAIN


def build_domain(domain):
    import torch
    from datasets import load_dataset
    from PIL import Image
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL).image_processor
    os.makedirs(os.path.join(OUT, "images", domain), exist_ok=True)
    os.makedirs(os.path.join(OUT, "pix"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "samples"), exist_ok=True)
    last_err = None
    for cand in DOMAINS[domain]:
        src = cand["source"]
        try:
            ds = load_dataset(src, cand["config"], split=cand["split"], streaming=True)
            it = iter(ds)
            first = next(it)
        except Exception as e:
            log(f"[{domain}] {src} unusable ({type(e).__name__}: {str(e)[:160]}) -> next candidate")
            last_err = e
            continue
        log(f"[{domain}] streaming {src} split={cand['split']} (sequential, keep_p={KEEP_P:.3f}, seed {SEED})")
        rng = random.Random(SEED)
        records = []
        seen = kept = 0
        for ex in itertools.chain([first], it):
            if kept >= N_TOTAL:
                break
            seen += 1
            if seen > 50 * N_TOTAL:
                break
            if rng.random() >= KEEP_P:
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
                orig_size = list(img.size)
                img = img.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BICUBIC)
            except Exception:
                continue
            out = proc(images=[img], return_tensors="pt")
            grid = out["image_grid_thw"][0].tolist()
            assert grid == EXPECT_GRID, f"unexpected grid {grid}"
            pv = out["pixel_values"].contiguous()
            assert tuple(pv.shape) == (N_VIS, 40, 40, 3) and pv.dtype == torch.float32, (pv.shape, pv.dtype)
            sid = f"{domain}_{kept:04d}"
            img_file = f"images/{domain}/{sid}.jpg"
            img.save(os.path.join(OUT, img_file), quality=90)
            pix_file = f"pix/{sid}.pt"
            torch.save({"pixel_values": pv, "image_grid_thw": out["image_grid_thw"].to(torch.int64)},
                       os.path.join(OUT, pix_file))
            records.append(dict(domain=domain, id=sid, caption=cap, image_file=img_file, pix_file=pix_file,
                                grid_thw=grid, n_vision_tokens=N_VIS, orig_size=orig_size, stream_index=seen - 1,
                                source=src, source_split=cand["split"], license=cand["license"],
                                split="calib" if kept < N_PER_DOMAIN else "heldout"))
            kept += 1
            if kept % 50 == 0:
                log(f"[{domain}] {kept}/{N_TOTAL} (scanned {seen}, peak rss "
                    f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.2f} GB)")
        if kept < 100:
            log(f"[{domain}] {src} too sparse ({kept}) -> next candidate")
            continue
        if kept < N_TOTAL:
            log(f"[{domain}] only {kept} usable samples from {src}; keeping")
        tmp = jl_path(domain) + ".tmp"
        with open(tmp, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, jl_path(domain))
        log(f"[{domain}] done: {kept} samples from {src} (scanned {seen})")
        return kept
    raise RuntimeError(f"no usable dataset for domain {domain}: {last_err}")


def orchestrate(parallel):
    os.makedirs(os.path.join(OUT, "samples"), exist_ok=True)
    todo = [d for d in DOMAINS if not domain_done(d)]
    for d in DOMAINS:
        if d not in todo:
            log(f"[{d}] already built -> skip")
    procs = []
    for d in todo:
        lf = open(os.path.join(OUT, "samples", f"build_{d}.log"), "w")
        p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--domain", d], stdout=lf, stderr=subprocess.STDOUT)
        procs.append((d, p))
        if not parallel:
            p.wait()
    for d, p in procs:
        rc = p.wait()
        if not domain_done(d):
            raise RuntimeError(f"domain {d} worker failed (rc={rc}); see samples/build_{d}.log")
    allr = [json.loads(l) for d in DOMAINS for l in open(jl_path(d))]
    cal = [r for r in allr if r["split"] == "calib"]
    ho = [r for r in allr if r["split"] == "heldout"]
    for name, rs in (("samples.jsonl", cal), ("heldout.jsonl", ho)):
        with open(os.path.join(OUT, "samples", name), "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
    ids_c, ids_h = {r["id"] for r in cal}, {r["id"] for r in ho}
    assert len(ids_c) == len(cal) and len(ids_h) == len(ho) and not (ids_c & ids_h)
    caps_c = {r["caption"] for r in cal}
    log(f"[verify] calib {len(cal)}, heldout {len(ho)}, ids disjoint; heldout captions also in calib: "
        f"{sum(r['caption'] in caps_c for r in ho)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=list(DOMAINS))
    ap.add_argument("--serial", action="store_true")
    a = ap.parse_args()
    if a.domain:
        resource.setrlimit(resource.RLIMIT_DATA, (MEM_CAP, MEM_CAP))
        build_domain(a.domain)
        sys.stdout.flush()
        os._exit(0)   # datasets streaming threads can crash in interpreter finalisation (T26)
    orchestrate(parallel=not a.serial)


if __name__ == "__main__":
    main()
