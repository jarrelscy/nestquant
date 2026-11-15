#!/usr/bin/env python3
"""Validate /tmp/nq_inkling/corpus_mm (CPU only).

Checks
  1. shapes/dtypes; split.json covers [0, W) with fit first; 1200 fit docs / 200 val docs; every sample id once.
  2. per window: segments.npy matches windows.jsonl segment table (contiguous, offsets, lengths), pad = token 199999 /
     segment -1 only in the tail; fit windows hold only calib samples, val windows only heldout samples.
  3. per window: the runs of image token 200054 in tokens.npy are EXACTLY the saved slots (same start/end, same order,
     no stray placeholders); each run has length == slot n_tokens == prod(grid_thw) == 256; run is preceded by
     <|message_user|><|content_image|> and followed by <|end_message|>; the slot lies inside its own segment.
  4. per slot: pix file exists; pixel_values fp32 [256, 40, 40, 3] in [-1, 1]; image_grid_thw == [[1, 16, 16]] and
     prod == run length; the pixels agree with images/<domain>/<id>.jpg up to JPEG error.
  5. rowkind: kind 1 exactly on <|content_image|> + run + closing <|end_message|>, 0 exactly on pad.
  6. processor parity (--n-proc samples, default 24): the real Inkling2Processor on (jpg, chat-template text) gives
     input_ids == the window segment and image_grid_thw == slot grid.
  7. cap_ink.py Data(...) loads every window (the capture's own reader).
"""
import argparse, json, os, sys
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = "/tmp/nq_inkling/corpus_mm"
MODEL = "/tmp/nq_inkling/model"
IMG_TOK, PAD, USER, IMAGE_CT, END_MESSAGE = 200054, 199999, 200000, 200005, 200010
C, N_VIS = 2048, 256


def runs(row, tok):
    m = np.concatenate([[False], row == tok, [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--n-proc", type=int, default=24)
    ap.add_argument("--skip-jpg", action="store_true", help="skip the pix-vs-jpg agreement check")
    a = ap.parse_args()
    import torch
    from PIL import Image
    R = a.root
    T = np.load(f"{R}/tokens.npy"); S = np.load(f"{R}/segments.npy"); K = np.load(f"{R}/rowkind.npy")
    W = T.shape[0]
    assert T.shape == S.shape == K.shape == (W, C) and T.dtype == S.dtype == np.int32 and K.dtype == np.int8
    for k in ("bnd_think_d", "bnd_end_d"):
        b = np.load(f"{R}/{k}.npy"); assert b.shape == (W, C) and b.dtype == np.int8
    sp = json.load(open(f"{R}/split.json"))
    assert sp["fit"][0] == 0 and sp["fit"][1] == sp["val"][0] and sp["val"][1] == W
    win = [json.loads(l) for l in open(f"{R}/windows.jsonl")]
    assert len(win) == W
    samples = {json.loads(l)["id"]: json.loads(l) for f in ("samples.jsonl", "heldout.jsonl")
               for l in open(f"{R}/samples/{f}")}
    images = json.load(open(f"{R}/images.json"))
    seen, nslots, ndocs = set(), 0, {"fit": 0, "val": 0}
    pix_dev = []
    for w in range(W):
        split = "fit" if w < sp["fit"][1] else "val"
        segs, slots = win[w]["segments"], win[w]["images"]
        p = 0
        for sg in segs:
            o, n = sg["window_offset"], sg["tokens"]
            assert o == p, (w, "non-contiguous")
            assert (S[w, o:o + n] == sg["segment_id"]).all() and sg["segment_id"] >= 0
            assert sg["split"] == split
            s = samples[sg["source_id"]]
            assert (s["split"] == "calib") == (split == "fit")
            assert sg["source_id"] not in seen; seen.add(sg["source_id"])
            ndocs[split] += 1
            p += n
        assert (S[w, p:] == -1).all() and (T[w, p:] == PAD).all() and (K[w, p:] == 0).all() and (K[w, :p] > 0).all()
        rr = runs(T[w], IMG_TOK)
        assert len(rr) == len(slots) == len(segs), (w, len(rr), len(slots), len(segs))
        for (r0, r1), sl, sg in zip(rr, slots, segs):
            assert (sl["start"], sl["end"]) == (int(r0), int(r1)), (w, sl["id"], r0, r1)
            g = sl["grid_thw"]
            assert r1 - r0 == sl["n_tokens"] == g[0] * g[1] * g[2] == N_VIS
            assert sl["id"] == sg["source_id"] and sl["segment_id"] == sg["segment_id"]
            o = sg["window_offset"]
            assert o <= r0 and r1 <= o + sg["tokens"]
            assert T[w, r0 - 2] == USER and T[w, r0 - 1] == IMAGE_CT and T[w, r1] == END_MESSAGE
            kk = np.full(sg["tokens"], 2, np.int8); kk[r0 - 1 - o:r1 + 1 - o] = 1
            assert (K[w, o:o + sg["tokens"]] == kk).all(), (w, sl["id"], "rowkind")
            d = torch.load(f"{R}/{sl['pix_file']}", weights_only=True)
            pv, gt = d["pixel_values"], d["image_grid_thw"]
            assert pv.dtype == torch.float32 and tuple(pv.shape) == (N_VIS, 40, 40, 3)
            assert gt.tolist() == [g] and int(gt.prod()) == r1 - r0
            assert pv.min() >= -1 and pv.max() <= 1
            if not a.skip_jpg:
                x = torch.from_numpy(np.array(Image.open(f"{R}/{sl['image_file']}").convert("RGB"))).float() / 127.5 - 1
                x = x.reshape(16, 40, 16, 40, 3).permute(0, 2, 1, 3, 4).reshape(N_VIS, 40, 40, 3)
                pix_dev.append(float((x - pv).abs().mean()))
            nslots += 1
    assert seen == set(samples) and nslots == len(images) == 1400
    assert ndocs == {"fit": sp["fit_docs"], "val": sp["val_docs"]} == {"fit": 1200, "val": 200}, ndocs
    for im in images:
        assert win[im["window"]]["images"][[s["id"] for s in win[im["window"]]["images"]].index(im["id"])]["start"] == im["start"]
    assert int((T == IMG_TOK).sum()) == 1400 * N_VIS
    print(f"[ok] {W} windows ({sp['fit'][1]} fit / {W - sp['fit'][1]} val), {nslots} image slots, all runs == slots, "
          f"rowkind/segments/pad consistent")
    if pix_dev:
        print(f"[ok] pix vs jpg mean |diff| (JPEG q90 error): mean {np.mean(pix_dev):.4f}, max {np.max(pix_dev):.4f}")
        assert np.max(pix_dev) < 0.1   # mean |diff| on the [-1,1] scale; q90 JPEG on dense text reaches ~0.05

    # 6. processor parity
    if a.n_proc:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(MODEL)
        rng = np.random.default_rng(0)
        pick = rng.choice(len(images), size=min(a.n_proc, len(images)), replace=False)
        for i in pick:
            im = images[i]; s = samples[im["id"]]
            sg = next(x for x in win[im["window"]]["segments"] if x["source_id"] == im["id"])
            prompt = "Read the text in the image." if s["domain"] == "ocr" else "Describe the image."
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
                    {"role": "assistant", "content": s["caption"]}]
            txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, reasoning_effort=0)
            out = proc(images=[Image.open(f"{R}/{s['image_file']}")], text=[txt], return_tensors="pt")
            ref = out["input_ids"][0].numpy()
            o = sg["window_offset"]
            assert np.array_equal(ref, T[im["window"], o:o + sg["tokens"]]), im["id"]
            assert out["image_grid_thw"].tolist() == [im["grid_thw"]]
            assert tuple(out["pixel_values"].shape) == (N_VIS, 40, 40, 3)
        print(f"[ok] Inkling2Processor parity on {len(pick)} samples (input_ids + grid identical)")

    # 7. capture reader
    sys.path.insert(0, "/tmp/nq_inkling/code")
    import cap_ink
    wins = [("c2048_mm", R, w, "fit" if w < sp["fit"][1] else "val") for w in range(W)]
    d = cap_ink.Data(wins, 0, 1)
    nv = int(d.fit.sum() + d.val.sum())
    assert nv == int((S >= 0).sum()) and len(d.segs) == 1400
    print(f"[ok] cap_ink.Data loads all {W} windows: {len(d.segs)} segments, {nv} valid rows "
          f"(fit {int(d.fit.sum())}, val {int(d.val.sum())})")
    print("VALID")


if __name__ == "__main__":
    main()
