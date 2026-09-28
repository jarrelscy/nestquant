"""T26: turn /tmp/nestquant/calib-mm (image+caption samples) into a thread-21-layout group corpus that T19's stage 1
(capture_fwd_fast.py, via capture_mm_fwd.py) can run unchanged.

Every sample is ONE attention segment (document-local causal attention, positions restart per segment, exactly like
the text corpora) in GLM-native chat form (the grafted checkpoint's chat_template.jinja, non-thinking turn):

    [gMASK]<sop><|user|><|begin_of_image|> 256 x <|image|> <|end_of_image|>{prompt}<|assistant|><think></think>{caption}

prompt = "Read the text in the image." (ocr) / "Describe the image." (others), as in glm52 capture_mm53.py.  The 256
<|image|> embeddings are overwritten at capture time with the projected 5.2V vision features (vis_feats.py).
Samples (shuffled, seed 42) are first-fit packed into 2048-token windows; the unused tail of a window is one PAD
segment (<|endoftext|>) whose rows are dropped before the statistics (rowkind 0).

Output dir (default /tmp/nestquant/19-capture-mm/corpus/c2048_mm):
  tokens.npy, segments.npy   [W, 2048] int32   (segment id = global sample ordinal; PAD tail = -1 - window)
  rowkind.npy                [W, 2048] int8    0 pad, 1 image token (incl. begin/end-of-image), 2 text (template+caption)
  bnd_think_d.npy, bnd_end_d.npy  zeros        (no boundary rows in the mm corpus)
  split.json {fit: calib windows, val: heldout windows}, windows.jsonl (T21 schema + image slots), images.json,
  manifest.json (sha256s)
"""
import argparse
import hashlib
import json
import os

import numpy as np

MM = "/tmp/nestquant/calib-mm"
TOKJ = "/tmp/nestquant/src/glm53-fp8/tokenizer.json"
C = 2048
N_VIS = 256
SEED = 42
PROMPTS = {"ocr": "Read the text in the image."}
PROMPT_DEFAULT = "Describe the image."


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mm", default=MM)
    ap.add_argument("--out", default="/tmp/nestquant/19-capture-mm/corpus/c2048_mm")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKJ)
    tid = lambda s: tok.token_to_id(s)
    sp = {k: tid(k) for k in ("[gMASK]", "<sop>", "<|user|>", "<|begin_of_image|>", "<|image|>", "<|end_of_image|>",
                              "<|assistant|>", "<think>", "</think>", "<|endoftext|>")}
    assert sp["<|image|>"] == 154854 and sp["<|begin_of_image|>"] == 154830 and sp["<|end_of_image|>"] == 154831, sp
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids
    os.makedirs(a.out, exist_ok=True)

    def pieces(split_file):
        samples = [json.loads(l) for l in open(split_file)]
        rng = np.random.default_rng(SEED)
        rng.shuffle(samples)
        out = []
        for s in samples:
            head = [sp["[gMASK]"], sp["<sop>"], sp["<|user|>"], sp["<|begin_of_image|>"]]
            mid = [sp["<|end_of_image|>"]] + enc(PROMPTS.get(s["domain"], PROMPT_DEFAULT)) + \
                  [sp["<|assistant|>"], sp["<think>"], sp["</think>"]]
            cap = enc(s["caption"])[: C // 4]                       # runaway-caption guard (as capture_mm53)
            ids = head + [sp["<|image|>"]] * N_VIS + mid + cap
            kind = [2] * 3 + [1] + [1] * N_VIS + [1] + [2] * (len(mid) - 1 + len(cap))
            out.append(dict(s=s, ids=ids, kind=kind, img_pos=len(head), n_caption=len(cap)))
        return out

    def pack(items):
        """First fit (in shuffled order) into C-token windows."""
        wins = []                                                  # [free, [items]]
        for it in items:
            n = len(it["ids"])
            assert n <= C
            for w in wins:
                if w[0] >= n:
                    w[1].append(it); w[0] -= n; break
            else:
                wins.append([C - n, [it]])
        return [w[1] for w in wins]

    fit = pack(pieces(f"{a.mm}/samples.jsonl"))
    val = pack(pieces(f"{a.mm}/heldout/samples.jsonl"))
    allw = fit + val
    W = len(allw)
    T = np.full((W, C), sp["<|endoftext|>"], np.int32)
    S = np.zeros((W, C), np.int32)
    K = np.zeros((W, C), np.int8)
    images, wj = [], []
    g = 0
    for w, items in enumerate(allw):
        p, segs = 0, []
        for it in items:
            n = len(it["ids"])
            T[w, p:p + n] = it["ids"]; S[w, p:p + n] = g; K[w, p:p + n] = it["kind"]
            s = it["s"]
            images.append(dict(window=w, pos=p + it["img_pos"], image_file=s["image_file"], id=s["id"],
                               tensor_file=s["tensor_file"], tensor_key=s["tensor_key"]))
            segs.append(dict(source_id=s["id"], category=f"mm_{s['domain']}", kind="mm", group="c2048_mm",
                             doc_index=g, segment_id=g, token_offset=0, window_offset=p, tokens=n,
                             image_tokens=N_VIS, caption_tokens=it["n_caption"], split=s.get("split")))
            p += n; g += 1
        if p < C:
            S[w, p:] = -1 - w
            segs.append(dict(source_id="pad", category="pad", kind="pad", group="c2048_mm", doc_index=-1 - w,
                             segment_id=-1 - w, token_offset=0, window_offset=p, tokens=C - p))
        wj.append(dict(segments=segs))
    np.save(f"{a.out}/tokens.npy", T); np.save(f"{a.out}/segments.npy", S); np.save(f"{a.out}/rowkind.npy", K)
    np.save(f"{a.out}/bnd_think_d.npy", np.zeros((W, C), np.int8)); np.save(f"{a.out}/bnd_end_d.npy", np.zeros((W, C), np.int8))
    with open(f"{a.out}/windows.jsonl", "w") as f:
        for r in wj:
            f.write(json.dumps(r) + "\n")
    json.dump(dict(fit=[0, len(fit)], val=[len(fit), W], fit_docs=sum(map(len, fit)), val_docs=sum(map(len, val))),
              open(f"{a.out}/split.json", "w"), indent=1)
    json.dump(images, open(f"{a.out}/images.json", "w"))
    nf = len(fit) * C
    kf, kv = K[:len(fit)], K[len(fit):]
    man = dict(schema="nestquant-26-mm-v1", group="c2048_mm", context=C, windows=W, fit_windows=len(fit),
               val_windows=len(val), fit_tokens=nf, documents=g,
               fit_rows=dict(image=int((kf == 1).sum()), text=int((kf == 2).sum()), pad=int((kf == 0).sum())),
               val_rows=dict(image=int((kv == 1).sum()), text=int((kv == 2).sum()), pad=int((kv == 0).sum())),
               template="[gMASK]<sop><|user|><|begin_of_image|>256x<|image|><|end_of_image|>{prompt}<|assistant|><think></think>{caption}",
               prompts=dict(ocr=PROMPTS["ocr"], default=PROMPT_DEFAULT), seed=SEED, source=a.mm,
               samples_sha256=sha(f"{a.mm}/samples.jsonl"), heldout_sha256=sha(f"{a.mm}/heldout/samples.jsonl"),
               tokenizer_sha256=sha(TOKJ))
    for k in ("tokens", "segments", "rowkind", "bnd_think_d", "bnd_end_d"):
        man[f"{k}_sha256"] = sha(f"{a.out}/{k}.npy")
    man["windows_sha256"] = sha(f"{a.out}/windows.jsonl"); man["split_sha256"] = sha(f"{a.out}/split.json")
    json.dump(man, open(f"{a.out}/manifest.json", "w"), indent=1)
    print(json.dumps({k: man[k] for k in ("windows", "fit_windows", "val_windows", "fit_tokens", "documents", "fit_rows", "val_rows")}))


if __name__ == "__main__":
    main()
