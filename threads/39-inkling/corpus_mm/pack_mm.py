#!/usr/bin/env python3
"""Inkling-2-Small multimodal calibration corpus (step 2 of 2): pack corpus_mm/samples/*.jsonl into 2048-token windows.

Mirrors nestquant T26 mm_corpus.py, in Inkling's native chat format and the Inkling text-corpus schema
(/tmp/nq_inkling/corpus/c2048, read by cap_ink.py Data/windows()/flags()).

Each sample = ONE attention segment, rendered with the model's chat_template.jinja (add_generation_prompt=False,
reasoning_effort=0 because the assistant turn has no reasoning, same rule as the text corpus) for
    [user: [image, text=prompt], assistant: caption]
which gives
    <|message_system|><|content_text|>Thinking effort level: 0<|end_message|>
    <|message_user|><|content_image|> 256 x <|unused_200054|> <|end_message|>
    <|message_user|><|content_text|>{prompt}<|end_message|>
    <|message_model|><|content_text|>{caption}<|end_message|><|content_model_end_sampling|>
The single template placeholder is expanded to grid_thw.prod() = 256 copies of 200054, exactly what
Inkling2Processor.replace_image_token does (validate_mm.py re-checks this against the real processor).
prompt = "Read the text in the image." (ocr) / "Describe the image." (others), as T26.
Caption guard as T26: caption body capped at C//4 = 512 tokens; a capped caption ends mid-message (no
<|end_message|>/EOS), the text corpus's convention for cut fragments.
Samples (shuffled with np.random.default_rng(42), as T26) are first-fit packed into 2048-token windows; the tail is
padding: token 199999 (<|endoftext|>), segment -1 (Inkling text-corpus convention; cap_ink treats seg < 0 as invalid).

Outputs (OUT=/tmp/nq_inkling/corpus_mm):
  tokens.npy, segments.npy [W,2048] int32 (segment id = doc_index = global sample ordinal, fit first; pad = -1)
  rowkind.npy [W,2048] int8: 0 pad, 1 image (<|content_image|>, 256 placeholders, the image's <|end_message|>), 2 text
  bnd_think_d.npy, bnd_end_d.npy [W,2048] int8 zeros (T26: no boundary rows in the mm corpus; files exist for flags())
  windows.jsonl: {"segments": [...], "images": [slot...]} per window; slot = {id, domain, segment_id, start, end
     (window token offsets, end exclusive, covering only the 200054 run), n_tokens, image_token_id, grid_thw,
     pix_file, image_file} (paths relative to OUT)
  split.json {fit: [0, nfit], val: [nfit, W], fit_docs, val_docs}; images.json (flat list of all slots + window);
  manifest.json
"""
import hashlib, json, os, sys
import numpy as np

OUT = os.environ.get("INK_MM_OUT", "/tmp/nq_inkling/corpus_mm")
MODEL = "/tmp/nq_inkling/model"
C = 2048
SEED = 42
N_VIS = 256
CAP_MAX = C // 4
PROMPTS = {"ocr": "Read the text in the image."}
PROMPT_DEFAULT = "Describe the image."
PAD = 199999
USER, MODEL_T, SYSTEM = 200000, 200001, 200002
TEXT, IMAGE_CT, END_SAMPLING, END_MESSAGE = 200004, 200005, 200006, 200010
IMG_TOK = 200054


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    tmpl = open(f"{MODEL}/chat_template.jinja").read()
    assert tok.convert_tokens_to_ids("<|unused_200054|>") == IMG_TOK
    assert tok.convert_tokens_to_ids("<|endoftext|>") == PAD

    def render(prompt, caption):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
                {"role": "assistant", "content": caption}]
        return tok.apply_chat_template(msgs, chat_template=tmpl, tokenize=False, add_generation_prompt=False,
                                       reasoning_effort=0)

    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    stats = dict(capped_captions=0)

    def piece(s):
        prompt = PROMPTS.get(s["domain"], PROMPT_DEFAULT)
        ids = enc(render(prompt, s["caption"]))
        assert ids.count(IMG_TOK) == 1
        i = ids.index(IMG_TOK)
        assert ids[i - 2:i] == [USER, IMAGE_CT] and ids[i + 1] == END_MESSAGE
        ids = ids[:i] + [IMG_TOK] * N_VIS + ids[i + 1:]
        # caption span: after the last <|message_model|><|content_text|>, before the closing <|end_message|><EOS>
        m = max(k for k in range(len(ids) - 1) if ids[k] == MODEL_T and ids[k + 1] == TEXT)
        assert ids[-2:] == [END_MESSAGE, END_SAMPLING]
        cap = ids[m + 2:-2]
        assert tok.decode(cap) == s["caption"], s["id"]
        if len(cap) > CAP_MAX:
            stats["capped_captions"] += 1
            ids = ids[:m + 2] + cap[:CAP_MAX]
            ncap = CAP_MAX
        else:
            ncap = len(cap)
        kind = [2] * len(ids)
        kind[i - 1] = 1                       # <|content_image|>
        for k in range(i, i + N_VIS + 1):     # placeholders + the image message's <|end_message|>
            kind[k] = 1
        assert len(ids) <= C
        return dict(s=s, ids=ids, kind=kind, img_pos=i, n_caption=ncap)

    def pieces(path):
        samples = [json.loads(l) for l in open(path)]
        rng = np.random.default_rng(SEED)
        rng.shuffle(samples)
        return [piece(s) for s in samples]

    def pack(items):                          # first fit in shuffled order (T26)
        wins = []
        for it in items:
            n = len(it["ids"])
            for w in wins:
                if w[0] >= n:
                    w[1].append(it); w[0] -= n; break
            else:
                wins.append([C - n, [it]])
        return [w[1] for w in wins]

    fit = pack(pieces(f"{OUT}/samples/samples.jsonl"))
    val = pack(pieces(f"{OUT}/samples/heldout.jsonl"))
    allw = fit + val
    W = len(allw)
    T = np.full((W, C), PAD, np.int32)
    S = np.full((W, C), -1, np.int32)
    K = np.zeros((W, C), np.int8)
    images = []
    g = 0
    with open(f"{OUT}/windows.jsonl", "w") as wf:
        for w, items in enumerate(allw):
            p, segs, slots = 0, [], []
            for it in items:
                n = len(it["ids"]); s = it["s"]
                T[w, p:p + n] = it["ids"]; S[w, p:p + n] = g; K[w, p:p + n] = it["kind"]
                a = p + it["img_pos"]
                slot = dict(id=s["id"], domain=s["domain"], segment_id=g, start=a, end=a + N_VIS, n_tokens=N_VIS,
                            image_token_id=IMG_TOK, grid_thw=s["grid_thw"], pix_file=s["pix_file"],
                            image_file=s["image_file"])
                slots.append(slot)
                images.append(dict(window=w, **slot))
                segs.append(dict(source_id=s["id"], category=f"mm_{s['domain']}", kind="mm", group="c2048_mm",
                                 doc_index=g, segment_id=g, token_offset=0, window_offset=p, tokens=n,
                                 image_tokens=N_VIS, caption_tokens=it["n_caption"],
                                 split="fit" if s["split"] == "calib" else "val", sample_split=s["split"]))
                p += n; g += 1
            wf.write(json.dumps(dict(segments=segs, images=slots)) + "\n")
    nfit = len(fit)
    for k, arr in (("tokens", T), ("segments", S), ("rowkind", K),
                   ("bnd_think_d", np.zeros((W, C), np.int8)), ("bnd_end_d", np.zeros((W, C), np.int8))):
        np.save(f"{OUT}/{k}.npy", arr)
    fit_docs = sum(map(len, fit)); val_docs = sum(map(len, val))
    json.dump(dict(fit=[0, nfit], val=[nfit, W], fit_docs=fit_docs, val_docs=val_docs), open(f"{OUT}/split.json", "w"), indent=1)
    json.dump(images, open(f"{OUT}/images.json", "w"))
    kf, kv = K[:nfit], K[nfit:]
    rows = lambda k: dict(image=int((k == 1).sum()), text=int((k == 2).sum()), pad=int((k == 0).sum()))
    man = dict(schema="nestquant-inkling2-mm-v1", group="c2048_mm", model="thinkingmachines/Inkling-2-Small",
               context=C, windows=W, fit_windows=nfit, val_windows=W - nfit, documents=g, fit_docs=fit_docs,
               val_docs=val_docs, fit_rows=rows(kf), val_rows=rows(kv),
               image_placeholder_rows=dict(fit=int((T[:nfit] == IMG_TOK).sum()), val=int((T[nfit:] == IMG_TOK).sum())),
               capped_captions=stats["capped_captions"], image_token_id=IMG_TOK, image_tokens_per_image=N_VIS,
               image_size=[640, 640], grid_thw=[1, 16, 16], pixel_values_shape=[N_VIS, 40, 40, 3],
               pad_token_id=PAD, eos_token_id=END_SAMPLING, seed=SEED,
               prompts=dict(ocr=PROMPTS["ocr"], default=PROMPT_DEFAULT),
               template=("<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|>"
                         "<|content_image|>256x<|unused_200054|><|end_message|><|message_user|><|content_text|>{prompt}"
                         "<|end_message|><|message_model|><|content_text|>{caption}<|end_message|>"
                         "<|content_model_end_sampling|>"),
               samples_sha256=sha(f"{OUT}/samples/samples.jsonl"), heldout_sha256=sha(f"{OUT}/samples/heldout.jsonl"),
               tokenizer_sha256=sha(f"{MODEL}/tokenizer.json"), chat_template_sha256=sha(f"{MODEL}/chat_template.jinja"),
               code_sha256={f: sha(os.path.join(os.path.dirname(os.path.abspath(__file__)), f))
                            for f in ("build_mm.py", "pack_mm.py")})
    for k in ("tokens", "segments", "rowkind", "bnd_think_d", "bnd_end_d"):
        man[f"{k}_sha256"] = sha(f"{OUT}/{k}.npy")
    man["windows_sha256"] = sha(f"{OUT}/windows.jsonl"); man["split_sha256"] = sha(f"{OUT}/split.json")
    man["images_sha256"] = sha(f"{OUT}/images.json")
    json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=1)
    print(json.dumps({k: man[k] for k in ("windows", "fit_windows", "val_windows", "documents", "fit_docs", "val_docs",
                                          "fit_rows", "val_rows", "image_placeholder_rows", "capped_captions")}, indent=1))


if __name__ == "__main__":
    main()
