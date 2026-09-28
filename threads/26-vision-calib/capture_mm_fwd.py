"""T26 stage 1 for the vision capture: T19/T24's capture_fwd_fast.py run UNCHANGED on the mm group corpus
(mm_corpus.py), with the 256 <|image|> embeddings of every sample replaced by the projected 5.2V vision features
(vis_feats.py) at the embedding layer.  Everything downstream of the embeddings (FP8 GLM-5.3 reference forward,
document-local attention, routing, acts format) is T19's code.

Two monkeypatches, applied before capture_fwd_fast.main():
  * capture_fwd_fast.Corpus.tok(w) records the global window index it returns;
  * nq19.Src.embed(tokens) splices the features of that window's images into the returned embeddings.
Only the initial state (first_layer == 0) calls embed on corpus windows; the matched docs are embedded without a
splice and --no-matched is enforced so they are never forwarded.

    run.sh-style env; CLI = capture_fwd_fast.py's (use --corpus <mm corpus> --shard-id k --no-matched ...)
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
T19 = os.path.join(os.path.dirname(HERE), "19-full-capture")
sys.path.insert(0, T19)
import nq19                     # noqa: E402
import capture_fwd_fast as cff  # noqa: E402

_cur = {"w": None}
_feat = {}


def _corpus_arg():
    return sys.argv[sys.argv.index("--corpus") + 1]


def _load_feats(corpus):
    if not _feat:
        imgs = json.load(open(f"{corpus}/images.json"))
        meta = json.load(open(f"{corpus}/feats.json"))
        n = meta["shape"][0]
        mm = np.memmap(f"{corpus}/feats.bf16", dtype=np.int16, mode="r", shape=(n, 256, nq19.D))
        byw = {}
        for i, r in enumerate(imgs[:n]):
            byw.setdefault(r["window"], []).append((r["pos"], i))
        _feat.update(mm=mm, byw=byw, n=n, used=0)
    return _feat


class MMCorpus(cff.Corpus):
    def tok(self, w):
        _cur["w"] = self.windows[w]
        return super().tok(w)


_orig_embed = nq19.Src.embed


def mm_embed(self, tokens):
    e = _orig_embed(self, tokens)
    w, _cur["w"] = _cur["w"], None
    if w is None:          # the matched docs (embedded unconditionally by capture_fwd_fast; unused with --no-matched)
        return e
    f = _load_feats(_corpus_arg())
    toks = tokens.view(-1)
    for pos, i in f["byw"].get(w, []):
        assert bool((toks[pos:pos + 256] == 154854).all()), (w, pos)
        v = torch.from_numpy(np.array(f["mm"][i])).view(torch.bfloat16).to(e.device)
        e[0, pos:pos + 256] = v
        f["used"] += 1
    return e


if __name__ == "__main__":
    if "--no-matched" not in sys.argv:
        raise SystemExit("capture_mm_fwd.py: pass --no-matched")
    cff.Corpus = MMCorpus
    nq19.Src.embed = mm_embed
    corpus = _corpus_arg()
    out = sys.argv[sys.argv.index("--out") + 1]
    os.makedirs(out, exist_ok=True)
    fm = json.load(open(f"{corpus}/feats.json"))
    json.dump(dict(role="T26 vision capture stage 1 (capture_fwd_fast.py + image-feature splice)", corpus=corpus,
                   feats_sha256=fm["sha256"], vision_tower_sha256=fm["vision_tower_sha256"],
                   mm_projector_sha256=fm["mm_projector_sha256"], argv=sys.argv[1:]),
              open(f"{out}/mm_protocol.json", "w"), indent=1)
    cff.main()
    if _feat:
        print(json.dumps(dict(images_spliced=_feat["used"])), flush=True)
