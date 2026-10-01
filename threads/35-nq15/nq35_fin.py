"""T35 campaign finalize: T25 nq25_finalize.py (nq_layer finalize via --fin-cmd, safetensors convert + round trip,
artifact-from-safetensors == E{E}.pt raw bytes, L2/L4 decode compare) with the nq15 base-K decoder installed in this
process (the st check decodes in-process; the artifacts carry meta base_K, so no env is needed here)."""
import os, sys, runpy
HERE = os.path.dirname(os.path.abspath(__file__))
T12 = "/home/coder/git/nestquant/threads/12-reference-encoder"
T25 = "/home/coder/git/nestquant/threads/25-campaign"
for p in (T12, T25, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import nq15                      # noqa: E402,F401
import nq35_t29 as T29           # noqa: E402


def check29(root, L, src):
    """T29 layers (fin29 step 4 analogue): manifest fields present; the safetensors-assembled artifact (+ manifest
    in_had_down) decodes == E.pt under the T29 decoder (every 16th expert, L2 + L4); and a sanity bound on one expert:
    the T29 L2 decode of down vs the FP8 teacher (rel err < 0.6; a Had128 decode of a Had512 down gives ~1.4)."""
    import json, torch
    import nq25_st as S, nq_io
    man = json.load(open(f"{root}/L{L}/manifest.json"))
    assert man["config"].get("in_had_down") == T29.WIDTH and "rotation" in man and man["campaign"].get("t29"), "manifest"
    parts = [S.load_shard(f"{root}/L{L}/tp{s}.safetensors") for s in range(S.NSH)]
    bad = 0
    for E in range(0, 256, 16):
        art = torch.load(f"{root}/L{L}/experts/E{E}.pt", weights_only=False, map_location="cpu")
        assert art["meta"].get(T29.FIELD) == T29.WIDTH, f"E{E} meta"
        sa = S.assemble(root, L, E, parts=parts); sa["meta"] = {T29.FIELD: int(man["config"]["in_had_down"])}
        for Lv in (2, 4):
            bad += not all(torch.equal(x, y) for x, y in zip(T29.decode_expert(art, Lv), T29.decode_expert(sa, Lv)))
    art = torch.load(f"{root}/L{L}/experts/E0.pt", weights_only=False, map_location="cpu")
    Wd = T29.decode_expert(art, 2)[2].cpu()
    ref = nq_io.FP8Model(src).expert(L, 0, "cpu")["down_proj"].float()
    rel = float((Wd - ref).norm() / ref.norm())
    print(f"[L{L}] T29 check: st decode29 == E.pt {'PASS' if not bad else f'FAIL {bad}'}; E0 down L2 rel {rel:.4f}",
          flush=True)
    return not bad and rel < 0.6


if __name__ == "__main__":
    argv = list(sys.argv)
    sys.argv[0] = f"{T25}/nq25_finalize.py"
    runpy.run_path(f"{T25}/nq25_finalize.py", run_name="__main__")
    L = int(argv[argv.index("--layer") + 1]); out = argv[argv.index("--out") + 1]
    if T29.is_t29(L) and not os.environ.get("NQ35_NO_T29"):
        sys.path.insert(0, "/home/coder/git/nestquant/threads/18-e2e-eval")
        if not check29(out, L, os.environ.get("NQ35_SRC", "/tmp/nestquant/src/glm53-fp8")):
            sys.exit(8)
