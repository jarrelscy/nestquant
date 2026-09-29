# Thread 28: serving release (records in the layout the SM120 streaming server reads)

NestQuant releases ship pre-packed per TP degree, so consumers never run `streaming/repack.py`. HF repo:
`jarrelscy/GLM-5.3-NestQuant-2-4bit` under `serving/tp4/` (TP4 only, per the user; the tools take `--tp` for TP2/TP8).

## Files

| file | what |
|---|---|
| `nq_release.py ROOT OUT --tp N [--layers ..]` | nestquant-v1 layers -> `OUT/serving/tpN/`, per (layer, rank), idempotent, content-hashed; rebuilds index/manifest/COMPLETE |
| `nq_upload.py OUT --tp N [--go]` | uploads changed layers in order (one verified commit per layer, with a partial index), then full index + manifest + `serving/nq_assemble.py`, then `COMPLETE` last; refits delete the Hub's COMPLETE first |
| `nq_check.py DIR [--ref ROOT] [--dev cuda]` | validator (structure, offsets, hashes, bit-exact decode of a few experts per layer) |
| `nq_assemble.py DIR [--move]` | stdlib-only: per-layer blocks -> `rank{r}.bin` (pure byte copy + sha256), incremental; also published as `serving/nq_assemble.py` |
| `nq_assemble.py --src DL --into RECDIR [--layers ..]` | stdlib-only: install a (refit) download into an existing record dir (the serve's `NQ_REPACK_DIR`, repack.py-built or not) in place; see below |
| `nq_verify_repack.py --release DL --repack RECDIR` | stdlib-only, read-only: sha256 of every (layer, rank) records range + resident file of a record dir vs the release; published as `serving/nq_verify_repack.py` |
| `nq_refit_hook.py ROOT --layers L.. [--upload]` | the refit hook: release -> check -> upload for every built TP degree |
| `neg_test.py SRC_TPDIR REF WORK [L]` | validator negative tests on a one-layer copy |

Big outputs: `/tmp/nestquant/28-serve-release/out/serving/tp4/` (local build), logs next to it.

## Layout: `serving/tp{N}/` (format nq-serve-v1)

- `rank{r}/L{L}.bin`: the NE=256 P4 records of layer L on rank r, in expert order. These are exactly the bytes at offset
  `(L-L0)*NE*rec_bytes` of the assembled `rank{r}.bin`, where L0 = 3.
- Record layout **nq-p4rec-v1**, from `streaming/p4rec.py`:
  - Segments in order: `gu.p4 | gu.d4 | dn.p4 | dn.d4 | lr4`, where lr4 = U4_g, U4_u, U4_d in fp16 at RMAX = 4.
  - Each segment is 256 B aligned, and the record is rounded to 4 KiB.
  - At TP4 (H 6144, I 512), rec_bytes = 2,560,000 = 625 x 4 KiB. Segment offsets: gu.p4 0 (1,572,864), gu.d4 1,572,864 (12,288), dn.p4 1,585,152 (909,316), dn.d4 2,494,720 (6,144), lr4 2,500,864 (57,344).
  - The layout is fixed for the format. Any change must bump the name (`nq-p4rec-v2`) in `nq_release.REC_FORMAT` and in the model card. `check_layout` asserts the invariants.
- `res/rank{r}/L{L}.pt`: resident planes, **nq-res-v1** (`streaming/resident.save`).
- `layers/L{L}.json`: the per-layer block. It holds:
  - size, sha256 and offset of every rank file
  - layout
  - `default_allocation`: the fixed level-4 set, never truncated, cross-checked against the fixed_set.json sha in the layer manifest
  - `floating_default` and `n_routed` (and `vision_n_routed`) from that fixed_set.json
  - the source manifest sha
  - `layer_hash` = sha256 of the canonical json without `layer_hash`/`time`
- `rank{r}.json`: a superset of repack.py's index. `stream_engine.RankFile` and `nq_vllm.available()` read it unchanged. It holds:
  - format, rec_bytes, seg, L0, NE, `layers{L: experts, rg, rd}`
  - per layer: `file`, `offset`, `bytes`, `sha256`, `res`, `res_bytes`, `res_sha256`, `layer_hash`
  - `bin` and `bin_bytes` of the assembled file
  - per layer `source_manifest_sha256` and, only where the layer manifest's config has them, the rotation fields
    `in_had_down` (absent = 128), `had_sign_seed`, `ics_down` (T30; the serve reads `in_had_down` from here)
- `manifest.json`: formats, layout, `layer_hash` per layer, `default_allocation`, `floating_default`, `n_routed`, fixed-set sha,
  rotation maps, `artifact_key`.
- `artifact_stamp.json` (full release only): `{"artifact", "key", "layers": {L: manifest sha256}}`, key = `sm120/eval/run_c2.sh`'s
  KEY (`nq_release.artifact_key`: sha256 of `"L{L} <sha256 of layers/L{L}/manifest.json>\n"` in `sort -V` order, first 16 hex),
  so an assembled release passes run_c2's stamp check. No time field (deterministic).
- `COMPLETE`: `{layers: {L: layer_hash}, manifest_sha256, index_sha256, stamp_sha256}`. It is uploaded last, and the release is complete only when it exists.

Checked equal to the reference converter: for L3 at TP4 the records and resident files are byte-identical (`cmp`) to
`streaming/repack.py` output. The resident .pt has to be staged under the same basename, because torch.save's zip prefix is the file name.

## Note for the SM120 agent

- Download `serving/tp4/` (plus `serving/nq_assemble.py`). Wait until `serving/tp4/COMPLETE` exists.
- Run `python nq_assemble.py serving/tp4 [--move]`. It pwrites every `rank{r}/L{L}.bin` into `rank{r}.bin` at
  `rank{r}.json layers[L].offset` and checks each block's sha256 while copying. `--move` deletes each block after copying, which halves the disk.
- After that, `serving/tp4/` is an `NQ_REPACK` dir as it stands: `rank{r}.bin` + `rank{r}.json` + `res/rank{r}/L{L}.pt`.
- After a refit, re-download and re-run it. Layers already in place (`rank{r}.assembled.json`, per-layer sha) are skipped, so only changed layers are rewritten.
- Optional change on your side: read the per-layer files directly and skip assembly. Each layer's records are
  contiguous in `rank{r}/L{L}.bin` with the same rec_bytes, so record (L, E) is at `E*rec_bytes` of the file
  `rank{r}.json layers[L].file`. That needs `stream_engine.RankFile` / the nqstream `Engine` to take a per-layer fd table
  instead of one path. It is your code, so I did not edit it. The index already carries both addresses.
- Validate a download with `nq_check.py serving/tp4 --dev cuda`. The reference is `../../layers/` of the same download.

## Drop-in record dir (T30)

`serving/tp4/` is a drop-in `NQ_REPACK_DIR`: `rank{r}/L{L}.bin` == the bytes `streaming/repack.py` (HEAD 750e317) writes at
`((L-3)*256)*rec_bytes` of `rank{r}.bin`, and `res/rank{r}/L{L}.pt` == its resident file, checked by sha256 on all 4 ranks for L10
(v1) and L3-6 (h512 refit), repack.py run on the HF-layout safetensors. `rank{r}.json` is a superset of repack.py's index; every
reader (stream_engine.RankFile, nq_vllm.available, serve_nq.sh, eval/nqeff.py, eval/eval_fp8.py, smoke_stream.py, repack.py's
own skip/merge) only uses repack.py's keys and ignores the rest.

`nq_assemble.py --src DL --into RECDIR`: per (layer, rank) current (same layer_hash + shas) -> skip; entry without/with another
layer_hash -> hash the region + resident file, adopt if equal, else install; absent -> install. Install = drop L from
rank{r}.json, pwrite + fsync + read-back sha, resident tmp + sha + read-back + rename, put the release entry back (all index
writes tmp+rename under rank{r}.lock, the lock repack.py uses). Stamp -> "updating" during, recomputed at the end (== the
release's stamp when RECDIR holds exactly the release). Planning (incl. every needed block present) happens before any write;
`--dry-run` prints it. Tested on a sparse repack.py-built dir (old L3-4 + L10): adopt L10, install L3-4, kill -9 mid-install
+ rerun, bad-block negative test (layer left out of the index, rerun repairs), rerun no-op, `--verify-existing`, holes of
other layers untouched, nq_vllm.available / RankFile / resident.load read the result.

## Validator (nq_check.py)

The checks:
1. Structure:
   - COMPLETE covers L0..L1 and its hashes match the layer blocks, manifest and index.
   - The layout invariants hold, and the layout equals the one implied by the reference manifest.
   - Per rank and layer: offset, bytes and file size are right, and index, layer block and manifest agree.
   - The fixed set equals the reference manifest's `level4_experts`. floating_default is disjoint from it, and n_routed has 256 entries.
2. sha256 of every block and resident file, threaded.
3. Decode check on 3 experts per layer: one fixed-set expert, one with lr rank 0, one random. It runs on rank L%N (`--ranks all` for every rank). For each expert:
   - (a) the record bytes equal `p4rec.pack` of the reference expert built from the nestquant-v1 shards
   - (b) the resident planes are bit-equal to the reference
   - (c) an independent decode of the **published** planes matches the threads/12 reference decoder bitwise, at levels 2 and 4. The published planes are the resident base/var plus the record P4, unpacked from the sub-array layout, plus the d4 block words, decoded with `moe.dense_W`. The U4 segment and resident lr also equal the reference lrU4 and lrV|lrU2.

Runs on CPU or `--dev cuda` (capped at 12 GB). One layer on 4 ranks with 3 experts takes 28 s on an A100 and 68 s on CPU.
The full TP4 release (75 layers, 600 files, 392.8 GB) passes: hashing takes 50 s from page cache (32 threads), and 225 decode checks take 259 s on one A100, for 311 s in total.

`nq_assemble.py` was tested on rank 0 of the full build. The L3 region is `cmp`-identical to `repack.py` output, the L40 and L77 regions match their shas, and a rerun is a no-op (0 s).

Negative tests (`neg_test.py`, L3): all 12 cases behave as expected.
- A clean copy passes.
- Each of the following fails with the expected message:
  - a flipped gu.p4 byte (sha, record bytes, level-4 decode)
  - a flipped dn.d4 byte (sha, record bytes, level-4 down decode)
  - a flipped byte in an unsampled expert (sha)
  - a wrong index offset
  - a wrong rec_bytes
  - a truncated block
  - an edited resident base plane re-saved (size, (b), level-2 decode)
  - a flipped resident byte with the size kept (sha)
  - an edited fixed set, and an edited fixed set with the layer_hash recomputed (hash chain / reference mismatch)
- The documented blind spot: an unsampled flipped byte with `--hash none` passes.

## Refit hook

Run this after a layer is re-finalized, whether by the T27 PV-tuning post-pass, a re-encode, or a changed fixed set:

    python threads/28-serve-release/nq_refit_hook.py ROOT --layers 17,40-42 --upload

ROOT is the nestquant-v1 tree holding the finalized layers, all L3-77 (e.g. `/tmp/nestquant/nq-encode-v1` after the refit is finalized in place).

1. It re-exports only the (layer, rank)s whose source manifest changed. The fingerprint is the manifest sha256 plus the byte-determining code plus the formats.
2. It rewrites the layer blocks, index, manifest and local COMPLETE.
3. It checks the refit layers on all ranks.
4. It uploads only the files whose hash changed. The Hub's COMPLETE is deleted first and written last.

A new fixed set (a new fixed_set.json sha in the layer manifests) changes only `layers/L{L}.json`, the manifest and the index, not the records. Publish it in the same run as the refit, so records and allocation stay consistent.

Publish the refit's `layers/L{L}/` safetensors in the same session (threads/25 `nq25_upload.py`). The pinned encoder code (threads/12, threads/23) is not touched.
