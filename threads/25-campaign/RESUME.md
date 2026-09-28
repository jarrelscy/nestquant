# T25 campaign: resume after box death or a /tmp wipe

Run one command. It is idempotent, so rerun it whenever you like:

    /home/coder/git/nestquant/threads/25-campaign/resume.sh --check   # report only, exit 0 when everything is in place
    /home/coder/git/nestquant/threads/25-campaign/resume.sh           # fix what is missing, then (re)start the driver

`resume.sh` does three things:
- It sets `LD_LIBRARY_PATH` to the T06 lib and the cuda-13.0 compat dir, and puts the glm52 venv on `PATH`.
- It creates `/tmp/nestquant/12-reference-encoder/bin/ninja` pointing to the venv's ninja.
- It runs `nq25_resume.py` with the definition in `campaign.json` in this directory (B3).

It has been tested from a fresh shell (`env -i HOME=$HOME PATH=/usr/bin:/bin bash -lc '.../resume.sh --check'`).

## What it checks and fixes, in order

| step | check | fix | estimate |
|---|---|---|---|
| env | venv python, LD_LIBRARY_PATH dirs, ninja link | recreate the link | - |
| code | `/home/coder/git/nestquant` contains pinned commit (`code.nestquant_commit`); every byte-defining file sha16 == `code.files` | none; the code comes from GitHub (`git clone/pull jarrelscy/nestquant`, lead's creds) | - |
| token | HF token present (HF_TOKEN or ~/.cache/huggingface/token, never printed) | none | - |
| fp8 | `zai-org/GLM-5.3` @ `aca966e4…` in `/tmp/nestquant/src/glm53-fp8`, sizes + LFS sha256 (cached in `.nq25_verified.json`) | `snapshot_download` of the missing files; sha256 all | 755.6 GB @ ~570 MB/s = ~22 min; sha256 ~13 min |
| stats | text `19-capture-glmfmt/stats/L3..77` and vision `19-capture-mm/stats/L3..77` | `threads/19-full-capture/fb_restore.py --set full` from flashblade | text ~3.2 TB @ ~290 MB/s = ~3 h; vision smaller |
| fixed | `fixed_set.json` present (it is a T19 global file, so the stats restore brings it back). It only feeds the manifests' `default_allocation`, and the driver refreshes and re-uploads manifests when the file changes | none | - |
| vision | graft files (vision_tower, mm_projector, tokenizer) sha256 == pins @ `68ebfe28…` | `hf_hub_download` at the pinned revision | ~1 GB, ~1 min |
| driver | running? (`ROOT/driver.pid`) | start `nq25_campaign.py run` detached (only when `status` = `launched`) | - |

## What "resume" means for the layers (B1: HF is the source of truth)

- **On start:** with `--upload go`, the driver lists `layers/` on `jarrelscy/GLM-5.3-NestQuant-2-4bit`. A layer counts as done when both hold:
  - Its remote `manifest.json` parses, and every file it lists is on the Hub with the same bytes and LFS sha256.
  - Its `campaign.config_id` equals ours.
  - Any other local "uploaded" claim is dropped.
- **Unfinished layers:** these resume from whatever local files exist:
  - E{E}.pt files that are present are kept.
  - Missing chunks are re-encoded.
  - A layer that is encoded but not finalized gets finalized.
- **Wiped ROOT:** if `ROOT/campaign.json` is gone, the driver is started with the full frozen config from `campaign.json`. The config_id then equals the uploaded layers' config_id because:
  - the paths and settings are the same;
  - the stats are restored byte-identical (sha256-verified).
- **fixed_set:** it is not part of config_id.
- **`--accept-code`:** only used after the code step verified that the file hashes equal the pinned ones.
- **Local layer dirs:** these stay in `/tmp` until their HF upload verifies (decision 7).

## Launch

The gates are per layer. The driver can be started early, and each layer waits on its own gate (`layer_gate` in `nq25_campaign.py`). Layer L encodes once all of these hold:
- **Text stats final:** T19's final text stats for L have every plan.json shard merged.
- **Text stats backed up:** T19's `done_full` backup marker lists that exact version (`logs/fb_backup_state_full*.json`).
- **Vision stats merged:** T26's vision stats for L are merged.
- **T23 gate** (t23b layers only): `T23_GO` exists next to this file. Commit it when T23's gate passes. If the gate fails, set `encoder` to `nq_layer`.

Layers that pass their gate are uploaded straight away (tp shards plus manifest). When T19's blended `fixed_set.json` lands, the driver refreshes each manifest's `default_allocation` and re-uploads only `manifest.json`.

Steps:
1. Set `status: "launched"` in `campaign.json`.
2. Run `nq25_resume.py pin`. It records the code file hashes, code_id/t23_id/t25_id and config_id.
3. Commit `campaign.json`.
4. Run `resume.sh --check`, then `resume.sh`.

## Watching the campaign

- `nq25_campaign.py status --out /tmp/nestquant/nq-encode-v1`
- `ROOT/campaign.log`
- `ROOT/ALERTS.jsonl` records:
  - spot flags
  - refcheck mismatch (the driver falls back to nq_layer for that layer)
  - check-decode failure
  - upload failure
  - code drift (new launches are held)
- `nq25_campaign.py stop --out ROOT` sends SIGTERM to the driver only. Workers keep running and the next run adopts them.

## Rebuilding the top-level files

- `nq25_nonexpert.py` writes the FP8 passthrough `nonexpert-*.safetensors`, the index and `nonexpert_manifest.json`.
- `nq25_configs.py` writes `config.json` and `config.mm.json`.
- Output goes to `/tmp/nestquant/25-campaign/repo-top`.
- Upload with `nq25_upload.py --root /tmp/nestquant/25-campaign --top /tmp/nestquant/25-campaign/repo-top --go`. It sends only files whose remote sha differs.
