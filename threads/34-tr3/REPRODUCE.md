# Reproducing the BF16-teacher KLD numbers (jF77 / jF128 / serve, fp8 KV)

These are the numbers in the HF README KLD table: full-vocabulary KL(BF16 teacher || arm) on confirmation windows
0000-0003 of brandonmusic/GLM-5.3-BF16-full-logits, with the serve's fp8_ds_mla KV cache emulated (`NQ_KVQ=kvq`).

## Inputs

| input | where from | path used below |
|---|---|---|
| NestQuant layers + serve manifest | HF `jarrelscy/GLM-5.3-NestQuant-2-4bit` (`layers/*`, `serving/`) | `$NQ_HF` |
| jF predictor | same repo, `serving/predictor/joint/jF.pt` (weights identical to the `jF_all.pt` used in our runs) | `$JF` |
| FP8 reference model (backbone, attention, dense layers, and the `fp8` arm) | `zai-org/GLM-5.3-FP8` | `$NQ_FP8` |
| BF16 teacher logits + token panel | `brandonmusic/GLM-5.3-BF16-full-logits`, `reference-full-panel` (revision 427368f1) | `$TEACHER` |

## Steps

1. Corpus (token ids + window map, 64 x 2048) from the teacher's panel:

       python threads/34-tr3/build_corpus.py $TEACHER $CORPORA

2. k0 manifest (no fixed set; floating_default = fixed-26 + floating-51 of the published manifest):

       python threads/34-tr3/mk_k0_manifest.py $NQ_HF/serving/tp4/manifest.json $K0

3. Pre-decode both levels to fp16 (about 10 min on 8 GPUs; byte-identical to the slow path):

       threads/18-e2e-eval/run.sh predecode-nq --fast --root $NQ_HF/layers --levels 2,4 --out $PD

   This writes `$PD/nq2` and `$PD/nq4`.

4. Run (one process per GPU; `WORLD=4` contiguous shards, one window each):

       export NQ_FP8 NQ_SHARD=contig NQ_CORPUS_DIR=$CORPORA NQ_KVQ=kvq
       export NQ_TEACHER=bf16conf=$TEACHER/reference-full-panel/logits/confirmation
       AD=adapt:lo=$PD/nq2,hi=$PD/nq4,chain=map,salstat=1,predictor=gbdt,gmode=sync,joint=$JF,manifest=$K0,hm=0.7
       for r in 0 1 2 3; do
         CUDA_VISIBLE_DEVICES=$r RANK=$r WORLD=4 threads/18-e2e-eval/run.sh run --corpora bf16conf --max-windows 4 \
           --moe-chunk 16384 --tag rep --cand jF77=$AD,n_float=77 --cand jF128=$AD,n_float=128 &
       done; wait
       threads/18-e2e-eval/run.sh merge --tag rep

   Any hot count works the same way (e.g. `n_float=121`). The FP8 reference is scored as arm `fp8`.

`run_kvq4.sh` is the exact script behind the published table (it also runs `serve`, `jF173` and `nq4`, and a
KV-cache-only `NQ_KVQ=kv` group). `jF173` uses a kB manifest whose order past the first 77 experts comes from
calibration salience; it is not derivable from the published files, so only n_float <= 128 rows reproduce exactly from
public inputs (larger n_float still runs from k0, filling the extra slots from the predictor).

## Expected (4 windows, mean KLD)

| arm | resident bpw | KLD |
|---|---|---|
| fp8 | 8 | 0.01179 |
| jF128 | 3.08 | 0.02150 |
| jF77 | 2.66 | 0.02607 |
| serve (GBDT, 26 fixed + 51 floating) | 2.66 | 0.02831 |

## Caveats

- `run.sh` hard-codes this box's venv (`/home/coder/git/glm52/.venv`), CUDA env and a lightgbm pylib path; edit those
  three lines for another machine.
- The harness is an offline teacher-forced evaluator: it swaps 4-bit experts in and out per 16-token block the way the
  server would, but it is not a serving engine. No server runs jF yet (see threads/33-search/joint/SERVE.md).
