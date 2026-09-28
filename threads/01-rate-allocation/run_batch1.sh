source /home/coder/git/glm52/artifacts/shared-bit-graphs/runtime/env.sh
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 PYTHONPATH=/home/coder/git/orbit-duet
P=/home/coder/git/glm52/.venv/bin/python; T=/tmp/nestquant/01-rate-allocation
$P quant_exp.py --rates 2,4 --eval --schemes uniform,blk_fine,blk_fine_tp,ashard_uniform,ashard_blk_fine_tp --tfile $T/tcv_glm_l16_e36_{proj}.npy --out results/glm_l16_e36_tcv.json > $T/l16e36_tcv.log 2>&1
for dmp in 0.1 0.3 1.0; do
$P quant_exp.py --rates 2,4 --eval --damp $dmp --schemes uniform,blk_fine --out results/glm_l16_e36_damp$dmp.json > $T/l16e36_damp$dmp.log 2>&1
done
$P quant_exp.py --model mimo --layer 55 --expert 70 --rates 2,3,4 --schemes uniform,blk_fine_tp,ashard_uniform,ashard_blk_fine_tp,ashard_blk_int_tp,ashard_blk_half_tp --out results/mimo_l55_e70_tp.json > $T/mimo_tp.log 2>&1
