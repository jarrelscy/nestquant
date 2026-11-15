#!/bin/bash
# A/B on layer 5: text-only H (enc_b175) vs 75/25 text/image blend (enc_b175mm); text val + image-caption val KLD.
until [ -f /tmp/nq_inkling/cap-mm/stats/L5/meta.json ]; do sleep 60; done
cd /tmp/nq_inkling/code
/home/coder/git/glm52/.venv/bin/python ink_campaign.py run --root /tmp/nq_inkling/enc_b175mm --cfg b175mm --order 5 --wpg 1 > /tmp/nq_inkling/logs/campaign_mm5.log 2>&1
( export LD_LIBRARY_PATH=/home/coder/git/nestquant/threads/06-expert-objective/lib:$LD_LIBRARY_PATH
  export PYTHONPATH=/home/coder/git/nestquant/threads/05-exl3-harness:/home/coder/git/orbit-duet
  export NQ35_BASE_K=1.75 TORCH_CUDA_ARCH_LIST=10.0 CUDA_VISIBLE_DEVICES=2
  mkdir -p /tmp/nq_inkling/pre_mm5
  /home/coder/git/glm52/.venv/bin/python predecode_ink.py /tmp/nq_inkling/enc_b175mm /tmp/nq_inkling/pre_mm5 5 )
ev() {  # name wdir wsel ref [mm]
  O=/tmp/nq_inkling/kld/$1; [ -f $O/final/kld.json ] && return; rm -rf $O
  env ${5:+INK_MM=/tmp/nq_inkling/corpus_mm} flock /tmp/nq_inkling/kld.lock /tmp/nq_inkling/venv/bin/torchrun --nproc-per-node 8 --master-port 29717 cap_ink.py \
    --out $O --layers 0-47 --nfit 0,0 --no-stats --wdir /tmp/nq_inkling/$2 --wsel $3 --wmax 6 --ref /tmp/nq_inkling/$4 > $O.log 2>&1
  echo "$1 $(grep 'KLD vs ref' $O.log)"
}
ev l5mm_2 pre_mm5 2 ref
ev l5mm_hot64 pre_mm5 hot:64 ref
ev v_l5b175_2 pre_l5 2 ref_mm mm
ev v_l5b175_hot64 pre_l5 hot:64 ref_mm mm
ev v_l5mm_2 pre_mm5 2 ref_mm mm
ev v_l5mm_hot64 pre_mm5 hot:64 ref_mm mm
/tmp/nq_inkling/venv/bin/python /tmp/nq_inkling/kld/tools/perkld.py l5b175_2 l5mm_2 l5b175_hot64 l5mm_hot64
echo AB_DONE
