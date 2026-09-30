#!/bin/bash
cd /home/coder/git/nestquant/threads/33-search/decide
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib
P="nice -n 10 /home/coder/git/glm52/.venv/bin/python"
$P train_heads.py h64 1 tweedie
$P train_heads.py h16 0 tweedie
$P train_heads.py h256 3 tweedie
$P train_heads.py h128 2 tweedie
for c in calib-fit glm52-heldout; do $P predict_heads.py $c h16 h64 h128 h256; done
$P arms_blend.py a > /tmp/nestquant/33-search/decide/arms_blend_a.txt 2>&1
$P train_heads.py q64_90 1 quantile 0.9 100
$P train_heads.py q64_70 1 quantile 0.7 100
for c in calib-fit glm52-heldout; do $P predict_heads.py $c q64_90 q64_70; done
echo DONE
