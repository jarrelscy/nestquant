#!/bin/bash
cd /home/coder/git/nestquant/threads/33-search/decide
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib
P="nice -n 10 /home/coder/git/glm52/.venv/bin/python"
$P train_heads.py h64 1 tweedie
$P train_heads.py h16 0 tweedie
$P train_heads.py h256 3 tweedie
$P train_heads.py h128 2 tweedie
$P train_heads.py q64_90 1 quantile 0.9 100
$P train_heads.py q64_70 1 quantile 0.7 100
