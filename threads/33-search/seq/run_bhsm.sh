#!/bin/bash
cd /home/coder/git/nestquant/threads/33-search/seq
export PYTHONPATH=/tmp/nestquant/18-e2e/pylib NQ_LAYOUT=k0
P="nice -n 10 /home/coder/git/glm52/.venv/bin/python binhist.py"
$P eval bh0 sm120tf; $P eval v2only1 sm120tf
