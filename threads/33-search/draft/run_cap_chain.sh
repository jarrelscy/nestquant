#!/bin/bash
# MTP chain redo (layers resumed from hfinal): step j>1 fed snorm(MTP output); 8 ranks.
H=/home/coder/git/nestquant/threads/33-search/draft
DRAFT_CHAIN=normed DRAFT_SKIP1=1 TAG=chain bash $H/run_cap.sh
