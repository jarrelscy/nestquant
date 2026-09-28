#!/bin/bash
# waits for each layer manifest, then runs the TP4 whole-layer smoke on it
cd /data/Jarrel/nestquant
for L in 3 4 5 6 7 8 9 10; do
  while [ ! -f /rawdata/Jarrel/nq-glm53-prod/L$L/manifest.json ]; do sleep 60; done
  sleep 30
  CUDA_VISIBLE_DEVICES=3 /data/Jarrel/nqenv/bin/python sm120/smoke_layer.py /rawdata/Jarrel/nq-glm53-prod $L 4 0.3 > sm120/results/smoke_layer_prod_L$L.log 2>&1
  echo "L$L exit $? $(tail -n1 sm120/results/smoke_layer_prod_L$L.log)"
done
