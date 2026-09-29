# C2 FP8-reference eval c2-gbdt24-aqlm-v1

| stream | corpus | KLD mean | +-se | p50 | p90 | p99 | top-1 | ppl ref | ppl |
|---|---|---|---|---|---|---|---|---|---|
| dyn | id | 0.21950 | 0.05525 | 0.01885 | 0.2737 | 4.6619 | 86.93% | 6.9616 | 6.9978 |
| dyn | wikitext | 0.07444 | 0.00497 | 0.01251 | 0.1676 | 1.0233 | 91.42% | 2.9775 | 3.0832 |
| dyn | code | 0.04255 | 0.00250 | 0.00969 | 0.1143 | 0.3754 | 92.53% | 3.8913 | 3.9170 |
| dyn_gbdt | id | 0.21358 | 0.05399 | 0.01817 | 0.2615 | 4.6398 | 87.08% | 6.9616 | 7.0074 |
| dyn_gbdt | wikitext | 0.06962 | 0.00411 | 0.01214 | 0.1590 | 0.9362 | 91.67% | 2.9775 | 3.0712 |
| dyn_gbdt | code | 0.04182 | 0.00244 | 0.00954 | 0.1119 | 0.3737 | 92.85% | 3.8913 | 3.9053 |
| aqlm | id | 0.32280 | 0.06374 | 0.05662 | 0.5879 | 5.2954 | 81.60% | 6.9616 | 6.8897 |
| aqlm | wikitext | 0.36022 | 0.02627 | 0.09233 | 0.9498 | 3.9549 | 80.01% | 2.9775 | 3.8281 |
| aqlm | code | 0.06888 | 0.00310 | 0.01823 | 0.1844 | 0.5754 | 90.74% | 3.8913 | 3.9410 |

Level-4 share per corpus (mean over NQ layers; ARVQ / AQLM = hot NVFP4): routed slots / gate-weighted

| stream | id | wikitext | code |
|---|---|---|---|
| dyn | 0.599 / 0.635 | 0.678 / 0.715 | 0.564 / 0.611 |
| dyn_gbdt | 0.630 / 0.666 | 0.688 / 0.725 | 0.580 / 0.625 |
| aqlm | 0.353 / 0.396 | 0.274 / 0.288 | 0.406 / 0.485 |

| stream | bits/expert resident | bits/expert served (routed-slot weighted) | level-4 share of routed slots |
|---|---|---|---|
| dyn | 2.818 | 3.488 | 0.610 |
| dyn_gbdt | 2.818 | 3.536 | 0.632 |
| aqlm | 2.753 | 2.870 | 0.346 |

Scheduler traffic (sum over NQ layers; bytes = rec_bytes x tp per upgrade, GB/s at 111 tok/s)

| stream | order | cap GB/s | upgrades | MB/token | GB/s @111 | deferred steps | big steps |
|---|---|---|---|---|---|---|---|
| dyn | corpus | 6 | 442726 | 34.588 | 3.839 | 255745 | 0 |
| dyn_gbdt | corpus | 24 | 1295514 | 101.212 | 11.235 | 11724 | 0 |
