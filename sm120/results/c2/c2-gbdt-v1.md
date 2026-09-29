# C2 FP8-reference eval c2-gbdt-v1

| stream | corpus | KLD mean | +-se | p50 | p90 | p99 | top-1 | ppl ref | ppl |
|---|---|---|---|---|---|---|---|---|---|
| dyn | id | 0.21950 | 0.05525 | 0.01885 | 0.2737 | 4.6619 | 86.93% | 6.9616 | 6.9978 |
| dyn | wikitext | 0.07444 | 0.00497 | 0.01251 | 0.1676 | 1.0233 | 91.42% | 2.9775 | 3.0832 |
| dyn | code | 0.04255 | 0.00250 | 0.00969 | 0.1143 | 0.3754 | 92.53% | 3.8913 | 3.9170 |
| dyn_reset | id | 0.21947 | 0.05461 | 0.01978 | 0.2858 | 4.5047 | 86.74% | 6.9616 | 6.9100 |
| dyn_reset | wikitext | 0.08702 | 0.00566 | 0.01432 | 0.2008 | 1.1980 | 90.90% | 2.9775 | 3.1038 |
| dyn_reset | code | 0.04678 | 0.00232 | 0.01094 | 0.1239 | 0.4149 | 92.57% | 3.8913 | 3.9201 |
| dyn_gbdt | id | 0.22105 | 0.05564 | 0.01929 | 0.2801 | 4.6444 | 86.83% | 6.9616 | 7.0199 |
| dyn_gbdt | wikitext | 0.07575 | 0.00523 | 0.01309 | 0.1714 | 1.0287 | 91.41% | 2.9775 | 3.0817 |
| dyn_gbdt | code | 0.04454 | 0.00265 | 0.00997 | 0.1198 | 0.3982 | 92.50% | 3.8913 | 3.9140 |
| dyn_gbdt_reset | id | 0.22256 | 0.05534 | 0.02018 | 0.2881 | 4.7318 | 86.40% | 6.9616 | 7.0250 |
| dyn_gbdt_reset | wikitext | 0.08865 | 0.00559 | 0.01450 | 0.2024 | 1.2156 | 90.69% | 2.9775 | 3.1102 |
| dyn_gbdt_reset | code | 0.04841 | 0.00254 | 0.01122 | 0.1268 | 0.4251 | 92.34% | 3.8913 | 3.9188 |

Level-4 share per corpus (mean over NQ layers; ARVQ / AQLM = hot NVFP4): routed slots / gate-weighted

| stream | id | wikitext | code |
|---|---|---|---|
| dyn | 0.599 / 0.635 | 0.678 / 0.715 | 0.564 / 0.611 |
| dyn_reset | 0.581 / 0.617 | 0.641 / 0.679 | 0.531 / 0.577 |
| dyn_gbdt | 0.585 / 0.623 | 0.664 / 0.703 | 0.542 / 0.590 |
| dyn_gbdt_reset | 0.566 / 0.605 | 0.624 / 0.664 | 0.508 / 0.555 |

| stream | bits/expert resident | bits/expert served (routed-slot weighted) | level-4 share of routed slots |
|---|---|---|---|
| dyn | 2.818 | 3.488 | 0.610 |
| dyn_reset | 2.818 | 3.430 | 0.583 |
| dyn_gbdt | 2.818 | 3.453 | 0.594 |
| dyn_gbdt_reset | 2.818 | 3.393 | 0.566 |

Scheduler traffic (sum over NQ layers; bytes = rec_bytes x tp per upgrade, GB/s at 111 tok/s)

| stream | order | upgrades | MB/token | GB/s @111 | deferred steps | big steps |
|---|---|---|---|---|---|---|
| dyn | corpus | 442726 | 34.588 | 3.839 | 255745 | 0 |
| dyn_reset | per-window reset | 558098 | 43.601 | 4.840 | 1403771 | 0 |
| dyn_gbdt | corpus | 688692 | 53.804 | 5.972 | 3035078 | 0 |
| dyn_gbdt_reset | per-window reset | 685802 | 53.578 | 5.947 | 2945468 | 0 |
