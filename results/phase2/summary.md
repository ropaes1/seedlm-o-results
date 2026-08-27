# Phase 2 results: run phase2-20260815-210326

- config: C=8 P=3 stage1_seeds=16384 stage2_seeds=65535 generator_seed=3407
- torch 2.11.0+cu128 | transformers 5.15.0 | device cuda
- layers compressed: 28 | target tensors: 196 | prompts sha256 0fd801b8d2b690a0…
- originals sha256 f47f71177f32bcd1… (verified: True)

## bits/weight decomposition (compressed tensors only)

| variant | base bpw | + side bits / numel | = total bpw | side bits | whole-model MB @bf16 rest |
|---|---|---|---|---|---|
| `seed_only` | 4.00 | 0.0000 | **4.0000** | 0 | 531.5 |
| `mag_p0.1` | 4.00 | 0.0480 | **4.0480** | 21,134,400 | 534.1 |
| `mag_p0.5` | 4.00 | 0.2400 | **4.2400** | 105,689,472 | 544.7 |
| `mag_p1.0` | 4.00 | 0.4800 | **4.4800** | 211,388,352 | 557.9 |
| `mag_p2.0` | 4.00 | 0.9600 | **4.9600** | 422,782,080 | 584.3 |
| `awq_p0.1` | 4.00 | 0.0480 | **4.0480** | 21,134,400 | 534.1 |
| `awq_p0.5` | 4.00 | 0.2400 | **4.2400** | 105,689,472 | 544.7 |
| `awq_p1.0` | 4.00 | 0.4800 | **4.4800** | 211,388,352 | 557.9 |
| `awq_p2.0` | 4.00 | 0.9600 | **4.9600** | 422,782,080 | 584.3 |
| `spike_p0.1` | 4.00 | 0.0458 | **4.0458** | 20,191,360 | 534.0 |
| `spike_p0.5` | 4.00 | 0.2459 | **4.2459** | 108,297,728 | 545.0 |
| `spike_p1.0` | 4.00 | 0.4751 | **4.4751** | 209,253,184 | 557.7 |
| `spike_p2.0` | 4.00 | 0.9607 | **4.9607** | 423,095,680 | 584.4 |
| `rtn4` | 4.50 | 0.0000 | **4.5000** | 0 | 559.0 |
| `rtn4_mag_p1.0` | 4.50 | 0.4800 | **4.9800** | 211,388,352 | 585.4 |
| `bf16` | 16.00 | 0.0000 | **16.0000** | 0 | 1192.1 |
| `seed_only@65535` | 4.00 | 0.0000 | **4.0000** | 0 | 531.5 |
| `awq_p1.0@65535` | 4.00 | 0.4800 | **4.4800** | 211,388,352 | 557.9 |
| `rtn4@65535` | 4.50 | 0.0000 | **4.5000** | 0 | 559.0 |
| `bf16@65535` | 16.00 | 0.0000 | **16.0000** | 0 | 1192.1 |

## behavioural damage (teacher-forced vs bf16 reference)

| variant | bpw | mean KL | p95 KL | max KL | top-1 agree | mean rel_err | fingerprint | fit s | eval s |
|---|---|---|---|---|---|---|---|---|---|
| `seed_only` | 4.0000 | 0.611849 | 2.786063 | 12.0954 | 78.03% | 0.1546 | `7c518e58cc2f43a5` | 1938 | 23 |
| `mag_p0.1` | 4.0480 | 0.556129 | 2.326664 | 11.1492 | 78.18% | 0.1519 | `0851ec1ce217fc2d` | 13 | 26 |
| `mag_p0.5` | 4.2400 | 0.506748 | 2.216768 | 12.1466 | 79.48% | 0.1465 | `9e1ad5d69ce543dd` | 65 | 22 |
| `mag_p1.0` | 4.4800 | 0.460370 | 2.018469 | 10.3462 | 80.49% | 0.1418 | `8043b5f18c81e445` | 130 | 23 |
| `mag_p2.0` | 4.9600 | 0.398403 | 1.758407 | 10.2697 | 81.50% | 0.1345 | `f3d7cecae2ed815e` | 251 | 22 |
| `awq_p0.1` | 4.0480 | 0.378891 | 1.658188 | 10.3694 | 82.51% | 0.1538 | `9cc37ff9dbb044c0` | 13 | 21 |
| `awq_p0.5` | 4.2400 | 0.305756 | 1.221972 | 10.9558 | 85.12% | 0.1513 | `14d0c83636f71820` | 61 | 15 |
| `awq_p1.0` | 4.4800 | 0.264871 | 1.021771 | 13.1738 | 86.27% | 0.1482 | `1520b17a18b010c4` | 120 | 17 |
| `awq_p2.0` | 4.9600 | 0.214646 | 0.772406 | 13.6358 | 87.86% | 0.1428 | `62e2ba9382277404` | 229 | 19 |
| `spike_p0.1` | 4.0458 | 0.418167 | 1.712125 | 7.9407 | 80.49% | 0.1530 | `51cb22c2e03af00f` | 1915 | 13 |
| `spike_p0.5` | 4.2459 | 0.445417 | 2.142709 | 15.4959 | 82.37% | 0.1495 | `fbf3863769ed2731` | 1917 | 20 |
| `spike_p1.0` | 4.4751 | 0.487027 | 2.090087 | 15.2135 | 82.80% | 0.1462 | `ecf7a28e004943da` | 1917 | 17 |
| `spike_p2.0` | 4.9607 | 0.497958 | 2.662918 | 7.2041 | 79.91% | 0.1399 | `9c2178f873ee7d48` | 1917 | 21 |
| `rtn4` | 4.5000 | 0.170426 | 0.684151 | 6.9979 | 87.57% | 0.0994 | `b9d7cde45a00cf4e` | 0 | 18 |
| `rtn4_mag_p1.0` | 4.9800 | 0.154498 | 0.654510 | 5.0744 | 87.43% | 0.0864 | `0291c414b555345b` | 0 | 18 |
| `bf16` | 16.0000 | 0.000000 | 0.000000 | 0.0000 | 100.00% | 0.0000 | `16998bd6cbab04e7` | 0 | 17 |
| `seed_only@65535` | 4.0000 | 0.302056 | 1.481540 | 7.1888 | 85.55% | 0.1253 | `0bd6adc5b00b3b80` | 8716 | 26 |
| `awq_p1.0@65535` | 4.4800 | 0.184999 | 0.743187 | 12.0007 | 87.57% | 0.1201 | `36be36ff7eaa5a60` | 540 | 21 |
| `rtn4@65535` | 4.5000 | 0.170426 | 0.684151 | 6.9979 | 87.57% | 0.0994 | `b9d7cde45a00cf4e` | 0 | 23 |
| `bf16@65535` | 16.0000 | 0.000000 | 0.000000 | 0.0000 | 100.00% | 0.0000 | `16998bd6cbab04e7` | 0 | 22 |

## spike variants: achieved rank per family (budget-matched, achieved not target)

| variant | achieved bpw | r per family |
|---|---|---|
| `spike_p0.1` | 4.0458 | down_proj:[2], gate_proj:[2], k_proj:[2], o_proj:[2], q_proj:[2], up_proj:[2], v_proj:[2] |
| `spike_p0.5` | 4.2459 | down_proj:[12], gate_proj:[12], k_proj:[8], o_proj:[10], q_proj:[10], up_proj:[12], v_proj:[8] |
| `spike_p1.0` | 4.4751 | down_proj:[23], gate_proj:[23], k_proj:[15], o_proj:[20], q_proj:[20], up_proj:[23], v_proj:[15] |
| `spike_p2.0` | 4.9607 | down_proj:[46], gate_proj:[46], k_proj:[31], o_proj:[41], q_proj:[41], up_proj:[46], v_proj:[31] |

## controls

- `bf16` (mean_kl == 0.0 and fingerprint == reference): **PASS**: mean_kl=0.0, fp=16998bd6cbab04e7, matches_reference=True
- `rtn4` (mean_kl > 0): **PASS**: mean_kl=0.17042645812034607, fp=b9d7cde45a00cf4e, matches_reference=False

## gate verdict (Phase 2)

**STRONG**: best salience variant `awq_p1.0@65535` @ 4.4800 bpw (budget-matched: bpw <= rtn4's 4.5000)

- KL seed_only = 0.302056 @ 4.0000 bpw
- KL best      = 0.184999 @ 4.4800 bpw
- KL rtn4      = 0.170426 @ 4.5000 bpw
- KL reduction vs seed_only = 38.75%  (STRONG needs >= 30%, WEAK needs >= 15%)
- gap closure to rtn4 = 88.93%  (STRONG needs >= 50%)
- rel_err reduction on 10 most outlier-heavy tensors = 7.57%  (WEAK needs >= 25%)
- all salience variants below 10% KL reduction: False

- verdict computed on: stage2 @65535 seeds

### side finding: salience rule ranking (best variant per rule)
_ranked on stage1 @16384 seeds (only the winner is refit at stage 2)_

| rank | rule | best variant | mean KL | KL reduction |
|---|---|---|---|---|
| 1 | **awq** | `awq_p2.0` | 0.214646 | 64.92% |
| 2 | **mag** | `mag_p2.0` | 0.398403 | 34.89% |
| 3 | **spike** | `spike_p0.1` | 0.418167 | 31.66% |

