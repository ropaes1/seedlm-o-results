# Static layer-skip KL probe (depth as a bytes-touched lever)

_Generated 2026-08-22 06:37:49 +0000 from `results.jsonl` (41 measurements)._

* model: `Qwen3-1.7B` (28 decoder layers)
* protocol: 12 prompts, 731 teacher-forced positions, KL(bf16 || skipped) in nats -- replicated from [../code/swap_eval.py](../code/swap_eval.py)
* prompts_sha256: `0fd801b8d2b690a0...`
* stack: `8ebd2f131f8f|NVIDIA GeForce RTX 4090|torch2.11.0+cu128`

Reference lines on the same ruler (same model, same protocol, measured previously by the main harness -- shown for scale only, never merged into this table):

| reference | mean KL |
|---|---|
| q4km | 0.032 |
| rtn4 | 0.176 |
| q3km | 0.205 |
| best seeded (c12p4w_awq_p1.0) | 0.436 |

## Verdict

**INCONCLUSIVE so far.** Best non-trivial pattern `full[27]` at KL 0.1449, 1.0 layer-equivalents. Sweep may be incomplete.

## Headline curve -- depth removed vs KL

Best (lowest) KL achieved at each depth reduction, over every pattern measured. `layer-equiv` is bytes saved expressed in whole-layer units, so an attention-only skip counts as 0.25 of a layer.

```
l-equiv   bytes        KL  0                          2.04
   1.00    2.9%   0.1449  |##                                |  <= q3km  full[27]
   2.00    5.9%   0.5620  |#########                         |  full[24,25]
   3.00    8.8%   0.7665  |#############                     |  full[13,14,15]
   4.00   11.7%   1.0747  |##################                |  full[13,14,15,16]
   6.00   17.6%   1.9430  |################################  |  full[12,13,14,15,16,17]

    ref           0.0320  |#                                 | <- q4km
    ref           0.1760  |###                               | <- rtn4
    ref           0.2050  |###                               | <- q3km
    ref           0.4360  |#######                           | <- best seeded (c12p4w_awq_p1.0)
```

## Depth-sensitivity map (stage 1: skip each layer alone)

```
layer        KL   top1%
    0   14.1837    2.1%  ######################################
    1    1.9219   75.8%  #####
    2    4.4894   52.8%  ############
    3    0.1950   92.1%  #
    4    0.4674   88.1%  #
    5    0.2710   89.6%  #
    6    0.4176   87.3%  #
    7    0.2762   88.6%  #
    8    0.4562   86.0%  #
    9    0.4420   85.1%  #
   10    0.3586   87.7%  #
   11    0.3801   87.4%  #
   12    0.2546   90.8%  #
   13    0.1965   91.4%  #
   14    0.2305   90.0%  #
   15    0.1961   90.0%  #
   16    0.2292   89.3%  #
   17    0.3260   87.1%  #
   18    0.2467   89.2%  #
   19    0.3953   85.9%  #
   20    0.4473   88.0%  #
   21    0.4708   87.6%  #
   22    0.3152   89.7%  #
   23    0.2645   90.6%  #
   24    0.1526   90.6%  
   25    0.2194   90.6%  #
   26    0.1780   90.7%  
   27    0.1449   91.8%  
```

Least sensitive: L27 (0.145), L24 (0.153), L26 (0.178), L3 (0.195), L15 (0.196), L13 (0.197)

Most sensitive: L0 (14.184), L2 (4.489), L1 (1.922), L21 (0.471)

## All measurements (sorted by KL)

| pattern | stage | layers | layer-equiv | bytes/tok saved | footprint saved | mean KL | p95 KL | max KL | top-1 % | roofline x | wall s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `none` | 0-anchor | 0 | 0.00 | 0.00% | 0.00% | **0.0000** | 0.0000 | 0.000 | 100.00 | 1.000 | 0.5 |
| `full[27]` | 1-single | 1 | 1.00 | 2.93% | 2.93% | **0.1449** | 0.6858 | 3.503 | 91.79 | 1.030 | 0.4 |
| `full[24]` | 1-single | 1 | 1.00 | 2.93% | 2.93% | **0.1526** | 0.9739 | 7.726 | 90.56 | 1.030 | 0.4 |
| `full[26]` | 1-single | 1 | 1.00 | 2.93% | 2.93% | **0.1780** | 0.9597 | 5.646 | 90.70 | 1.030 | 0.4 |
| `full[3]` | 1-single | 1 | 1.00 | 2.93% | 2.93% | **0.1950** | 0.8934 | 12.766 | 92.07 | 1.030 | 0.4 |
| `full[15]` | 1-single | 1 | 1.00 | 2.93% | 2.93% | **0.1961** | 1.0862 | 7.803 | 90.01 | 1.030 | 0.4 |
| `full[13]` | 1-single | 1 | 1.00 | 2.93% | 2.93% | **0.1965** | 0.9884 | 13.874 | 91.38 | 1.030 | 0.4 |
| `full[25]` | 1-single | 1 | 1.00 | 2.93% | 2.93% | **0.2194** | 1.3115 | 6.834 | 90.56 | 1.030 | 0.4 |
| `full[16]` | 1-single | 1 | 1.00 | 2.93% | 2.93% | **0.2292** | 1.1871 | 13.550 | 89.33 | 1.030 | 0.4 |
| `full[14]` | 1-single | 1 | 1.00 | 2.93% | 2.93% | **0.2305** | 1.2424 | 11.687 | 90.01 | 1.030 | 0.4 |
| `full[18]` | 1-single | 1 | 1.00 | 2.93% | 2.93% | **0.2467** | 1.2219 | 15.204 | 89.19 | 1.030 | 0.4 |
| `full[12]` | 1-single | 1 | 1.00 | 2.93% | 2.93% | **0.2546** | 1.1501 | 20.065 | 90.83 | 1.030 | 0.4 |
| `full[23]` | 1-single | 1 | 1.00 | 2.93% | 2.93% | **0.2645** | 1.1788 | 12.251 | 90.56 | 1.030 | 0.4 |
| `full[5]` | 1-single | 1 | 1.00 | 2.93% | 2.93% | **0.2710** | 1.3667 | 20.443 | 89.60 | 1.030 | 0.4 |
| `full[7]` | 1-single | 1 | 1.00 | 2.93% | 2.93% | **0.2762** | 1.5265 | 18.750 | 88.65 | 1.030 | 0.4 |
| `full[22]` | 1-single | 1 | 1.00 | 2.93% | 2.93% | **0.3152** | 1.2698 | 21.250 | 89.74 | 1.030 | 0.4 |
| `full[17]` | 1-single | 1 | 1.00 | 2.93% | 2.93% | **0.3260** | 1.8456 | 14.000 | 87.14 | 1.030 | 0.4 |
| `full[10]` | 1-single | 1 | 1.00 | 2.93% | 2.93% | **0.3586** | 1.6303 | 19.999 | 87.69 | 1.030 | 0.4 |
| `full[11]` | 1-single | 1 | 1.00 | 2.93% | 2.93% | **0.3801** | 1.7546 | 21.547 | 87.41 | 1.030 | 0.4 |
| `full[19]` | 1-single | 1 | 1.00 | 2.93% | 2.93% | **0.3953** | 1.7895 | 18.875 | 85.91 | 1.030 | 0.4 |
| `full[6]` | 1-single | 1 | 1.00 | 2.93% | 2.93% | **0.4176** | 2.3125 | 15.000 | 87.28 | 1.030 | 0.4 |
| `full[9]` | 1-single | 1 | 1.00 | 2.93% | 2.93% | **0.4420** | 2.4198 | 15.125 | 85.09 | 1.030 | 0.4 |
| `full[20]` | 1-single | 1 | 1.00 | 2.93% | 2.93% | **0.4473** | 2.0343 | 23.750 | 87.96 | 1.030 | 0.4 |
| `full[8]` | 1-single | 1 | 1.00 | 2.93% | 2.93% | **0.4562** | 2.4283 | 22.459 | 86.05 | 1.030 | 0.4 |
| `full[4]` | 1-single | 1 | 1.00 | 2.93% | 2.93% | **0.4674** | 1.8681 | 27.375 | 88.10 | 1.030 | 0.4 |
| `full[21]` | 1-single | 1 | 1.00 | 2.93% | 2.93% | **0.4708** | 1.9933 | 22.000 | 87.55 | 1.030 | 0.4 |
| `full[24,25]` | 2-contig | 2 | 2.00 | 5.85% | 5.85% | **0.5620** | 3.4896 | 24.110 | 84.27 | 1.062 | 0.3 |
| `full[26,27]` | 2-contig | 2 | 2.00 | 5.85% | 5.85% | **0.5738** | 3.5928 | 17.400 | 84.13 | 1.062 | 0.4 |
| `full[13,14,15]` | 2-contig | 3 | 3.00 | 8.78% | 8.78% | **0.7665** | 4.0207 | 24.252 | 78.52 | 1.096 | 0.3 |
| `full[25,26]` | 2-contig | 2 | 2.00 | 5.85% | 5.85% | **1.0685** | 4.2335 | 14.262 | 78.11 | 1.062 | 0.3 |
| `full[13,14,15,16]` | 2-contig | 4 | 4.00 | 11.70% | 11.70% | **1.0747** | 4.7357 | 24.638 | 70.18 | 1.133 | 0.3 |
| `full[25,26,27]` | 2-contig | 3 | 3.00 | 8.78% | 8.78% | **1.4225** | 7.3947 | 25.760 | 74.15 | 1.096 | 0.3 |
| `full[1]` | 1-single | 1 | 1.00 | 2.93% | 2.93% | **1.9219** | 11.5176 | 34.937 | 75.79 | 1.030 | 0.4 |
| `full[12,13,14,15,16,17]` | 2-contig | 6 | 6.00 | 17.55% | 17.55% | **1.9430** | 7.4134 | 27.226 | 52.12 | 1.213 | 0.4 |
| `full[13,14,15,16,17,18]` | 2-contig | 6 | 6.00 | 17.55% | 17.55% | **2.3302** | 8.0906 | 20.701 | 46.92 | 1.213 | 0.3 |
| `full[24,25,26,27]` | 2-contig | 4 | 4.00 | 11.70% | 11.70% | **2.5664** | 12.8467 | 32.148 | 65.12 | 1.133 | 0.3 |
| `full[24,25,26]` | 2-contig | 3 | 3.00 | 8.78% | 8.78% | **3.2951** | 7.7813 | 21.012 | 62.24 | 1.096 | 0.3 |
| `full[22,23,24,25,26,27]` | 2-contig | 6 | 6.00 | 17.55% | 17.55% | **3.4881** | 14.7861 | 42.307 | 54.58 | 1.213 | 0.3 |
| `full[2]` | 1-single | 1 | 1.00 | 2.93% | 2.93% | **4.4894** | 18.1903 | 30.250 | 52.80 | 1.030 | 0.4 |
| `full[23,24,25,26]` | 2-contig | 4 | 4.00 | 11.70% | 11.70% | **5.1231** | 9.5241 | 20.289 | 51.98 | 1.133 | 0.3 |
| `full[0]` | 1-single | 1 | 1.00 | 2.93% | 2.93% | **14.1837** | 26.8038 | 41.123 | 2.05 | 1.030 | 0.6 |

## Byte accounting

<details><summary>methodology</summary>

```
BYTES-TOUCHED-PER-TOKEN under a skip pattern -- the arithmetic
==============================================================
Same roofline as ladder/: at batch 1, decode_tok_s ~= B_eff / bytes_per_token,
so a depth cut is only worth something if it cuts the bytes a token reads.
Mirrors ../ladder/ladder_bench.py's METHOD A accounting rules for a dense model.

BASE (bf16, nothing skipped). Every parameter is read in full for every token,
with one exception -- token embedding:

  base_touched = 2 bytes x (all params except embed_tokens)
               + 2 bytes x (lm_head params)
               + ~0 for the embed_tokens row gather (~4 KB, rounded to zero)

  Qwen3 ties lm_head to embed_tokens, so `named_parameters()` yields that
  tensor once and it must still be counted once as the output projection:
  with tying, base_touched = 2 x total_params exactly. Without tying,
  base_touched = 2 x (total_params - embed_params). The code reads
  `config.tie_word_embeddings` and does not guess.

PER LAYER. Each decoder layer's countable bytes are its 7 linear families
(../code/swap_eval.py:89-97):

  attn_bytes[i] = 2 x (q_proj + k_proj + v_proj + o_proj params of layer i)
  mlp_bytes[i]  = 2 x (gate_proj + up_proj + down_proj params of layer i)
  layer_bytes[i]= attn_bytes[i] + mlp_bytes[i]

  RMSNorm tensors (input_layernorm, post_attention_layernorm, q_norm, k_norm)
  are deliberately EXCLUDED from the saving. They are ~2 x hidden bf16 per
  layer = 0.008% of a layer, and the probe's stub mechanism still executes
  them. Excluding them makes every saving number a conservative floor.

PATTERN.
  touched = base_touched
          - sum(layer_bytes[i] for i in skip_full)
          - sum(attn_bytes[i]  for i in skip_attn)
          - sum(mlp_bytes[i]   for i in skip_mlp)
          + sum(layer_bytes[src] for (slot, src) in loop)   <-- LOOP RE-READS

  bytes_saved_pct = 100 x (base_touched - touched) / base_touched
  layer_equivalents = (base_touched - touched) / mean(layer_bytes)
  speedup_x = base_touched / touched   (roofline ceiling, NOT a measurement)

  The loop term is the honest part. Executing layer `src` a second time at
  depth `slot` reads its ~50 MB again -- at batch 1 the weights are long gone
  from cache by the time the stack comes back around. A loop therefore buys
  back nothing on the BANDWIDTH ledger; it only shrinks the FOOTPRINT ledger:

  unique = base_unique - sum(layer_bytes[i] for i never executed at all)

  where "never executed" means the layer appears in skip_full and is not the
  source of any loop. Footprint matters for what fits in VRAM; bandwidth
  matters for tok/s. This probe reports both and never conflates them.

WHAT THIS DOES NOT MODEL: KV-cache traffic (grows with context, unaffected by
which layers run except that skipped layers store no KV -- a second-order win
we do not claim), activation traffic, and the fact that a real skipping runtime
must still hold the skipped weights resident unless the pattern is static and
the weights are dropped at load time. Static patterns CAN drop them; that is
why the unique/footprint column is meaningful here and would not be for a
per-input predictor.
```
</details>

## Raw rows

Every row above is one line of `results.jsonl`, written and fsync'd before the next measurement started. If the host hard-crashes, at most the in-flight eval is lost; re-running the same stage resumes.

