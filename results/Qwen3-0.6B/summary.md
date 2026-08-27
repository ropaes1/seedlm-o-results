# Phase 3 results: Qwen3-0.6B, run phase3-20260817-233312

- stack: `test-laptop|NVIDIA GeForce RTX 5070 Laptop GPU|torch2.11.0+cu128`  (all rows below were measured on this one stack; same-stack determinism rule)
- seed configs present: c12p4 | eval mode: cached
- stage1_seeds=65535 stage2_seeds=65535 generator_seed=3407
- torch 2.11.0+cu128 | transformers 5.15.0 | device cuda
- layers compressed: 28 | target tensors: 196 | prompts sha256 0fd801b8d2b690a0…
- originals sha256 f47f71177f32bcd1… (verified: True)

## bits/weight decomposition

| variant | kind | config | fit obj | base bpw | + side bits / numel | = total bpw | side bits | whole-model MB @bf16 rest |
|---|---|---|---|---|---|---|---|---|
| `c12p4_seed_only` | seed | c12p4 | none | 3.00 | 0.0000 | **3.0000** | 0 | 476.4 |
| `c12p4_awq_p0.5` | seed | c12p4 | none | 3.00 | 0.2400 | **3.2400** | 105,689,472 | 489.7 |
| `c12p4_awq_p1.0` | seed | c12p4 | none | 3.00 | 0.4800 | **3.4800** | 211,388,352 | 502.9 |
| `c12p4_awq_p2.0` | seed | c12p4 | none | 3.00 | 0.9600 | **3.9600** | 422,782,080 | 529.3 |
| `rtn4` | proxy | - | none | 4.50 | 0.0000 | **4.5000** | 0 | 559.0 |
| `bf16` | control | - | none | 16.00 | 0.0000 | **16.0000** | 0 | 1192.1 |
| `c12p4_seed_only@2048` | seed | c12p4 | none | 3.00 | 0.0000 | **3.0000** | 0 | 1141.0 |
| `c12p4_awq_p2.0@2048` | seed | c12p4 | none | 3.00 | 0.9600 | **3.9600** | 30,198,720 | 1144.8 |

_Comparator rows carry base 0.00: their entire cost is the side-bits column, read from the GGUF/AWQ container's own stored bytes rather than from a nominal label._

## behavioural damage (teacher-forced vs bf16 reference)

| variant | bpw | mean KL | p95 KL | max KL | top-1 agree | mean rel_err | act-wtd rel_err | fingerprint | fit s | eval s |
|---|---|---|---|---|---|---|---|---|---|---|
| `c12p4_seed_only` | 3.0000 | 2.133638 | 8.420650 | 24.0395 | 56.21% | 0.2274 | 0.2555 | `993831dec4ed89fc` | 5181 | 89 |
| `c12p4_seed_only@2048` | 3.0000 | 1.149904 | 6.376790 | 21.7735 | 74.42% | 0.3268 | - | `a2c7bc3c2432f6a4` | 10 | 30 |
| `c12p4_awq_p0.5` | 3.2400 | 0.856934 | 3.819799 | 12.9135 | 73.27% | 0.2226 | 0.2186 | `892075f8b7ee06fe` | 256 | 56 |
| `c12p4_awq_p1.0` | 3.4800 | 0.758701 | 3.324669 | 12.2968 | 74.57% | 0.2180 | 0.2066 | `865ce9c752fca48c` | 465 | 55 |
| `c12p4_awq_p2.0` | 3.9600 | 0.592965 | 2.637664 | 10.0090 | 78.76% | 0.2100 | 0.1916 | `5276490760c29f60` | 857 | 43 |
| `c12p4_awq_p2.0@2048` | 3.9600 | 0.115675 | 0.457511 | 4.5436 | 91.04% | 0.3019 | - | `fd61147fb0d55a05` | 2 | 19 |
| `rtn4` | 4.5000 | 0.170426 | 0.684151 | 6.9979 | 87.57% | 0.0994 | 0.1151 | `b9d7cde45a00cf4e` | 0 | 20 |
| `bf16` | 16.0000 | 0.000000 | 0.000000 | 0.0000 | 100.00% | 0.0000 | - | `16998bd6cbab04e7` | 0 | 19 |

## controls

- `bf16` (mean_kl == 0.0 and fingerprint == reference): **PASS**: mean_kl=0.0, fp=16998bd6cbab04e7, matches_reference=True
- `rtn4` (mean_kl > 0): **PASS**: mean_kl=0.17042645812034607, fp=b9d7cde45a00cf4e, matches_reference=False

## gate verdict (Phase 3)

**INCOMPLETE**: pass_a=None pass_b=None (comparators available: none)

- **PASS (a), the sub-4 claim**: a seed variant at bpw <= 3.6 with KL <= KL(q3km) and bpw <= 0.92·bpw(q3km).
  - q3km: KL n/a @ n/a bpw
  - best qualifying: `None` KL n/a @ n/a bpw
- **PASS (b), the parity claim**: a seed variant with KL <= 1.1·KL(q4km) and bpw <= 0.85·bpw(q4km).
  - q4km: KL n/a @ n/a bpw
  - best qualifying: `None` KL n/a @ n/a bpw
- best seed variant overall: `c12p4_awq_p2.0@2048` KL 0.115675 @ 3.9600 bpw
- best sub-4-bpw seed variant: `c12p4_awq_p1.0` KL 0.758701 @ 3.4800 bpw
- benchmark-consistent (winner within 3 points of q4km, reported not gating): None 

> `INCOMPLETE` means no real comparator row was available in this run (it predates the GGUF/AWQ comparator artifacts). It is **not** a negative result: rerun `--stage comparators` once the artifacts exist.

