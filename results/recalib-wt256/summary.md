# Phase 3 results: Qwen3-1.7B (wt256 recalibration), run phase3-20260822-084526

- stack: `8ebd2f131f8f|NVIDIA GeForce RTX 4090|torch2.11.0+cu128`  (all rows below were measured on this one stack; same-stack determinism rule)
- seed configs present: c12p4, c12p4w, c12p4w@wt256, c12p4wr, c8p3 | eval mode: cached
- stage1_seeds=65535 stage2_seeds=65535 generator_seed=3407
- torch 2.11.0+cu128 | transformers 5.14.1 | device cuda
- layers compressed: 28 | target tensors: 196 | prompts sha256 0fd801b8d2b690a0…
- act-scale calibration: `wt256` (256 prompts, corpus wikitext-2-raw `wiki.test.raw`, sha256 3e4cc6171387b03f…)
- originals sha256 42fb692854712fa1… (verified: True)

## bits/weight decomposition

| variant | kind | config | fit obj | rounding | incoh | calib | base bpw | + side bits / numel | + scales | = total bpw | side bits | whole-model MB @bf16 rest |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `c12p4w@wt256_seed_only` | seed | c12p4w@wt256 | awq | nearest | none | wt256 | 3.00 | 0.0000 | 0.0000 | **3.0000** | 0 | 1151.1 |
| `c12p4w@wt256_awq_p1.0` | seed | c12p4w@wt256 | awq | nearest | none | wt256 | 3.00 | 0.4800 | 0.0000 | **3.4800** | 676,455,360 | 1235.6 |
| `rtn4` | proxy | - | none | nearest | none | - | 4.50 | 0.0000 | 0.0000 | **4.5000** | 0 | 1415.3 |
| `q4km` | comparator | - | none | nearest | none | - | 0.00 | 4.8008 | 0.0000 | **4.8008** | 6,765,674,496 | 1468.3 |
| `q3km` | comparator | - | none | nearest | none | - | 0.00 | 3.8478 | 0.0000 | **3.8478** | 5,422,710,784 | 1300.4 |
| `bf16` | control | - | none | nearest | none | - | 16.00 | 0.0000 | 0.0000 | **16.0000** | 0 | 3441.1 |

_Comparator rows carry base 0.00: their entire cost is the side-bits column, read from the GGUF/AWQ container's own stored bytes rather than from a nominal label._

_`fit obj = awq` rows (config slug `…w`) were fitted against the activation-weighted objective (W1). The stored format is unchanged (same seed, same shared exponent, same P 4-bit coefficients), so their bpw is identical to the matching unweighted row by construction, and the comparison is at exactly equal bits._

_`calib` names the prompt set the activation scales were captured from. `p12` is the original 12-prompt harness every earlier result was calibrated on; other ids carry an `@id` in the config slug and were captured into their own `act_scales@id` file, so the 12-prompt scales and every fit that depends on them are untouched. `calib = -` marks a row that never reads activation scales and is therefore the same object under every calibration set. Recalibration changes no bits: these rows are at equal bpw with their `p12` twins._

## behavioural damage (teacher-forced vs bf16 reference)

| variant | bpw | mean KL | p95 KL | max KL | top-1 agree | mean rel_err | act-wtd rel_err | fingerprint | fit s | eval s |
|---|---|---|---|---|---|---|---|---|---|---|
| `c12p4w@wt256_seed_only` | 3.0000 | 0.597257 | 3.320129 | 20.8393 | 82.08% | 0.2581 | 0.1748 | `f15677f2a878713d` | 6123 | 42 |
| `c12p4w@wt256_awq_p1.0` | 3.4800 | 0.569260 | 2.925778 | 24.8750 | 83.45% | 0.2489 | 0.1656 | `94b24f9d7023c4fc` | 851 | 41 |
| `q3km` | 3.8478 | 0.204809 | 0.966012 | 13.7500 | 90.29% | 0.0000 | - | `ea45e80c2083e804` | 0 | 38 |
| `rtn4` | 4.5000 | 0.176448 | 0.895672 | 7.7504 | 91.66% | 0.1004 | - | `2b0802153224bacc` | 0 | 37 |
| `q4km` | 4.8008 | 0.032270 | 0.142677 | 2.1059 | 96.85% | 0.0000 | - | `7de7ee7bb6f8dae5` | 0 | 36 |
| `bf16` | 16.0000 | 0.000000 | 0.000000 | 0.0000 | 100.00% | 0.0000 | - | `6f7a7d9b5d3fd99d` | 0 | 19 |

## controls

- `bf16` (mean_kl == 0.0 and fingerprint == reference): **PASS**: mean_kl=0.0, fp=6f7a7d9b5d3fd99d, matches_reference=True
- `rtn4` (mean_kl > 0): **PASS**: mean_kl=0.17644770443439484, fp=2b0802153224bacc, matches_reference=False

## gate verdict (Phase 3)

**FAIL**: pass_a=False pass_b=False (comparators available: q3km, q4km)

- **PASS (a), the sub-4 claim**: a seed variant at bpw <= 3.6 with KL <= KL(q3km) and bpw <= 0.92·bpw(q3km).
  - q3km: KL 0.204809 @ 3.8478 bpw
  - best qualifying: `None` KL n/a @ n/a bpw
- **PASS (b), the parity claim**: a seed variant with KL <= 1.1·KL(q4km) and bpw <= 0.85·bpw(q4km).
  - q4km: KL 0.032270 @ 4.8008 bpw
  - best qualifying: `None` KL n/a @ n/a bpw
- best seed variant overall: `c12p4w@wt256_awq_p1.0` KL 0.569260 @ 3.4800 bpw
- best sub-4-bpw seed variant: `c12p4w@wt256_awq_p1.0` KL 0.569260 @ 3.4800 bpw
- benchmark-consistent (winner within 3 points of q4km, reported not gating): None 

## cross-stack drift

Two complete verdicts were computed independently, one per stack; **no gate ever mixes them** (same-stack rule). This table only measures how reproducible the numbers are across machines. Reference stack: `8ebd2f131f8f|NVIDIA GeForce RTX 4090|torch2.11.0+cu128`.

- stack 0: `8ebd2f131f8f|NVIDIA GeForce RTX 4090|torch2.11.0+cu128` -> verdict **FAIL**
- stack 1: `bfba9230e93f|NVIDIA GeForce RTX 4090|torch2.11.0+cu128` -> verdict **INCOMPLETE**
- stack 2: `c4ad6bb586c5|NVIDIA GeForce RTX 4090|torch2.11.0+cu128` -> verdict **FAIL**
- stack 3: `e60684062468|NVIDIA GeForce RTX 4090|torch2.11.0+cu128` -> verdict **FAIL**

**DIVERGED**: the 4 same-stack verdicts disagree (2 shared variants compared).

| variant | mean KL @stack0 | mean KL @stack1 | mean KL @stack2 | mean KL @stack3 | abs ΔKL | rel ΔKL | Δtop-1 pp |
|---|---|---|---|---|---|---|---|
| `bf16` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | - | +0.00 |
| `rtn4` | 0.176448 | 0.176448 | 0.176448 | 0.176448 | 0.000000 | +0.00% | +0.00 |

_rows in the JSON: 40 across 4 stacks._

