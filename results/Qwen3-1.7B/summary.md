# Phase 3 results: Qwen3-1.7B, run phase3-20260823-051039

- stack: `2e170720595f|NVIDIA GeForce RTX 4090|torch2.11.0+cu128`  (all rows below were measured on this one stack; same-stack determinism rule)
- seed configs present: c12p4, c12p4r, c12p4w, c12p4w@wt128a, c12p4w@wt128b, c12p4w@wt256, c12p4wr, c8p3 | eval mode: cached
- stage1_seeds=65535 stage2_seeds=65535 generator_seed=3407
- torch 2.11.0+cu128 | transformers 5.14.1 | device cuda
- layers compressed: 28 | target tensors: 196 | prompts sha256 0fd801b8d2b690a0…
- act-scale calibration: `p12` (12 prompts)
- originals sha256 42fb692854712fa1… (verified: True)

## bits/weight decomposition

| variant | kind | config | fit obj | rounding | incoh | calib | base bpw | + side bits / numel | + scales | = total bpw | side bits | whole-model MB @bf16 rest |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `c12p4_seed_only` | seed | c12p4 | none | nearest | none | - | 3.00 | 0.0000 | 0.0000 | **3.0000** | 0 | 1151.1 |
| `c12p4_awq_p1.0` | seed | c12p4 | none | nearest | none | p12 | 3.00 | 0.4800 | 0.0000 | **3.4800** | 676,455,360 | 1235.6 |
| `rtn4` | proxy | - | none | nearest | none | - | 4.50 | 0.0000 | 0.0000 | **4.5000** | 0 | 1415.3 |
| `q4km` | comparator | - | none | nearest | none | - | 0.00 | 4.8008 | 0.0000 | **4.8008** | 6,765,674,496 | 1468.3 |
| `q3km` | comparator | - | none | nearest | none | - | 0.00 | 3.8478 | 0.0000 | **3.8478** | 5,422,710,784 | 1300.4 |
| `bf16` | control | - | none | nearest | none | - | 16.00 | 0.0000 | 0.0000 | **16.0000** | 0 | 3441.1 |
| `c12p4w@wt128a_seed_only` | seed | c12p4w@wt128a | awq | nearest | none | wt128a | 3.00 | 0.0000 | 0.0000 | **3.0000** | 0 | 1151.1 |
| `c12p4w@wt128a_awq_p1.0` | seed | c12p4w@wt128a | awq | nearest | none | wt128a | 3.00 | 0.4800 | 0.0000 | **3.4800** | 676,455,360 | 1235.6 |
| `c12p4w@wt128b_seed_only` | seed | c12p4w@wt128b | awq | nearest | none | wt128b | 3.00 | 0.0000 | 0.0000 | **3.0000** | 0 | 1151.1 |
| `c12p4w@wt128b_awq_p1.0` | seed | c12p4w@wt128b | awq | nearest | none | wt128b | 3.00 | 0.4800 | 0.0000 | **3.4800** | 676,455,360 | 1235.6 |
| `c12p4r_seed_only` | seed | c12p4r | none | weighted | none | - | 3.00 | 0.0000 | 0.0000 | **3.0000** | 0 | 1151.1 |
| `c12p4r_awq_p1.0` | seed | c12p4r | none | weighted | none | p12 | 3.00 | 0.4800 | 0.0000 | **3.4800** | 676,455,360 | 1235.6 |

_Comparator rows carry base 0.00: their entire cost is the side-bits column, read from the GGUF/AWQ container's own stored bytes rather than from a nominal label._

_`fit obj = awq` rows (config slug `…w`) were fitted against the activation-weighted objective (W1). The stored format is unchanged (same seed, same shared exponent, same P 4-bit coefficients), so their bpw is identical to the matching unweighted row by construction, and the comparison is at exactly equal bits._

_`rounding = weighted` rows (slug `…r`) chose each block's coefficients by exact search over the 3 x 2^P grid points instead of rounding each coefficient to its own nearest one (W2). Same seed, same exponent field, same P 4-bit coefficients: the `+ scales` and `= total bpw` columns are unchanged, so this comparison is also at exactly equal bits._

_`calib` names the prompt set the activation scales were captured from. `p12` is the original 12-prompt harness every earlier result was calibrated on; other ids carry an `@id` in the config slug and were captured into their own `act_scales@id` file, so the 12-prompt scales and every fit that depends on them are untouched. `calib = -` marks a row that never reads activation scales and is therefore the same object under every calibration set. Recalibration changes no bits: these rows are at equal bpw with their `p12` twins._

## behavioural damage (teacher-forced vs bf16 reference)

| variant | bpw | mean KL | p95 KL | max KL | top-1 agree | mean rel_err | act-wtd rel_err | fingerprint | fit s | eval s |
|---|---|---|---|---|---|---|---|---|---|---|
| `c12p4_seed_only` | 3.0000 | 3.450684 | 13.432451 | 22.3710 | 48.43% | 0.2279 | 0.2741 | `a1f5b79511b23581` | 5142 | 40 |
| `c12p4w@wt128a_seed_only` | 3.0000 | 0.586783 | 3.159957 | 31.1250 | 81.81% | 0.2577 | 0.1757 | `f34aa80a5f72bbee` | 6120 | 29 |
| `c12p4w@wt128b_seed_only` | 3.0000 | 0.655821 | 3.428167 | 22.7736 | 81.81% | 0.2596 | 0.1731 | `414f7895bd28648a` | 6119 | 30 |
| `c12p4r_seed_only` | 3.0000 | 3.231275 | 12.372326 | 21.8241 | 48.84% | 0.2239 | 0.2690 | `b105b91b045a3d3e` | 5166 | 33 |
| `c12p4_awq_p1.0` | 3.4800 | 1.121121 | 6.328577 | 22.6250 | 76.33% | 0.2192 | 0.2276 | `e7dee6f1b513e004` | 569 | 23 |
| `c12p4w@wt128a_awq_p1.0` | 3.4800 | 0.550612 | 2.637969 | 27.3125 | 83.17% | 0.2484 | 0.1664 | `a2d258489d4bc583` | 842 | 27 |
| `c12p4w@wt128b_awq_p1.0` | 3.4800 | 0.656162 | 3.337585 | 26.7500 | 82.49% | 0.2503 | 0.1640 | `2a0dc5503b3cc127` | 842 | 29 |
| `c12p4r_awq_p1.0` | 3.4800 | 0.939671 | 5.548840 | 21.5000 | 77.02% | 0.2154 | 0.2231 | `be786e35df0df287` | 572 | 24 |
| `q3km` | 3.8478 | 0.204809 | 0.966012 | 13.7500 | 90.29% | 0.0000 | - | `ea45e80c2083e804` | 0 | 26 |
| `rtn4` | 4.5000 | 0.176448 | 0.895672 | 7.7504 | 91.66% | 0.1004 | - | `2b0802153224bacc` | 0 | 25 |
| `q4km` | 4.8008 | 0.032270 | 0.142677 | 2.1059 | 96.85% | 0.0000 | - | `7de7ee7bb6f8dae5` | 0 | 25 |
| `bf16` | 16.0000 | 0.000000 | 0.000000 | 0.0000 | 100.00% | 0.0000 | - | `6f7a7d9b5d3fd99d` | 0 | 8 |

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
- best seed variant overall: `c12p4w@wt128a_awq_p1.0` KL 0.550612 @ 3.4800 bpw
- best sub-4-bpw seed variant: `c12p4w@wt128a_awq_p1.0` KL 0.550612 @ 3.4800 bpw
- benchmark-consistent (winner within 3 points of q4km, reported not gating): None 

## cross-stack drift

Two complete verdicts were computed independently, one per stack; **no gate ever mixes them** (same-stack rule). This table only measures how reproducible the numbers are across machines. Reference stack: `2e170720595f|NVIDIA GeForce RTX 4090|torch2.11.0+cu128`.

- stack 0: `2e170720595f|NVIDIA GeForce RTX 4090|torch2.11.0+cu128` -> verdict **FAIL**
- stack 1: `8ebd2f131f8f|NVIDIA GeForce RTX 4090|torch2.11.0+cu128` -> verdict **FAIL**
- stack 2: `bfba9230e93f|NVIDIA GeForce RTX 4090|torch2.11.0+cu128` -> verdict **INCOMPLETE**
- stack 3: `c4ad6bb586c5|NVIDIA GeForce RTX 4090|torch2.11.0+cu128` -> verdict **FAIL**
- stack 4: `e60684062468|NVIDIA GeForce RTX 4090|torch2.11.0+cu128` -> verdict **FAIL**

**DIVERGED**: the 5 same-stack verdicts disagree (2 shared variants compared).

| variant | mean KL @stack0 | mean KL @stack1 | mean KL @stack2 | mean KL @stack3 | mean KL @stack4 | abs ΔKL | rel ΔKL | Δtop-1 pp |
|---|---|---|---|---|---|---|---|---|
| `bf16` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | - | +0.00 |
| `rtn4` | 0.176448 | 0.176448 | 0.176448 | 0.176448 | 0.176448 | 0.000000 | +0.00% | +0.00 |

_rows in the JSON: 52 across 5 stacks._

