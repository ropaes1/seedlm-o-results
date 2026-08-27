# The experiment we are chasing (status of record)

Updated 2026-08-18. One page: what we're testing, what's measured, what
decides the next step.

## The claim under test

Seeded pseudorandom bulk (LFSR bases regenerated from a 16-bit seed) +
exact outlier side-channel can be competitive with real llama.cpp K-quants
at ~3 bpw, **if the fit places its error where activations are quiet.**

## Measured so far (Qwen3-1.7B, rented 4090 stack, all same-stack)

| variant | bpw | mean KL | top-1 | status |
|---|---|---|---|---|
| q4km (imatrix, measured) | 4.80 | 0.032 | 96.9% | comparator |
| rtn4 | 4.50 | 0.176 | 91.7% | comparator |
| **q3km (imatrix, measured)** | **3.85** | **0.205** | 90.3% | **the gate target** |
| c12p4_awq_p1.0 (plain L2 fit, 16,384 seeds) | 3.48 | 1.818 | 66.2% | Phase 3, FAIL, 9x off |
| c12p4_awq_p1.0 (plain L2 fit, 65,535 seeds) | 3.48 | 1.121 | 76.3% | the W1 one-variable control |
| c12p4r_awq_p1.0 (W2 without W1) | 3.48 | 0.940 | 77.0% | exact rounding under plain L2 |
| **c12p4w_awq_p1.0 (W1: weighted fit)** | **3.48** | **0.436** | 84.4% | **current best, 2.1x off** |
| c12p4w@wt128a_awq_p1.0 (held-out calib a) | 3.48 | 0.551 | 83.2% | calibration replicate |
| c12p4w@wt128b_awq_p1.0 (held-out calib b) | 3.48 | 0.656 | 82.5% | calibration replicate |

Formal Phase 3 gate verdict on the unweighted pipeline: **FAIL** (both
gates). W1 result: **concept alive**. The one-variable effect of the
objective is a **2.6x** KL improvement at zero bit cost (1.121 -> 0.436,
both rows at 65,535 seeds), with plain rel_err moving *backwards* by
15.6%: error was RELOCATED, not reduced; structure was the failure mode.
(A 4.2x figure sometimes quoted, 1.818 -> 0.436, crosses two variables:
1.62x from the seed budget, 2.57x from the objective.)

Calibration error bar: the two disjoint held-out draws span **0.106 KL**
(0.551 to 0.656), which is the noise floor for any weighted comparison
here; the p12-calibrated 0.436 sits **below that entire band**, putting
the calibration-on-test-set leakage advantage at approx. 0.12-0.22 KL.

- 0.6B (Phase 2, c8p3): seeded ~matched RTN, the small-model result that
  started this. (0.6B W1 completion, twin + weighted, remains a pending
  rented-GPU job.)

## Branch tree (W1/W2/W3 follow-up, built 2026-08-18)

Implementation: runner.py flags `--fit-weighting {none,awq}` /
`--coeff-rounding {nearest,weighted}` / `--incoherence {none,had}`,
slugs `c12p4{w}{r}{h}`; `runner.py --help` documents the exact
commands.

- **Rung 0:** single-tensor probe on 1.7B. Decides whether W3 rungs run
  at all. Note the pre-result below.
- **Rung 1: W1+W2** (weighted coefficient rounding, 48-candidate
  exponent+rounding search; provably never regresses the weighted error).
- **Rung 2: W1+W3** (design-A incoherence: fit T = W diag(s) H under
  plain L2). **Pre-result (CPU, one real tensor): design A is 17% WORSE
  than W1.** Qwen3 rows are already near-Gaussian (kurtosis ~3.9), so
  there is no incoherence to fix, and rotating destroys the per-block
  concentration that lets 4 coefficients specialise. Rung 0 exists to
  confirm this cheaply before skipping W3.
- **Rung 3: W1+W2+W3**, only if rung 2 surprises.
- **Verdict rung:** if best KL <= 0.205 @ <=3.6 bpw, run formal
  `--stage verdict` for PASS(a). If 0.205-0.30: near-miss, consider
  repositioning (decode-bandwidth claim, Phase 4). If > 0.30: ceiling
  measured, write up.

**LADDER COMPLETE (2026-08-19). Ceiling measured: KL 0.436 @ 3.48 bpw
(W1), gap to q3km 2.1x.**
- Rung 0 (1.7B probe): W3 29% worse than W1 on the shared metric. Dead,
  mechanism understood (rows already Gaussian; decode amplification).
- Rung 1 (c12p4wr full run): weighted rel_err improved 1.7%, **KL
  degraded 11%** (0.436 -> 0.486). The weighted-L2 proxy (12 calibration
  prompts) has exhausted its alignment with KL: the Goodhart point is
  reached, and optimizing the current objective harder now overfits it.
- Formal gate: FAIL (both), final. Best-ever variant: c12p4w_awq_p1.0.

Remaining unexplored ideas would need a BETTER OBJECTIVE (more
calibration data, Hessian/KL-aware proxy), not more optimization
pressure: GPTQ-style compensation, joint seed x rounding, richer
calibration. Each is a new spec-build-measure cycle against a measured
2.1x with a proxy known to saturate. The fork (write-up, more cycles, or
Phase 4 bandwidth repositioning) is open as of 2026-08-19.

## Standing rules

- All c12p4-family runs happen on rented GPUs only (the local test
  machine hit a reproducible hardware fault under this workload class).
- One variable per rung; measure cheaply before running the full rung
  (the rung-0 pattern).
- rel_err does not track KL across scale; only same-stack KL decides.
- Every result row must be same-stack; local-machine rows are drift data
  only.
