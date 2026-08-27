# SeedLM+O: a measured negative result on seeded-bulk weight compression

This repository is the public record of a compression experiment that
**failed its own gates**. It is released because the failure is
informative and the measurements are clean: four findings survive the
failure, and one of them (the error-structure result) is the reason the
method got as close as it did.

---

## What was attempted

**The goal** was to create an alternative style of compression, to
substitute for or add on to existing inference harnesses, so that large
open-source models can run on mid-tier consumer hardware with less VRAM.

The project started from a working hypothesis that pre-selecting certain
weights to be stored exactly, while regenerating the rest on the fly
from a small seed, would shrink what has to sit in VRAM and free it up
for token processing. I thought of a bell curve: if the few weights in
the tails of the distribution could carry enough information stored
exactly, the rest could be regenerated from the encoded seed. (As the
findings below document, that intuition was half right: protecting a
few weights exactly does matter, but the magnitude tails turned out to
be the wrong tails to protect. Ranking outliers by activation salience
beat ranking them by size, which is finding 2's error-structure
result.) The thought was that this could work because at batch size 1, decode speed
is limited by how many bytes of weights must be read per token, not by
compute. A weight that can be regenerated from a 16-bit
seed costs almost nothing to store, so if most weights could live as
seeds and only the few that matter most were stored exactly, the model
would both fit in less memory and touch fewer bytes per token. The
experiment below tests whether that trade holds up against the standard
quantization formats it would have to beat; the appendix measures the
same bytes-per-token idea from a different angle, using the
mixture-of-experts offloading that llama.cpp already ships.

**The claim under test.** Represent the bulk of a weight tensor as a
pseudorandom basis regenerated from a 16-bit LFSR seed (so the bulk
costs *only* the seed plus a handful of 4-bit coefficients), and pay for
the parts that matter with an exact outlier side-channel. If the error
is placed where activations are quiet, the resulting format should be
competitive with llama.cpp's K-quants at approximately 3 bits per
weight.

**How it was judged.** Same-stack, teacher-forced mean KL divergence
against a bf16 reference on Qwen3-1.7B, with real measured comparators
in the same harness: `Q3_K_M` (imatrix) at 3.85 bpw and `Q4_K_M` at
4.80 bpw. Two formal gates: a sub-4-bpw claim and a parity claim.

**Verdict: FAIL, both gates, final.**

| variant | bpw | mean KL | top-1 agree |
|---|---:|---:|---:|
| `q4km` (imatrix, measured) | 4.80 | 0.032 | 96.9% |
| `rtn4` | 4.50 | 0.176 | 91.7% |
| **`q3km` (imatrix, measured), the gate target** | **3.85** | **0.205** | 90.3% |
| `c12p4_awq_p1.0` (plain L2 fit, 16,384 seeds) | 3.48 | 1.818 | 66.2% |
| `c12p4_awq_p1.0` (plain L2 fit, 65,535 seeds, the W1 control) | 3.48 | 1.121 | 76.3% |
| **`c12p4w_awq_p1.0` (W1, activation-weighted fit), best ever** | **3.48** | **0.436** | 84.4% |
| `c12p4w@wt128a_awq_p1.0` (W1, held-out calibration) | 3.48 | 0.551 | 83.2% |

Best-ever was measured at **2.1x worse KL than Q3_K_M at 10% fewer
bits**.

---

## The four findings

### 1. The niche is dead at scale

Seeded bulk plus an outlier side-channel does not reach K-quant quality
at approximately 3.5 bpw on a 1.7B model. Best-ever `c12p4w_awq_p1.0`
KL 0.436 @ 3.48 bpw against `Q3_K_M` 0.205 @ 3.85 bpw: a 2.1x ceiling
after every optimization the project could afford. Both formal gates
FAIL. This is final for the method as specified.

A related result inverts the usual folklore: `c12p4` **got worse** going
from 0.6B to 1.7B. KL moved 0.759 to 1.121 at the same 3.48 bpw and the
same 65,535-seed budget, a 1.48x penalty, while per-tensor `rel_err`
moved 0.2180 to 0.2192, a 0.55% change. **Bigger models are harder for
this method, and weight-space error does not see it.**

### 2. The structure of the error is the failure mode (W1)

Switching the fit objective from plain L2 to an activation-weighted L2
cut KL **2.6x, 1.121 to 0.436, at exactly zero bit cost** (identical
seed, identical shared exponent, identical 4-bit coefficients; the
harness asserts byte-equality of the stored format). Both rows were
fitted at 65,535 candidate seeds, so the objective is the only variable.

Meanwhile plain `rel_err` moved the wrong way: 0.2192 to 0.2534, 15.6%
worse. Under the metric normally used to rank fits, W1 is the
inferior fit, and it does 2.6x less behavioural damage at the same bits.

For completeness, the raw distance from the first plain-L2 row ever
measured (16,384 seeds, KL 1.818) to the best weighted row is 4.17x,
but that comparison crosses two variables. It decomposes cleanly into a
seed-budget factor (1.818 to 1.121, 1.62x) and the fit-objective factor
(1.121 to 0.436, 2.57x). The one-variable W1 effect is 2.6x.

W1's gain came from relocating the error; the total amount of it grew.
That is the project's most transferable measurement, and the reason
plain weight-space error should never be used to compare fit
mechanisms.

### 3. The Goodhart point of the weighted-L2 proxy (W2)

W2 replaced per-coefficient nearest rounding with an exact search over
the same grid, provably never increasing the weighted objective. It
lowered the objective on 4 of 4 rows (by 1.6-1.7%) and **raised KL on 3
of 4**, by 11% at the frontier row (0.436 to 0.486).

Beyond the single bad row, the result is geometric: the entire
`c12p4wr` family sits **above** the `c12p4w` family's proxy-to-KL curve
at equal weighted error, by +6.8% to +12.8% of KL (mean +8.8%
displacement). Two fits with the *same* weighted error do measurably
different amounts of behavioural damage depending on *how* the error was
obtained. **The proxy-to-KL map is mechanism-dependent.** "Minimise
the objective harder" is therefore not a plan.

The displacement is attributable. The identical exact-rounding search
run *without* W1 (plain L2 objective, `c12p4r`) **improves** KL 16%
(1.121 to 0.940), against the 11% degradation it causes on top of W1.
The same search flips sign according to the objective it minimises, so
**the Goodhart displacement is a property of the weighted metric**,
and any mechanism that pushes harder on that metric inherits it.

Full analysis with elasticities, matched-bits splits, and falsifiable
claims C1-C5: [`docs/experiment/PROXY-ALIGNMENT.md`](docs/experiment/PROXY-ALIGNMENT.md).

### 4. Calibration-on-test-set leakage, measured

Both the fit weighting `s` and the AWQ outlier ranking are mean
|activation| over **12 calibration prompts**. Those 12 prompts **are the
12 evaluation prompts.** Every weighted result in this project was
calibrated on its own test set, and no experiment ever varied that
number.

Refitting with activation scales re-estimated from 256 wikitext
paragraphs moved KL **0.436 to 0.569 (+31%)** at identical bits, with a
scale-drift cosine median of **0.9400** between the two calibration
sets. The recalibrated run is in
[`results/recalib-wt256/`](results/recalib-wt256/) and it also fails
both gates.

The magnitude now has an error bar. Two disjoint 128-paragraph
halves of that corpus, fitted and evaluated independently at identical
bits, land at **KL 0.551 (`wt128a`) and 0.656 (`wt128b`)**.

- **Noise floor: 0.106 KL**, approx. 19% of the lower value, from
  nothing but which paragraphs the scales were captured on. No
  comparison between two weighted configurations closer than approx.
  0.1 KL is resolvable in this harness, which retrospectively disposes
  of several gaps this project once treated as signal.
- **Leakage effect size: approx. 0.12 to 0.22 KL.** The leaked 0.436
  sits **below the entire held-out band [0.551, 0.656]**. The held-out
  figure is 1.26x to 1.50x the headline.

Every weighted number in this repository is a point measurement from a
single calibration draw, carrying that approx. 0.1 KL uncertainty; the
p12-calibrated ones are additionally biased low.

---

## Appendix result: laptop MoE inference

A separate bytes-touched ladder measured what a 30B-class MoE actually
does on consumer hardware: RTX 5070 8GB laptop, 24 logical CPUs,
Windows 11.

**Qwen3-30B-A3B decodes at 54.2 tok/s** at `-ncmoe 30`, against a
win-condition bar of 12 tok/s. The offload curve is smooth from 38.2
tok/s (`-ncmoe 48`, all routed experts on CPU) up to 54.2, and then
**falls off a cliff at `-ncmoe 28`, 10.5 tok/s**, a 5.2x collapse from
one rung earlier as the working set stops fitting in 8 GB of VRAM.
Server-mode real throughput at `-ncmoe 32` is 48.8 tok/s.

> `-ncmoe` (`--n-cpu-moe`) is llama.cpp's own existing feature. This
> project measured it, found the optimal operating point for this
> hardware, and provides a way for other consumers to tune it on
> hardware smaller than the full model would normally need.

**Speculative decoding measured NEGATIVE on this hardware.** A dense
Qwen3-0.6B draft model gave 0.87x and 0.71x speedup at 47.4% and 51.4%
acceptance: the draft's own weight reads cost more than the
verification saved. The n-gram drafter gave 0.99x at 0% acceptance.
Full rows and the bytes-per-token methodology:
[`ladder/results.md`](ladder/results.md).

Known bug encountered: **llama.cpp b10502 crashes with "invalid vector
subscript" when a dense draft model is used together with `-ncmoe`.**

## Appendix result: skip-depth probe (negative)

A depth-routing probe came back negative overall: there is no broad
layer slack to harvest. **The one stacking nugget is L27: KL 0.145 for
2.9% of bytes touched.**

> Footnote for anyone quoting that 0.145 against `q3km` 0.205, `rtn4`
> 0.176, or the 0.436 headline: the skip probe evaluated **731
> positions** against the seed harness's **726**, container drift, not
> a like-for-like KL. Treat cross-quotes as indicative.

A build-time audit of the reference implementation (PoLar, ICML 2026)
found its released headline protocol is best-of-5 scored against ground
truth with a default-on answer-key short-circuit, and that its
supervision generator is not in the release:
[`skipdepth/NOTES-polar.md`](skipdepth/NOTES-polar.md).

---

## Reporting notes

- **W3 (Hadamard incoherence) is a dud, and the mechanism is
  understood.** Measured **29% worse than W1** on 1.7B. Qwen3 rows are
  already near-Gaussian (kurtosis approx. 3.9), so there is no
  incoherence to fix, and the rotation destroys the per-block
  concentration that lets 4 coefficients specialise.
- **`rel_err` does not track KL across scale.** Weight-space error is
  at best a prior.
- **Partial-fit rows are a trap.** Two `--layers-limit` smoke rows carry
  full-model-shaped bpw and KL and, taken at face value, produce a
  spectacular *fake* Goodhart result. They are detectable from their own
  effective-bytes arithmetic; the guard is `flag_partial_fits` in
  `code/proxy_alignment.py`.
- **Stacks are never pooled.** Every gate verdict is computed inside one
  hardware/software stack.

---

## How to reproduce

The harness is `code/runner.py` (fit, evaluate, gate) plus
`code/comparators.py` (real llama.cpp GGUF comparators). Fits are
deterministic from `generator_seed=3407` and cached per tensor, so an
interrupted run resumes.

```bash
# 1. environment (a GPU pod)
bash code/run_phase3_pod.sh          # end-to-end: deps, model, fit, comparators, gate

# 2. or drive the runner directly
python code/runner.py --model Qwen3-1.7B --config c12p4w \
    --fit-weighting awq --coeff-rounding nearest --incoherence none \
    --stage fit --stage eval --stage verdict

# 3. re-derive the proxy analysis from results + cache (CPU only, no model load)
python code/proxy_alignment.py

# 4. the inference ladder (llama.cpp, consumer hardware)
python ladder/ladder_bench.py
```

Flags that select the four findings' variants:
`--fit-weighting {none,awq}` (W1), `--coeff-rounding {nearest,weighted}`
(W2), `--incoherence {none,had}` (W3), and the calibration-set id that
produces the `@wt256` slug (finding 4). `python code/runner.py --help`
documents every flag.

Model weights, fit caches, and reference logits are **not** in this
repo; they are large and fully regenerable. Everything needed to
regenerate them is.

**Note on the fit environment.** During development, sustained
`c12p4`-class fit runs caused repeated unexplained crashes on one test
laptop. Rolling the GPU driver back to the vendor-validated version
resolved them. The measured results were produced on rented GPUs
regardless, so no result in this repository depends on that machine.

---

## Repo map

| path | what |
|---|---|
| `paper/report.md` | arXiv-style write-up **(complete, all numbers measured; pending author sign-off)** |
| `code/` | the fit/eval/gate harness, comparators, proxy analysis, run scripts |
| `docs/experiment/` | briefs, the status-of-record `EXPERIMENT.md`, `PROXY-ALIGNMENT.md` |
| `results/phase2/` | Qwen3-0.6B Phase 2 (the small-model result that started this) |
| `results/Qwen3-1.7B/`, `results/Qwen3-0.6B/` | Phase 3 / 3.6 summaries and raw JSON |
| `results/recalib-wt256/` | the 256-paragraph recalibration run (finding 4); the `wt128a`/`wt128b` replicates are in `results/Qwen3-1.7B/` |
| `ladder/` | bytes-touched inference ladder: `results.md`, `results.jsonl`, bench script |
| `skipdepth/` | depth-routing probe and the PoLar reference-implementation audit |

Start with [`docs/experiment/EXPERIMENT.md`](docs/experiment/EXPERIMENT.md)
(status of record) and
[`docs/experiment/PROXY-ALIGNMENT.md`](docs/experiment/PROXY-ALIGNMENT.md)
(the analysis behind findings 2-4).

---

## License

Apache-2.0. See [`LICENSE`](LICENSE).

---

*Experiment gates and tests by Claude (Fable 5, Anthropic).*
