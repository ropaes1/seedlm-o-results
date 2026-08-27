# BRIEF: The bytes-touched ladder (laptop inference benchmark)

## Context

Decode speed on consumer hardware = bytes-touched-per-token / effective memory
bandwidth. Every celebrated speedup attacks the numerator: quantization (fewer
bytes per weight), MoE (fewer weights per token), speculative decoding (more
tokens per weight-read). The ladder measures each lever SEPARATELY on the
actual target machine, converting folklore into a measured curve.

Target machine: a Windows 11 laptop with an RTX 5070 Laptop GPU (8 GB VRAM),
32 GB DDR5 (31.5 usable), Intel Ultra 9 275HX. Results are appended to disk
after every measurement, so an interruption loses only the current rung.

Background references (cited in the runbook's intro):
- https://jax-ml.github.io/scaling-book/ (roofline mental model, ch. 1, 7, 8)
- https://inco.ai/blog/dflash2/ (speculative decoding: 2.7-3.4x batch-1 on a
  27B; supported in llama.cpp)

## The ladder

| Rung | Model | Format | ~DL size | Lever isolated |
|---|---|---|---|---|
| R0 | Qwen3-0.6B | Q4_K_M | ~0.5 GB | machine's effective bandwidth (baseline) |
| R1 | Qwen3-4B | Q4_K_M | ~2.5 GB | bpw + dense scaling |
| R2 | Qwen3-30B-A3B | Q4_K_M | ~17.5 GB | bytes-TOUCHED (MoE: ~3B active of 30B) |
| +S | each rung where supported | n/a | n/a | amortization (speculative decoding) |

R2 memory plan: experts in system RAM, attention + KV on GPU (llama.cpp
`--n-gpu-layers` / MoE offload flags; the runbook gives the exact invocation).
Note RAM pressure: ~17.5 GB model + OS in 32 GB, so the runbook says what to
close first.

Speculative decoding: use what llama.cpp actually supports for these models
(draft-model speculation with a small Qwen3 draft). The runbook documents
which speculation mode applies per rung, and states explicitly where an
architecture lacks draft support rather than hand-waving.

**Win condition: R2 decode >= 12 tok/s** (the threshold chosen for
"daily-drivable").

## Deliverables

1. `RUNBOOK.md`: top-to-bottom procedure: downloads (llama.cpp prebuilt CUDA
   Windows zip + the three GGUFs, exact URLs, sha-verifiable where possible),
   then per-rung: exact commands (llama-bench for raw numbers, llama-server or
   llama-cli for a feel-check), expected wall time, what to record. Includes a
   crash protocol (what survives, how to resume) and a short cloud-GPU
   fallback section (not the default).
2. `ladder_bench.py`: orchestrates a rung: runs llama-bench (and the
   speculative variant), parses output, computes tok/s (prefill + decode),
   estimated bytes/token (model bytes touched per token: full size for dense,
   active-expert fraction for MoE; the estimate's assumptions are documented),
   RAM/VRAM occupancy, and APPENDS a row to `results.jsonl` + renders
   `results.md` as a growing table. Runs with Python 3.12; stdlib plus
   optional psutil.
3. Smoke test: `ladder_bench.py selftest` runs the parser and table renderer
   end-to-end against recorded sample llama-bench outputs, with no models
   present.

## Acceptance

1. RUNBOOK.md executes top-to-bottom with zero decisions left to the operator.
2. `ladder_bench.py selftest` passes against fixtures.
3. Every rung's measurement appends to results.jsonl BEFORE the next rung
   starts (crash-safe).
4. bytes/token estimate methodology documented inline.

## Out of scope

Fine-tuning, sparsity induction, quality evals (tok/s only).
