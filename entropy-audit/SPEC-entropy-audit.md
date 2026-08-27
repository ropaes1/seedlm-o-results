# BRIEF: Entropy audit of a Q4_K_M index stream

## Context

Cashing the free-compute asymmetry: quantized index streams are not uniformly
distributed, so a nominal 4-bit K-quant contains losslessly recoverable slack.
On bandwidth-bound consumer decode, every recovered 0.5 bpw is ~11% more
tokens/sec at exactly zero quality cost. DFloat11 showed ~30% lossless on
bf16; nobody appears to have shipped entropy-coded K-quants. This audit
MEASURES the slack; it builds no decoder.

The measurement: parse a real Q4_K_M GGUF, extract the quantized index
streams and the scale/min metadata, compute empirical entropy, and report
the gap between stored bits and measured entropy, per tensor family and
whole-model.

Subject file: `unsloth/Qwen3-1.7B-GGUF` Q4_K_M (~1.1 GB), downloaded via the
runbook. CPU-only by construction; no GPU calls at all. Requires Python with
numpy and the `gguf` package. Reference material for the Q4_K block layout:
llama.cpp's format docs/source and the GGUF dequantizer in
[`../code/comparators.py`](../code/comparators.py).

## What to measure

Q4_K_M blocks store: 4-bit indices (256 per super-block), 6-bit per-sub-block
scales/mins, fp16 super-block scale/min (d, dmin). Measure each stream:

1. **Order-0 entropy of the 4-bit indices**, per tensor family (attn q/k/v/o,
   mlp gate/up/down, embeddings/output if quantized) and whole-model.
   Headline: H0 bits vs 4.0 stored.
2. **Order-1 (previous-index-conditioned) entropy** of the indices: the
   cheap upper estimate of what a context model recovers.
3. **Entropy of the 6-bit sub-block scales/mins** vs 6.0 stored.
4. Whole-model accounting: stored bpw (from real file bytes) vs achievable
   bpw at H0 and at order-1; the headline line: "recoverable: X.XX bpw of
   Y.YY stored (Z% smaller, ~W% decode speedup if bandwidth-bound)".
5. Sanity guard: verify parsed tensor count/shapes against the GGUF header;
   fail loud on layout mismatch (Q4_K_M files mix Q4_K/Q5_K/Q6_K per tensor:
   handle, or explicitly skip non-Q4_K tensors WITH a count of skipped
   bytes so the whole-model number stays honest).

## Deliverables

1. `entropy_audit.py`: CLI: `python entropy_audit.py <path.gguf>`
   prints a per-family table + headline, and writes `results-<model>.md`.
   Runtime target: < 10 min on CPU for a 1.7B (stream the file; do not load
   all tensors into RAM at once).
2. `make_test_gguf.py`: constructs a synthetic smoke-test GGUF (a few small
   Q4_K tensors with a KNOWN skewed index distribution so the expected
   entropy is analytically checkable).
3. Smoke test: audit the synthetic file; assert measured H0 within tolerance
   of the constructed distribution's true entropy.
4. `RUNBOOK.md`: the exact download command for the subject GGUF and the
   audit invocation.

## Acceptance

1. Smoke test passes with an analytically verified entropy number.
2. Real-file path handles mixed quant types without lying in the totals.
3. Output table readable at a glance: family, stored bpw, H0 bpw, order-1
   bpw, recoverable bpw, % of family bytes.
4. < 10 min CPU runtime documented.

## Out of scope

Building any encoder/decoder; GPU work; other quant formats (Q3_K etc. noted
as follow-ups only).
