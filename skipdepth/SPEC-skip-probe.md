# BRIEF: Static layer-skip KL probe (depth as a bytes-touched lever)

## Context

Decode speed on consumer hardware = bytes-touched-per-token / effective
bandwidth. MoE cuts bytes-touched across WIDTH (measured in the ladder
experiment, [../ladder/results.md](../ladder/results.md)); this probe asks
whether Qwen3 has slack across DEPTH. PoLar
(arXiv:2606.06574, ICML 2026, code github.com/tianyi-lab/PoLar)
claims training-free per-input layer skipping/looping preserves or improves
accuracy. Their evidence is benchmark accuracy; our yardstick is KL to the
bf16 model, the same metric that exposed Phase 3's failure. Rung 0
is STATIC skip patterns only (no predictor): if no static pattern gets
useful depth reduction at tolerable KL, the predictor phase is not worth
pursuing.

Measured context to compare against (Qwen3-1.7B, RTX 4090, eval prompts,
cached-logits KL): q4km 0.032, rtn4 0.176, q3km 0.205, our best seeded
variant 0.436. A skip pattern's KL lands on this same ruler: "skip 4
of 28 layers costs KL X" reads directly against "q3km costs 0.205 for a 21%
byte cut".

## Hard rules

1. Build and smoke-test with a TINY random-weight model (e.g. a 4-layer toy
   config built in-code, or Qwen3-0.6B config with random init at reduced
   hidden size). `--selftest` must run CPU-only in under a minute; the full
   probe runs on the GPU host where Qwen3-1.7B already exists.
2. The probe does not import from the main harness. It may READ
   [../code/runner.py](../code/runner.py) and
   [../code/swap_eval.py](../code/swap_eval.py) to replicate the eval
   protocol (prompt list, max positions, KL definition), replicated in its
   own code, byte-for-byte on the protocol constants, citing in a comment
   which lines were mirrored.
3. Same-stack discipline: all rows this probe reports are computed by THIS
   script on ONE stack. Recompute the bf16 reference and at least one anchor
   row (see acceptance #3); never import numbers from the main results files
   into the table.

## The probe

`skip_probe.py` (this folder): CLI, torch + transformers, loads Qwen3-1.7B
bf16, and evaluates KL(bf16 || skipped) on the exact main eval protocol
(same prompts, same token positions, same KL formula). Skipping = bypassing
a decoder layer's entire block (residual stream passes through untouched)
via forward hooks or a wrapped module list, NOT deleting weights; the
mechanism is documented precisely.

Pattern sweep, in priority order (stop-loss rules below):

1. **Single-layer scan**: skip each layer alone (28 evals). Output: KL per
   skipped layer index, the depth-sensitivity map. Expect first/last layers
   to be load-bearing (literature consensus); measure, don't assume.
2. **Contiguous blocks**: skip the k least-sensitive contiguous runs for
   k = 2, 3, 4, 6 (from the scan, pick the best few candidates each, cap 12
   evals).
3. **Greedy non-contiguous**: greedily grow a skip set from the single-layer
   scan (re-evaluate at each step, since sensitivities interact; cap 8
   evals).
4. **Attention-only vs MLP-only** skipping for the 4 least-sensitive layers
   (8 evals): the two sublayers differ in bytes and may differ in KL cost.
5. **Loop probe (stretch, cap 4 evals)**: repeat the 2 least-sensitive
   layers in place of their skipped neighbors (PoLar's "loop" arm). Does
   re-using resident weights buy KL back at zero extra bytes?

Every eval appends one JSON line to `results.jsonl` (crash-safe, same
pattern as [../ladder/](../ladder/results.md)) with: pattern spec,
n_layers_skipped, KL, top-1 agreement, bytes/token saved (computed from the
GGUF-style tensor accounting: per-layer byte size x layers skipped / total,
with the arithmetic documented), and wall seconds. `results.md` is rendered
after each append: table sorted by KL, plus the headline curve "depth
removed vs KL" with the q3km=0.205 and rtn4=0.176 reference lines marked.

Stop-loss: if the BEST single-layer skip already costs KL > 0.4, stop after
stage 1 and report; depth has no slack and the verdict is cheap. If stage
2's best 4-layer pattern costs KL > 1.0, skip stages 3-5.

## Deliverables

1. `skip_probe.py` with `--selftest` (CPU, tiny model, asserts:
   skipping an identity-behaving layer changes nothing; skipping a real
   layer changes logits; results.jsonl row schema stable; KL of
   model-vs-itself is exactly 0).
2. `RUNBOOK.md`: copy-paste style:
   env setup, file sync, exact invocations per stage, expected wall time
   per stage, and the decision rule:
   - some pattern reaches >=4 layers skipped (>=14% bytes) at KL <= 0.205
     (q3km-competitive) -> the predictor phase (PoLar repo) earns a spec;
   - best is 2-3 layers at KL <= 0.205 -> marginal; composes with
     quantization but thin alone; report and hold;
   - nothing beats KL 0.4 at any useful depth -> depth is load-bearing in
     Qwen3-1.7B; write the negative result into the table and close.
3. `NOTES-polar.md`: one page on what the PoLar repo actually
   ships (predictor architecture, training cost, inference overhead,
   supported models) so the phase-2 decision is informed. If the repo
   diverges from the paper, say so.

## Acceptance

1. `--selftest` ALL PASS, CPU-only, < 60 s, nothing downloaded that exceeds
   a few MB (the PoLar repo clone is fine).
2. Eval protocol constants provably mirrored from the main harness (cited
   by line).
3. Anchor validation on the GPU host, in the runbook: before the sweep, the
   script evaluates the skip-nothing pattern (KL must be exactly 0.0) and
   reports bf16 top-1 self-agreement (must be 100%); plus ONE spot-check
   row that can be eyeballed against known behavior.
4. results.jsonl appended before each next eval starts; a crash loses at
   most the in-flight eval.

## Out of scope

Training or fine-tuning anything (the PoLar predictor is phase 2, only if
rung 0 passes); MoE models; quality benchmarks beyond KL + top-1 (tok/s
implications are computed analytically from bytes saved, not measured).
