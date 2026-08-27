# RUNBOOK: static layer-skip KL probe (rung 0: is there slack in DEPTH?)

Spec: [SPEC-skip-probe.md](SPEC-skip-probe.md).
Read section 0 before running: it tells you what this probe can and cannot buy,
and section 5 tells you when to stop early.

Everything below runs on a single CUDA host (measured on an RTX 4090) with
Qwen3-1.7B already downloaded. Only the probe script itself
(`skip_probe.py`, ~60 KB) needs to be copied over. The `--selftest` step is
CPU-only (~7 s) and can be run anywhere first.

---

## 0. What this probe is, in one paragraph

MoE cuts bytes-touched-per-token across **width**. This asks whether Qwen3-1.7B
has slack across **depth**: if you simply do not execute some decoder layers,
what does it cost on the same KL ruler that priced the quantizers?

| reference (same model, same 12-prompt protocol, cached-logits KL) | mean KL |
|---|---|
| q4km | 0.032 |
| rtn4 | 0.176 |
| **q3km** (the gate) | **0.205** |
| best seeded variant `c12p4w_awq_p1.0` | 0.436 |

A skip pattern's KL lands on that same ruler. "Skip 4 of 28 layers costs KL X"
reads directly against "q3km costs 0.205 for a 21% byte cut". This is rung 0:
**static patterns only, no predictor**. If no static pattern gets useful depth
reduction at tolerable KL, the PoLar predictor phase (see
[NOTES-polar.md](NOTES-polar.md)) is not worth pursuing.

The probe never modifies weights. It bypasses a decoder layer by temporarily
replacing its `self_attn` / `mlp` children with stubs returning exact zeros, so
the residual stream passes through untouched. `--selftest` proves this is
bit-identical to deleting the layer from the `ModuleList`.

The whole sweep is ~62 KL evaluations and takes roughly 25-30 minutes of wall
time on a 4090. Each eval is 12 teacher-forced forward passes over ~225-token
sequences plus a fp32 log-softmax over `[200, 151936]` per prompt; the
expensive parts are the one-time reference generation and the per-process
model load.

---

## 1. Setup (~5 min)

### 1a. Selftest, then copy the one file

The selftest is CPU-only, builds a 4-layer random-weight Qwen3 in memory, and
downloads nothing. Run it before using GPU time:

```bash
python skip_probe.py --selftest
```

**CHECKPOINT: must print `ALL PASS  67/67 checks` in well under 60 s.**
If any check in section `[A]` fails, the eval-protocol transcription has
drifted from the harness ([../code/swap_eval.py](../code/swap_eval.py)) and
every KL you measure would be on a different ruler than q3km/rtn4. Do not
proceed.

Copy `skip_probe.py` to the GPU host by whatever transfer you use (scp, rsync,
or a git/HF sync); it is the only file needed.

### 1b. On the GPU host: env and verify

```bash
PY=python                     # your venv's python, torch + transformers installed
M=/path/to/models/Qwen3-1.7B  # bf16 safetensors checkpoint

ls -la $M/config.json $M/*.safetensors | head
$PY -c "import torch, transformers; print('torch', torch.__version__,
 'tf', transformers.__version__, 'gpu', torch.cuda.get_device_name(0))"
$PY skip_probe.py --selftest
```

**CHECKPOINT: `ALL PASS  67/67 checks`.** The selftest cross-checks the skip
mechanism against a ModuleList rebuild on whatever torch/transformers version
is installed, so version differences from your build machine are fine, but
note them: rows carry `stack`, and rows from two stacks may never be compared
(the same-stack determinism rule).

---

## 2. Reference + anchor (~4 min)

This is the gate that makes every later number trustworthy. It generates the
frozen bf16 reference (12 prompts x 200 greedy tokens, plus the 160-token
harness fingerprint), caches the fp32 teacher-forced reference logits, then
evaluates the **skip-nothing** pattern against them.

```bash
cd <skipdepth dir>
$PY skip_probe.py anchor --model-dir $M --device cuda 2>&1 | tee logs_anchor.txt
```

**CHECKPOINT: the last line must read:**

```
[skip] ANCHOR PASS: mean_kl=0.0 top1=100.0% positions=726 base_bytes/token=... MB
```

Three things to eyeball, all hard requirements:

1. `mean_kl=0.0`: **exactly** zero, not `1e-9`. The model compared against its
   own cached logits must be bit-identical. Anything else means the reference
   and the live model disagree and the run is void; the script raises
   `SystemExit` rather than continue.
2. `top1=100.0%`: bf16 top-1 self-agreement.
3. `positions=726`: the 1.7B protocol's position count, matching
   [../results/Qwen3-1.7B/phase3_results.json](../results/Qwen3-1.7B/phase3_results.json)
   `-> variants[].kl_positions`. **If this is not 726, stop.** It means the
   greedy continuations differ from the Phase-3 run and your KL is on a
   different ruler. (The run recorded in
   [results.md](results.md) landed on 731 with a newer transformers; see
   section 7's third row for how that was handled.)

`base_bytes/token` must print **3441.1 MB** (2 bytes x 1,720,574,976 tied
params). Qwen3-1.7B bf16 is a ~3.4 GB read per token at batch 1; that is the
number a skip pattern is trying to cut. The full byte table, verified against
the real `config.json`:

| quantity | value | share of bytes/token |
|---|---:|---:|
| bytes/token, bf16 | 3441.1 MB | 100% |
| one decoder layer | 100.66 MB | 2.93% |
| ... its attention sublayer | 25.17 MB | 0.73% |
| ... its MLP sublayer | 75.50 MB | 2.19% |
| all 28 layers | 2818.6 MB | **81.9%** |
| tied `embed_tokens` read as `lm_head` | 622.5 MB | **18.1%** |

**The 18.1% is the ceiling on this whole probe.** Even skipping all 28 decoder
layers only removes 81.9% of bytes/token, because the output projection is read
in full for every token no matter how shallow the stack is. Depth is not a
lever on that part of the budget.

Useful conversions to have in front of you while reading results:

| layers skipped | bytes/token cut | roofline ceiling |
|---:|---:|---:|
| 1 | 2.93% | 1.030x |
| 2 | 5.85% | 1.062x |
| 3 | 8.78% | 1.096x |
| **4** | **11.70%** | 1.133x |
| 6 | 17.55% | 1.213x |
| 8 | 23.40% | 1.306x |

Also confirm the reference is on the expected prompt set:

```bash
$PY - <<'PY'
import json
r = json.load(open('reference.json'))
print('prompts_sha256', r['prompts_sha256'])
print('harness_fingerprint', r['harness_fingerprint'])
print('total continuation positions',
      sum(len(s['ids']) - s['prompt_len'] for s in r['sequences']))
PY
```

**CHECKPOINT: `prompts_sha256` must be
`0fd801b8d2b690a0791e432e237c5c96d987795d7227c0e1392407702e5fb30c`.**
That is the value recorded in Phase 3's `config.prompts_sha256`. The script
aborts on mismatch, but check it with your own eyes once.

### Single-shot alternative

Sections 2-4 can run as one command, which applies both stop-losses
automatically and saves the ~40 s model reload each separate stage pays:

```bash
$PY skip_probe.py all --model-dir $M --device cuda 2>&1 | tee logs_all.txt
```

**Prefer the stage-by-stage path below the first time**, so you see each
checkpoint and can stop on a surprise. Use `all` for a re-run after a crash:
it resumes from `results.jsonl` and re-measures nothing.

---

## 3. Stage 1: the depth-sensitivity map (28 evals, ~5 min)

Skip each layer alone. This is the whole probe's foundation: every later stage
picks its candidates from this ranking, and the stop-loss is read off it.

```bash
$PY skip_probe.py stage1 --model-dir $M --device cuda 2>&1 | tee logs_stage1.txt
```

Each line looks like:

```
[skip] EVAL 1-single   full[13]    KL=0.0412 p95=0.0910 top1=98.21% bytes-=2.9% (3.1s)
```

**CHECKPOINT: 28 rows in `results.jsonl`, plus the anchor:**

```bash
wc -l results.jsonl                       # expect 29
grep -c '"stage": "1-single"' results.jsonl   # expect 28
sed -n '/Depth-sensitivity map/,/^$/p' results.md
```

**Expect first and last layers to be load-bearing** (literature consensus:
layer 0 builds the representation, the final layers write the output
distribution). We measure rather than assume; if L0 or L27 turn out cheap,
that is itself the finding.

### STOP-LOSS #1

Read the `STAGE 1 best single skip:` line.

* **best single-layer KL > 0.4** -> **STOP HERE.** Depth is load-bearing in
  Qwen3-1.7B: if removing *one* layer already costs more than our worst seeded
  variant, no multi-layer pattern will be competitive. Skip to section 6.
  This is a real result, not a failure: write it into the table and close the
  line of enquiry.
* **best single-layer KL <= 0.4** -> continue to stage 2.

---

## 4. Stages 2-5: the pattern sweep (~32 evals, ~10 min)

Each stage reads the sensitivity map out of `results.jsonl` and refuses to run
if stage 1 is incomplete. Run them in order.

```bash
# 2 -- contiguous runs of k = 2,3,4,6; top-3 candidates each by summed
#      single-layer KL (a first-order proxy, which is exactly why we measure)
$PY skip_probe.py stage2 --model-dir $M --device cuda 2>&1 | tee logs_stage2.txt
```

**CHECKPOINT: 12 new rows; note the `STAGE 2 best 4-layer contiguous:` line.**

### STOP-LOSS #2

* **best 4-layer contiguous KL > 1.0** -> **skip stages 3-5**, go to section 6.
  Contiguous depth removal at the target width is already ~2.3x worse than our
  worst variant; the non-contiguous and sublayer arms will not rescue it.
* otherwise continue.

```bash
# 3 -- non-contiguous growth curve: skip the n least-sensitive layers for
#      n = 2..9, re-measuring at every step because sensitivities interact
$PY skip_probe.py stage3 --model-dir $M --device cuda 2>&1 | tee logs_stage3.txt

# 4 -- attention-only vs MLP-only on the 4 least-sensitive layers.
#      On Qwen3-1.7B the MLP is 75% of a layer's bytes and attention 25%, so
#     mlp-only is the better lever IF it costs less than 3x an attn-only skip.
$PY skip_probe.py stage4 --model-dir $M --device cuda 2>&1 | tee logs_stage4.txt

# 5 -- loop probe (PoLar's "repeat a layer" arm)
$PY skip_probe.py stage5 --model-dir $M --device cuda 2>&1 | tee logs_stage5.txt
```

**CHECKPOINT after stage 3**: some rows may print `SKIP ... already in
results.jsonl`. That is correct and free: stage 3's larger sets can coincide
with stage 2's, and resume dedupes them by `pattern_id`.

**How to read stage 5, so as not to misquote it.** A loop re-executes an
already-resident layer in a skipped slot. It therefore **re-reads that layer's
~50 MB**, so on the *bandwidth* ledger (`bytes_saved_pct`) a full loop buys back
**nothing**; expect `bytes-=0.0%`. What it does cut is the *footprint* ledger
(`unique_bytes_saved_pct`), because the looped-over layer's weights are never
needed and can be dropped at load time. Stage 5 is therefore not a tok/s
result; it answers one narrow scientific question: **is the KL damage from
removing depth about the missing WEIGHTS, or about the missing DEPTH?** If
loops recover most of the KL of the equivalent plain skip, it is the weights,
and a predictor has something to work with. If they do not, depth itself is the
constraint.

---

## 5. The decision rule

Render and read the verdict block, which applies the rule mechanically:

```bash
$PY skip_probe.py render
sed -n '/^## Verdict/,/^## All measurements/p' results.md
```

| outcome | criterion | action |
|---|---|---|
| **PASS** | some pattern reaches **>=4 layer-equivalents** at **KL <= 0.205** | the predictor phase (PoLar) **earns a spec**. Read [NOTES-polar.md](NOTES-polar.md) first; it changes what that spec has to prove. |
| **MARGINAL** | best q3km-competitive pattern is **2-3 layer-equivalents** at KL <= 0.205 | thin alone, but **composes with quantization** (a skip and a quantizer cut different bytes). Report and hold; do not fund the predictor on this alone. |
| **NEGATIVE** | **nothing beats KL 0.4** at any useful depth | depth is load-bearing in Qwen3-1.7B. Write the negative result into the table and **close the line**. This is the cheapest good outcome. |

### A clarification of the brief's PASS gate, applied here

The brief states the PASS gate as ">=4 layers skipped (>=14% bytes)".
Those two clauses are not the same threshold. **4 of 28 layers is 14.3% of the
depth but only 11.70% of bytes/token**, because the tied embedding/`lm_head` is
18.1% of every token's read and is never skippable (section 2's table). Reaching
14% of *bytes* would need 4.8 -> **5** layers.

`skip_probe.py` implements the operative clause, **>=4 layer-equivalents**,
and reports the true byte percentage next to it, so nothing is hidden. If 14%
of bytes is taken as the real bar, the gate is 5 layers, and the verdict
block's PASS row should be re-read with that in mind. Say which one you used
in the write-up.

Scale note for the write-up: q3km buys its KL 0.205 with a ~21% byte cut. A
4-layer skip at KL 0.205 would buy 11.7%, i.e. **worse bytes for the same KL**.
So even a PASS here is not a win *against* quantization; it is only interesting
because a skip and a quantizer cut **different** bytes and therefore **stack**.
The write-up must say that explicitly rather than presenting depth as a rival
to q3km.

---

## 6. Spot-check and collect results (~3 min)

One qualitative row you can eyeball against known behaviour: generate the
160-token harness under the best pattern and see whether the model still
answers like itself. Substitute the winning layer list from `results.md`:

```bash
$PY skip_probe.py spot --model-dir $M --device cuda \
    --skip 20,21,22,23 --harness 2>&1 | tee logs_spot.txt
sed -n '/Spot-check harness output/,$p' results.md | head -40
```

**CHECKPOINT: eyeball the answers.** "Canberra", "Frank Herbert",
"ACKNOWLEDGED", valid JSON. Strict-format and long-tail-knowledge prompts break
first; if those
are gone while KL still looks small, say so in the write-up: KL is a mean and
can hide a mode collapse on 2 of 12 prompts. Compare against
`reference.json -> harness_texts` for the bf16 originals.

Then copy `results.jsonl`, `results.md`, `reference.json`, and the `logs_*.txt`
files off the host, and confirm the row count arrived intact before tearing
the host down.

---

## 7. Failure modes and what to do

| symptom | cause | fix |
|---|---|---|
| `PROTOCOL DRIFT: prompts_sha256() = ...` | the 12-prompt block in `skip_probe.py` was edited | restore it from [../code/swap_eval.py](../code/swap_eval.py) (lines 63-85). Do not "fix" the expected hash. |
| `ANCHOR FAIL: skip-nothing mean_kl = 1.2e-08` | non-determinism between the reference pass and the eval pass (TF32, a different attn kernel, a second process on the GPU) | rerun `anchor --force-reference` on an idle GPU. Do not proceed with a non-zero anchor. |
| `positions=` anything but 726 | the greedy continuations differ from Phase 3 (different transformers version changing the chat template or sampling path) | record it loudly in the write-up; the rows are still internally consistent (same-stack rule) but may not be compared to the q3km/rtn4 numbers. |
| `stage N incomplete (k/28 single-layer rows)` | stage 1 was interrupted | just re-run `stage1`; resume skips what is already on disk. |
| host dies mid-sweep | (any) | re-run the same stage command. `results.jsonl` is fsync'd per row, so at most the in-flight eval is lost. |
| eval wall time much more than ~5 s/row | the fp32 reference logits (~1.5 GB) are being paged, or the GPU is shared | check `nvidia-smi` and free RAM; the logits are held resident in CPU RAM by design. |
| CUDA OOM | something else is resident on the card | one bf16 1.7B model is ~3.5 GB; a 4090 has 24. Kill stale processes. |

---

## 8. Rules that did not change

- This probe replicates the eval protocol into its own file (cited by line in
  the `skip_probe.py` docstring) and imports nothing from the main harness.
- **Same-stack discipline.** Every row this probe reports is computed by
  `skip_probe.py` on one host, including the bf16 reference and the
  skip-nothing anchor. The q3km / rtn4 / 0.436 numbers appear **only** as
  annotations on the chart and are never merged into the results table.
- No training, no fine-tuning, no MoE, no quality benchmarks beyond KL and
  top-1. tok/s implications are computed analytically from bytes saved
  (`results.md -> Byte accounting`), not measured.
