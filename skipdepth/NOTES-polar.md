# NOTES-polar: what the PoLar repo actually ships

Read from a fresh clone of `github.com/tianyi-lab/PoLar`
(shallow, single squashed commit
`30d0efd`, Thu 11 Jun 2026, no tags, no LFS, no release artifacts).

Paper: *Skip a Layer or Loop It? Learning Program-of-Layers in LLMs*, ICML 2026
Oral, arXiv:2606.06574. Preliminary version (CoLa): arXiv:2507.07996.

**Bottom line: the repo diverges from the paper in ways that matter to us.
The headline accuracy protocol is best-of-5 scored against ground truth, with a
default-on short-circuit that awards credit by matching the label file without
running the model at all. And the component that generates its only supervision
signal (the offline MCTS search) is not in the repo.** Phase 2 cannot be
"port their predictor"; it would be "rebuild their pipeline from the paper".

---

## 1. What is actually released

| component | shipped? | where |
|---|---|---|
| layer-path execution patches (Llama, Qwen2, Qwen2-MoE, **Qwen3**) | yes | `llm_depth_router/patches/*.py` |
| `PolarPredictor` network + beam decoding | yes | `polar/model.py` (147 lines) |
| supervised training loop | yes | `polar/train.py` |
| evaluation loop | yes | `polar/eval.py` |
| **MCTS program discovery** | **NO** | absent; grep for "mcts" hits only filenames and a wandb project string |
| **training data** (`merged_mcts_samples.json`) | **NO** | `.gitignore:8` excludes `data/`; repo contains zero JSON |
| **checkpoints** | **NO** | `.gitignore:9-11` excludes `*.pt/*.pth/*.ckpt` |
| baselines / prior dynamic-depth methods | no | never implemented |

The README is upfront about the first one (line 15: *"The paper uses MCTS as an
offline tool to discover valid execution programs... This release focuses on the
lightweight POLAR predictor trained from those discovered programs."*). It is
still the blocking gap: without the search you cannot produce supervision for
any model, so nothing in the repo can be trained end-to-end as released.

---

## 2. Predictor architecture

`PolarPredictor` (`polar/model.py:17-80`) is **not** conditioned on the base
model's hidden states. Its input is the **raw question text**, encoded by a
separate frozen sentence encoder:

```python
embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B"   # model.py:17
self.embedding_model = AutoModel.from_pretrained(embedding_model_name)
for p in self.embedding_model.parameters():
    p.requires_grad = False                               # model.py:27-30
```

Hardcoded as a default argument; no CLI flag overrides it.

On top of that: a learned `nn.Embedding(num_layers, d_model)` supplying one
query per layer, one `nn.MultiheadAttention` cross-attending those queries into
the question's token features, a 2-block `nn.TransformerEncoder` over the layer
axis, and two linear heads (`model.py:34-47`).

* Output is **per layer**, not per segment: `seg_logits (B, D)` (a boundary
  indicator) and `op_logits (B, D, 3)` over `{SKIP, KEEP, REPEAT}`
  (`polar/config.py:10-12`).
* Segments are **derived at decode time**, not predicted: `sigmoid(seg) >= 0.5`
  gives starts, then any run longer than `max_pack=4` is force-chopped
  (`model.py:99-127`; `max_pack=4` hardcoded at `eval.py:297` and `data.py:379`).
* **Repeat is fixed at exactly 2x**: `cnt = 1 if o == OP_REPEAT else 0`
  (`model.py:141`). There is no learned repeat count.

**Parameter budget, at defaults `d_model=256, nheads=4, n_layer_blocks=2`
(`run_polar.py:50-52`) and encoder hidden 1024:**

| part | params |
|---|---:|
| `q_proj` 1024→256 | 262,400 |
| `layer_embedding` (D=28) | 7,168 |
| `cross_attn` | 263,168 |
| 2 x `TransformerEncoderLayer` | 1,579,520 |
| `seg_head` + `op_head` | 1,028 |
| **trainable total** | **≈2.11 M** |
| **frozen encoder that must be loaded and run** | **596 M** |

That is the honest "lightweight predictor" figure: **2 M trained, 600 M
executed**. For our purposes that number is fatal on its own; see section 5.

---

## 3. Training cost

* **The base LLM is never loaded during training.** `polar/train.py` imports
  nothing from `llm_depth_router`; the only model instantiated is
  `PolarPredictor(...).cuda()` (`train.py:105`), and the optimizer filters to
  trainable params (`train.py:108`). Training is pure supervised learning (BCE
  on segment flips + CE on ops, `train.py:174-185`) over precomputed paths.
* Defaults: 20 epochs, batch 16, lr 1e-4 (`run_polar.py:41-43`); the README
  sample uses 10 / 128 / 5e-4. Train slice is samples `[0:1250]`, val
  `[1250:1500]` (`train.py:61,88-89`), ≤50 valid paths per sample.
* So predictor training itself is **cheap: minutes on one GPU**.

**The cost is entirely in the missing half.** Producing
`merged_mcts_samples.json` means MCTS over the space of layer programs, with
every rollout requiring a full generation from the base LLM to check
correctness, for 2000 problems x 5 difficulty levels x 4 models. That is the
expensive artefact, and it is the one not released.

---

## 4. Inference overhead: the part that changes the phase-2 outlook

**The evaluation is best-of-k against ground truth.** `polar/eval.py:317-362`:

```python
for actions, _score in beams[: args.top_k_paths]:
    path = actions_to_path(actions, args.original_depth)
    lookup_type = _polar_path_lookup_type(pt, final_valid_transitions,
                                          final_invalid_transitions)
    if lookup_type == "valid_cache" and trust_valid_cache:
        score = 1.0; has_correct = True          # no LLM call at all
    elif lookup_type == "invalid_cache" and trust_valid_cache:
        score = 0.0
    else:
        score = _online_eval_math_single(..., transition=path, ..., gt=gt)
        if score == 1.0: has_correct = True
...
if has_correct:
    total_correct += 1
```

Three consequences:

1. **`--top_k_paths` defaults to 5** (`run_polar.py:73`). Up to five distinct
   layer programs are executed per question and the sample counts correct if
   **any one** succeeds. There is no mechanism anywhere for choosing which of
   the five to use without already knowing the answer. The repo's own print
   statement names it honestly: `"correct found rate"` (`eval.py:380`).
2. **`--trust_valid_cache` defaults to `True`** (`run_polar.py:68`). Eval reads
   `final_valid_transitions` from **the sample being evaluated**
   (`eval.py:284-286`) and awards `score = 1.0` with no model call if a decoded
   path is in that set. The eval slice `[1500:2000]` is held out from
   *predictor* training, but the MCTS-discovered answer key for those same
   samples sits in the same JSON and is used. The repo prints a
   `via valid_cache` vs `via online LLM only` split (`eval.py:232-239`)
   precisely because this matters; any headline number must be read with that
   split in hand.
3. **No baseline is ever computed.** `evaluate_polar` never runs the full-depth
   path for comparison; it copies `initial_score` out of the data file
   (`eval.py:376`). So "improves over standard inference" is, in this code,
   best-of-5-with-answer-key versus a number someone else wrote into the JSON.

Also: **no layer-count accounting.** Nothing computes executed layers vs
`original_depth`. `estimate_path_length_from_actions` exists (`data.py:449`) but
is used only for optional decode reranking (`eval.py:304-309`), never reported.
The paper's "often while executing fewer layers" is not measurable from this
code without writing new analysis.

**Routing granularity:** one predictor call per prompt, path fixed for the whole
generation. `setup_custom_path` just sets `model.model.custom_path = list(path)`
(`llm_depth_router/model.py:117`), and the patched forward reads it every step
(`patches/qwen3.py:74`). No per-token routing, no mid-generation re-routing.

**Evaluation domain is math-only.** Supervision and eval both come from
DART-Math difficulty buckets 1-5 (`config.py:118`); correctness runs through
`EvaluatorMathBatch(dataset="math")` (`eval.py:85-98`); `--max_new_tokens`
defaults to **50** with a prompt demanding *"output ONLY the final answer...
formatted strictly as \boxed{ANSWER}"* (`eval.py:129-135`), so not even
chain-of-thought math. `dart_math/data.py` carries loaders for GSM8K,
OlympiadBench etc., but `polar/eval.py` never calls them; dead code inherited
from upstream. **There is no general-language evaluation of any kind**: no
perplexity, no MMLU, nothing. Any claim that depth routing preserves general
capability is untested in this repo. That is exactly the gap our KL probe fills.

---

## 5. What this changes about phase 2

Ordered by how much it should move the decision:

1. **The 600 M-parameter encoder makes the method byte-negative for us at
   1.7B.** Our whole thesis is bytes-touched-per-token. Qwen3-1.7B bf16 is
   3441 MB/token; one decoder layer is 100.7 MB. Running
   `Qwen3-Embedding-0.6B` once per prompt costs ~1.2 GB of reads. Amortised
   over a 200-token generation that is ~6 MB/token (**about 6% of a layer**),
   so it is survivable *if* the routing saves more than ~0.06 layers, which is
   trivially true. But at prompt granularity with short prompts and short
   generations it is not free, and the encoder must be **resident**, adding
   1.2 GB to the footprint of a 3.4 GB model. On a 24 GB GPU, irrelevant; on the
   consumer-hardware target this probe exists to serve, a 35% footprint increase
   to save some fraction of 81.9% of bytes is a real trade, not a rounding
   error. **Any phase-2 spec must price the encoder in the byte budget from
   line one.** A predictor conditioned on the base model's own layer-0 hidden
   state instead would cost ~0 extra bytes and is the obvious first
   simplification to try.
2. **Best-of-5-with-an-answer-key is not a deployable inference mode**, so the
   paper's accuracy numbers do not transfer to a single-path setting. Whatever
   PoLar's real single-path accuracy is, this repo does not report it. Treat the
   published gains as an **upper bound on an oracle**, not as an expected value.
   Our rung-0 measurement is the honest floor for the same question, and if
   rung 0 says depth has no slack, PoLar's best-of-5 numbers are not evidence
   against that; they are measuring a different thing.
3. **The supervision generator is missing**, so "adopt PoLar" is a
   from-the-paper reimplementation of an MCTS search whose cost is dominated by
   base-LLM rollouts. That is a large, GPU-expensive project, not a port. Budget
   accordingly, or design a cheaper supervision signal. Note that **our KL
   metric is a far cheaper validity oracle than their generate-and-check**: one
   teacher-forced forward per candidate path versus a full generation plus
   answer matching. If phase 2 ever happens, that substitution is the single
   biggest cost saving available.
4. **Qwen3 is supported, but only `Qwen3-8B`, and by substring matching on the
   model name.** Depth is inferred from the name, not from
   `config.num_hidden_layers` (`polar/config.py:88-102`); a local path or
   renamed directory silently breaks it or raises.
   `llm_depth_router/model.py:10-45` hard-gates to four checkpoints. Qwen3-1.7B
   would need explicit registration.
5. **The KV-cache handling for repeated layers looks broken.** All layers in a
   path receive the same `past_key_values` object (`patches/qwen3.py:82`), and
   the cache is keyed by `self_attn.layer_idx`, a property of the *module*. A
   repeated index therefore re-invokes the same module and writes to the **same
   cache slot twice in one forward pass**, appending duplicate KV entries for
   the same positions; the causal mask, computed once for the non-repeated
   length (`patches/qwen3.py:63-65`), no longer matches. Skipping layer 0
   similarly leaves cache slot 0 empty while `get_seq_length()` defaults to
   `layer_idx=0`. Nothing in these patches deduplicates, resets, or offsets the
   cache. (Structural fact from the code; the precise runtime symptom is
   inferred, not executed.) **Scrutinise any latency or accuracy number that
   involves `OP_REPEAT` before trusting it.** Our probe sidesteps this entirely
   by forcing `use_cache=False` for the single teacher-forced forward the KL
   protocol needs.
6. **`requirements.txt` pins `transformers==4.52.4`**, and it is load-bearing:
   the patches are verbatim copies of that version's `forward` internals and
   expect `layer_outputs[0]`. In transformers 5.x `Qwen3DecoderLayer.forward`
   returns a **bare tensor**, so `patches/qwen3.py` fails immediately on our
   eval stack (transformers 5.15). This is why `skip_probe.py` uses
   child-module stubs instead of a forward patch: no transformers internals,
   nothing to rot.
7. Minor: `dart_math/data.py:7` does `import datasets`, which is not in
   `requirements.txt`. Not hit on the main flow (`polar/eval.py` imports only
   `dart_math.eval`), but it means the repo was never installed clean.

---

## 6. What is genuinely good here, and worth stealing

* **The mechanism is one line.** A layer program is just
  `for layer in [self.layers[i] for i in path]` (`patches/qwen3.py:74`). Skip is
  an omitted index; loop is a repeated one. Our `--selftest` cross-checks the
  probe's stub mechanism against exactly this construction and gets bit-identical
  logits, so we know the two are the same operation.
* **The skip/keep/repeat framing is the right search space**, and "repeat"
  deserves the measurement it gets in our stage 5: it is the only operation
  that changes depth without changing which weights are resident.
* **Prompt-level routing is the right granularity** for a bandwidth argument:
  per-token routing would defeat weight residency entirely.

## 7. Licensing

The clone contains **no LICENSE file**:
verify the licence on the GitHub page before any code is copied rather than
re-derived. Nothing in `skip_probe.py` copies from it; the only relationship is
that the selftest independently reconstructs the same `custom_path` iteration to
cross-check our own mechanism, and that construction is two lines of obvious
PyTorch.
