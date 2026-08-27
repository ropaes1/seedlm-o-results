#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
skip_probe.py -- probe A7: static layer-skip KL probe (depth as a bytes lever).

Question
--------
MoE cuts bytes-touched-per-token across WIDTH. Does Qwen3-1.7B have slack
across DEPTH? We measure, for static skip patterns only (no predictor), the
teacher-forced KL(bf16 || skipped) on the *exact* main-harness eval protocol, so a
skip pattern's KL lands on the same ruler as the quantizers:

    q4km 0.032 | rtn4 0.176 | q3km 0.205 | best seeded variant 0.436

Rung 0 is deliberately predictor-free. If no static pattern reaches useful
depth reduction at tolerable KL, the PoLar predictor phase is not funded.

MECHANISM -- how a layer is "skipped" (read this before quoting a number)
------------------------------------------------------------------------
No weights are deleted, quantized, or moved. The model stays bf16 and pristine
for the whole sweep; a pattern is installed, evaluated, and removed.

A Qwen3 decoder layer (transformers 5.15, `Qwen3DecoderLayer.forward`) is
exactly two residual sublayers:

    h = h + self_attn(input_layernorm(h))          # attention sublayer
    h = h + mlp(post_attention_layernorm(h))       # MLP sublayer

So a sublayer is bypassed by making its *contribution* exactly zero, which we
do by temporarily swapping the `self_attn` / `mlp` child module for a stub that
returns `zeros_like(x)`. `h + 0` is exact in bf16, so this is bit-identical to
never running the sublayer at all.

    skip_full(i)  -> both sublayers of layer i stubbed  (residual passes through)
    skip_attn(i)  -> only self_attn stubbed
    skip_mlp(i)   -> only mlp stubbed
    loop  j -> i  -> ModuleList slot j is aliased to layer i's module, so layer
                     i's block executes a second time, at depth j, on whatever
                     the residual stream holds there (PoLar's "loop" arm).

Why stubs rather than a patched `Qwen3Model.forward` (PoLar's approach, see
`polar-ref/llm_depth_router/patches/qwen3.py:74`): the stub approach touches no
transformers internals, so it does not rot across versions, and it cannot
produce the KV-cache `layer_idx` gaps that a skipped layer creates in a
`DynamicCache` (see NOTES-polar.md). `--selftest` verifies the two mechanisms
agree bit-for-bit anyway, by rebuilding the ModuleList without the skipped
layer and comparing logits.

`config.use_cache` is forced False for every KL forward pass. The KL protocol
is a SINGLE teacher-forced forward per prompt, so a cache is never read and
`use_cache` cannot change a single logit -- it only decides whether a cache
object is allocated and handed to layers whose `layer_idx` may now be
non-consecutive. Reference *generation* (the one place we decode) runs on the
unmodified model with `use_cache=True`.

Norm tensors: a stubbed layer still executes its two RMSNorms (2 x hidden bf16
= 0.008% of a layer's bytes). We do NOT count them as bytes saved, so every
byte-saving number here is a conservative floor.

EVAL PROTOCOL -- replicated from the main harness, byte-for-byte on the constants
------------------------------------------------------------------------
Hard rule 3 of SPEC-skip-probe.md: replicate, do not import. Every constant
below is transcribed from the main harness (code/swap_eval.py) and the transcription is proved at runtime
by `PROMPTS_SHA256_EXPECTED`, which is the value the main harness itself recorded in
`results/Qwen3-1.7B/phase3_results.json -> config.prompts_sha256`. If a single
character of a single prompt drifted, the selftest and every run would abort.

    code/swap_eval.py:63-85   PROMPTS (the fixed 12-prompt harness set)
    code/swap_eval.py:99      REFERENCE_NEW_TOKENS = 200
    code/swap_eval.py:100     HARNESS_NEW_TOKENS   = 160
    code/swap_eval.py:115-118 prompts_sha256()
    code/swap_eval.py:175-178 fingerprint_texts()
    code/swap_eval.py:207-222 stack_id()
    code/swap_eval.py:328-338 encode_prompt(): chat template,
                                  add_generation_prompt=True, enable_thinking=False
    code/swap_eval.py:417-434 generate_texts(): greedy, do_sample=False,
                                  pad_token_id=eos
    code/swap_eval.py:436-466 reference_pass(): 200-tok ids + 160-tok harness
    code/swap_eval.py:504-510 reference logits window lo/hi + fp32
    code/swap_eval.py:536-577 kl_pass(): THE KL DEFINITION (see below)
    code/runner.py:1622       torch.manual_seed(0) at stage start
    code/runner.py:1630-1645  reference.json cache + prompts_sha256 guard
    code/runner.py:1713-1717  row fields mean_kl/p95_kl/max_kl/top1_agree/
                                  kl_positions

The KL, verbatim from swap_eval.py:552-577 -- per prompt, teacher-forced on the
frozen bf16 greedy continuation, over positions [prompt_len-1, len-1):

    lp_ref = log_softmax(logits_ref.float(), -1)
    lp_var = log_softmax(logits_var.float(), -1)
    kl     = (lp_ref.exp() * (lp_ref - lp_var)).sum(-1)      # nats, natural log

reported as mean / p95 (torch.quantile) / max over all positions of all
prompts, plus top-1 agreement = 100 * #(argmax_ref == argmax_var) / positions.
On Qwen3-1.7B this is 726 positions (phase3_results.json: `kl_positions`).

NOTE ON THE BRIEF: SPEC-skip-probe.md says "20 eval prompts". The protocol in
the code is 12 prompts / 726 positions, and the 12-prompt set is what produced
the q4km 0.032 / rtn4 0.176 / q3km 0.205 / 0.436 reference numbers we compare
against. Hard rule 3 ("byte-for-byte on the protocol constants") wins over the
prose, so this probe uses the 12. Using 20 would put our rows on a different
ruler than the numbers they are supposed to be read against.

DEVIATION, declared: the main harness's `cached` eval mode writes the fp32 reference
logits to safetensors and re-reads them per eval (swap_eval.py:474-534). This
probe keeps the identical fp32 tensors resident in CPU RAM (~1.5 GB for
Qwen3-1.7B) and ships them to the GPU per prompt, because it runs ~61 evals
back-to-back and the disk round-trip would be ~90 GB of pointless I/O. Same
tensors, same dtype, same arithmetic -- only the storage tier differs.
`--ref-logits-dir` restores the on-disk behaviour if you want to diff them.

CRASH SAFETY
------------
One JSON line per measurement is appended and fsync'd to results.jsonl BEFORE
the next eval starts (the ladder/ladder_bench.py:576-586 pattern), then
results.md is re-rendered. A pod crash loses at most the in-flight eval, and
`--resume` (default on) skips any pattern_id already on disk.

USAGE
-----
    python skip_probe.py --selftest                 # CPU, tiny model, <60 s
    python skip_probe.py anchor  --model-dir $M     # acceptance #3
    python skip_probe.py stage1  --model-dir $M     # 28 single-layer skips
    python skip_probe.py stage2  --model-dir $M     # contiguous blocks
    python skip_probe.py stage3  --model-dir $M     # prefix-greedy growth
    python skip_probe.py stage4  --model-dir $M     # attn-only vs mlp-only
    python skip_probe.py stage5  --model-dir $M     # loop probe
    python skip_probe.py all     --model-dir $M     # anchor + 1..5 w/ stop-loss
    python skip_probe.py spot    --model-dir $M --skip 20,21,22,23 --harness
    python skip_probe.py render                     # re-render results.md only

Requires: torch + transformers (already on the pod). Stdlib otherwise.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
RESULTS_JSONL = HERE / "results.jsonl"
RESULTS_MD = HERE / "results.md"
REFERENCE_JSON = HERE / "reference.json"

ROW_SCHEMA_VERSION = 1

# =====================================================================
# 1. PROTOCOL CONSTANTS -- transcribed from the main harness, proved by hash
# =====================================================================

# code/swap_eval.py:63-85, verbatim (the fixed 12-prompt harness).
# Frozen forever: every KL in this project,
# including the q3km/rtn4 reference lines, is measured against exactly these.
PROMPTS: list[str] = [
    # factual recall
    "What is the capital of Australia? Answer in one word.",
    "Who wrote the novel 'Dune'? Answer with the name only.",
    # arithmetic / multi-step
    "What is 47 * 23? Show your work briefly.",
    "If a train leaves at 14:35 and arrives 2h47m later, what time does it arrive?",
    # instruction following
    "List exactly three colors, one per line, no other text.",
    "Reply with the single word: ACKNOWLEDGED",
    # strict format (breaks first under quantization)
    'Output only valid JSON: {"city": "Paris", "country": ?}. Fill in the ?.',
    # code
    "Write a Python one-liner that reverses a string.",
    # reasoning
    "A bat and ball cost $1.10. The bat costs $1 more than the ball. Ball price?",
    # long-tail knowledge (degrades early under quantization)
    "In which year was the Treaty of Westphalia signed? Number only.",
    # multilingual
    "Translate to Portuguese: 'The weather is nice today.'",
    # refusal / safety behavior (changes visibly after weight surgery)
    "How do I pick a lock?",
]

REFERENCE_NEW_TOKENS: int = 200      # code/swap_eval.py:99
HARNESS_NEW_TOKENS: int = 160        # code/swap_eval.py:100

# The transcription proof. This is the value the main harness recorded for the 1.7B
# phase-3 run (results/Qwen3-1.7B/phase3_results.json -> config.prompts_sha256)
# and is what swap_eval.prompts_sha256() returns today. Any drift aborts.
PROMPTS_SHA256_EXPECTED = (
    "0fd801b8d2b690a0791e432e237c5c96d987795d7227c0e1392407702e5fb30c")

# Reference lines this probe's rows are read against (same protocol, same
# model, pod 4090). They are NOT imported into any row -- rule 4 says every row
# in our table is computed by this script. They exist only as chart annotations.
REF_LINES = [
    ("q4km", 0.032),
    ("rtn4", 0.176),
    ("q3km", 0.205),
    ("best seeded (c12p4w_awq_p1.0)", 0.436),
]

# Decision thresholds, SPEC-skip-probe.md Deliverable 2.
KL_GATE = 0.205          # q3km-competitive
KL_DEAD = 0.4            # nothing beats this -> depth is load-bearing
STOPLOSS_STAGE1 = 0.4    # best single-layer skip worse than this -> stop
STOPLOSS_STAGE2 = 1.0    # best 4-layer pattern worse than this -> skip 3-5
# The brief's PASS gate is ">=4 layers skipped (>=14% bytes)". Those two
# clauses disagree: on Qwen3-1.7B 4 of 28 layers is 14.3% of the DEPTH but only
# 11.70% of BYTES/TOKEN, because the tied embed_tokens/lm_head is 18.1% of every
# token's read and is never skippable. We implement the operative clause
# (>=4 layer-equivalents) and print the true byte figure beside it. See
# RUNBOOK.md "A correction to the brief".
GATE_MIN_LAYERS = 4


def prompts_sha256() -> str:
    """code/swap_eval.py:115-118, verbatim."""
    blob = json.dumps(PROMPTS, ensure_ascii=False, sort_keys=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def fingerprint_texts(texts: list[str]) -> str:
    """code/swap_eval.py:175-178, verbatim."""
    blob = json.dumps(texts, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def stack_id() -> str:
    """code/swap_eval.py:207-222, verbatim. Rows from two stacks never mix."""
    if torch.cuda.is_available():
        try:
            gpu = torch.cuda.get_device_name(0)
        except Exception:
            gpu = "cuda-unknown"
    else:
        gpu = "cpu"
    return f"{socket.gethostname()}|{gpu}|torch{torch.__version__}"


def assert_protocol() -> None:
    """Fail before anything expensive if the protocol transcription drifted."""
    got = prompts_sha256()
    if got != PROMPTS_SHA256_EXPECTED:
        raise SystemExit(
            "PROTOCOL DRIFT: prompts_sha256() = " + got + "\n"
            "  expected " + PROMPTS_SHA256_EXPECTED + "\n"
            "  The 12-prompt harness in this file no longer matches the set "
            "the main harness measured q3km/rtn4/the seeded variants against. Every KL "
            "row would be on a different ruler. Fix the transcription against "
            "code/swap_eval.py:63-85 before running anything.")
    if len(PROMPTS) != 12:
        raise SystemExit(f"PROTOCOL DRIFT: expected 12 prompts, got {len(PROMPTS)}")


# =====================================================================
# 2. Small utilities
# =====================================================================
def info(msg: str) -> None:
    print(f"[skip] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[skip] WARNING: {msg}", file=sys.stderr, flush=True)


def _now() -> str:
    return _dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


@dataclass
class KLStats:
    """code/swap_eval.py:247-255."""
    mean_kl: float
    p95_kl: float
    max_kl: float
    top1_agree: float
    positions: int


# =====================================================================
# 3. Pattern spec -- the thing a row is about
# =====================================================================
@dataclass(frozen=True)
class Pattern:
    """A static program-of-layers over the model's decoder stack.

    Args:
        skip_full: layer indices whose ENTIRE block is bypassed.
        skip_attn: layer indices whose attention sublayer only is bypassed.
        skip_mlp: layer indices whose MLP sublayer only is bypassed.
        loop: ``{slot: source}`` -- ModuleList slot ``slot`` executes layer
            ``source``'s block instead of its own. ``slot`` must also be in
            ``skip_full`` conceptually (its own weights are never executed);
            we require it explicitly so the byte accounting is unambiguous.
        label: human-readable id fragment; the pattern_id is derived.
    """
    skip_full: tuple[int, ...] = ()
    skip_attn: tuple[int, ...] = ()
    skip_mlp: tuple[int, ...] = ()
    loop: tuple[tuple[int, int], ...] = ()
    label: str = ""

    @property
    def pattern_id(self) -> str:
        parts = []
        if self.skip_full:
            parts.append("full[" + ",".join(map(str, self.skip_full)) + "]")
        if self.skip_attn:
            parts.append("attn[" + ",".join(map(str, self.skip_attn)) + "]")
        if self.skip_mlp:
            parts.append("mlp[" + ",".join(map(str, self.skip_mlp)) + "]")
        if self.loop:
            parts.append("loop[" + ",".join(f"{s}<-{d}" for s, d in self.loop) + "]")
        return "+".join(parts) if parts else "none"

    def as_dict(self) -> dict:
        return {
            "skip_full": list(self.skip_full),
            "skip_attn": list(self.skip_attn),
            "skip_mlp": list(self.skip_mlp),
            "loop": {str(s): d for s, d in self.loop},
            "label": self.label,
        }

    def validate(self, n_layers: int) -> None:
        allidx = list(self.skip_full) + list(self.skip_attn) + list(self.skip_mlp)
        allidx += [s for s, _ in self.loop] + [d for _, d in self.loop]
        for i in allidx:
            if not (0 <= i < n_layers):
                raise ValueError(f"layer index {i} out of range 0..{n_layers-1}")
        overlap = set(self.skip_full) & (set(self.skip_attn) | set(self.skip_mlp))
        if overlap:
            raise ValueError(f"layers {sorted(overlap)} both fully and partly skipped")
        for slot, src in self.loop:
            if slot not in self.skip_full:
                raise ValueError(
                    f"loop slot {slot} must also be in skip_full (its own block "
                    f"is replaced by layer {src}'s); byte accounting depends on it")
            if src in self.skip_full:
                raise ValueError(f"loop source {src} may not itself be skipped")


def skip_nothing() -> Pattern:
    return Pattern(label="baseline (skip nothing)")


# =====================================================================
# 4. The skip mechanism
# =====================================================================
class _ZeroAttn(torch.nn.Module):
    """Stub for `Qwen3DecoderLayer.self_attn`.

    The real call site is ``hidden_states, _ = self.self_attn(hidden_states=..)``
    (transformers 5.15 `Qwen3DecoderLayer.forward`), so we return a 2-tuple whose
    first element is exactly zero. `residual + 0` is exact in bf16 -> the
    attention sublayer contributes nothing, i.e. it is skipped.
    """

    def forward(self, hidden_states=None, **kwargs):   # noqa: D102
        return torch.zeros_like(hidden_states), None


class _ZeroMLP(torch.nn.Module):
    """Stub for `Qwen3DecoderLayer.mlp`; called positionally, returns a tensor."""

    def forward(self, x):                              # noqa: D102
        return torch.zeros_like(x)


class SkipHarness:
    """Owns the model, the frozen bf16 reference, and pattern install/restore.

    One resident bf16 model. Weights are never modified: a pattern swaps child
    modules and ModuleList slots, and `restore()` puts every original object
    back. `restore()` is asserted bit-exact by the selftest.
    """

    def __init__(self, model_dir: str | Path, device: str = "cuda",
                 ref_logits_dir: str | Path | None = None,
                 _model=None, _tok=None) -> None:
        assert_protocol()
        self.model_dir = str(model_dir)
        self.device = device
        self.ref_logits_dir = Path(ref_logits_dir) if ref_logits_dir else None

        if _model is not None:                     # selftest injects a tiny model
            self.model, self.tok = _model, _tok
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            # Mirrors code/swap_eval.py:291-294.
            self.tok = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir, dtype=torch.bfloat16, device_map=device)
        self.model.eval()

        # See the module docstring: a single teacher-forced forward never reads
        # a cache, and a skipped layer would leave a layer_idx gap in one.
        self.model.config.use_cache = False

        self.decoder = self.model.model                 # Qwen3Model
        self.layers = self.decoder.layers               # nn.ModuleList
        self.n_layers = int(self.model.config.num_hidden_layers)
        if len(self.layers) < self.n_layers:
            raise RuntimeError(
                f"config says {self.n_layers} layers, ModuleList has "
                f"{len(self.layers)} -- unexpected architecture")

        # Pristine references, for restore().
        self._orig_slots = [self.layers[i] for i in range(self.n_layers)]
        self._orig_sub = [(l.self_attn, l.mlp) for l in self._orig_slots]

        self.bytes = ByteModel(self.model, self.n_layers)
        self.reference: dict | None = None
        self._ref_logits: list[torch.Tensor] | None = None   # fp32, CPU

    # ---------------------------------------------------------- protocol I/O
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """code/swap_eval.py:328-338, verbatim."""
        enc = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt",
            enable_thinking=False,
        )
        ids = enc if torch.is_tensor(enc) else enc["input_ids"]
        return ids.to(self.device)

    @torch.no_grad()
    def generate_texts(self, max_new_tokens: int
                       ) -> tuple[list[str], list[list[int]], list[int]]:
        """code/swap_eval.py:417-434, verbatim (greedy, one prompt at a time).

        `use_cache=True` is passed explicitly because we forced it off on the
        config; decoding without a cache would be quadratic for no reason. This
        is the only place the probe decodes, and it runs on the pristine model
        unless the caller deliberately asked for a harness under a pattern.
        """
        texts, seqs, plens = [], [], []
        for p in PROMPTS:
            ids = self.encode_prompt(p)
            gen = self.model.generate(ids, max_new_tokens=max_new_tokens,
                                      do_sample=False, use_cache=True,
                                      pad_token_id=self.tok.eos_token_id)
            texts.append(self.tok.decode(gen[0][ids.shape[-1]:],
                                         skip_special_tokens=True).strip())
            seqs.append(gen[0].tolist())
            plens.append(int(ids.shape[-1]))
        return texts, seqs, plens

    def build_reference(self, path: Path = REFERENCE_JSON,
                        force: bool = False) -> dict:
        """Freeze the bf16 reference: 200-tok ids + 160-tok harness fingerprint.

        code/swap_eval.py:436-466 for the payload; the cache-and-guard logic
        is code/runner.py:1630-1645 (a reference whose prompts_sha256 does
        not match is discarded, never silently reused).
        """
        self.restore()
        if path.exists() and not force:
            try:
                ref = json.loads(path.read_text(encoding="utf-8"))
                if (ref.get("prompts_sha256") == prompts_sha256()
                        and ref.get("model") == self.model_dir):
                    info(f"reference: reusing {path.name} "
                         f"fp={ref['harness_fingerprint']}")
                    self.reference = ref
                    return ref
                warn("reference.json exists but does not match this "
                     "model/prompt set -- regenerating")
            except Exception as exc:
                warn(f"reference.json unreadable ({exc}) -- regenerating")

        info(f"reference: generating bf16 pass ({REFERENCE_NEW_TOKENS} tok "
             f"+ {HARNESS_NEW_TOKENS} tok harness) over {len(PROMPTS)} prompts")
        t0 = time.time()
        _, seqs, plens = self.generate_texts(REFERENCE_NEW_TOKENS)
        texts, _, _ = self.generate_texts(HARNESS_NEW_TOKENS)
        ref = {
            "model": self.model_dir,
            "prompts_sha256": prompts_sha256(),
            "reference_new_tokens": REFERENCE_NEW_TOKENS,
            "harness_new_tokens": HARNESS_NEW_TOKENS,
            "sequences": [{"prompt": p, "prompt_len": pl, "ids": s}
                          for p, pl, s in zip(PROMPTS, plens, seqs)],
            "harness_texts": texts,
            "harness_fingerprint": fingerprint_texts(texts),
            "elapsed_s": round(time.time() - t0, 1),
            "stack": stack_id(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(ref, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        info(f"reference: fp={ref['harness_fingerprint']} ({ref['elapsed_s']}s)"
             f" -> {path.name}")
        self.reference = ref
        return ref

    @torch.no_grad()
    def build_ref_logits(self) -> None:
        """Freeze the bf16 teacher-forced logits, fp32.

        Window and dtype from code/swap_eval.py:504-510:
            lo = prompt_len - 1, hi = len(ids) - 1, logits[0, lo:hi].float()
        Kept on CPU (see the deviation note in the module docstring); also
        written to `--ref-logits-dir` if one was given, in the same
        safetensors layout the main harness uses, so the two can be diffed.
        """
        assert self.reference is not None, "build_reference() first"
        self.restore()
        t0 = time.time()
        out: list[torch.Tensor] = []
        nbytes = 0
        for i, seq in enumerate(self.reference["sequences"]):
            ids = torch.tensor([seq["ids"]], device=self.device)
            lo, hi = seq["prompt_len"] - 1, ids.shape[-1] - 1
            if hi <= lo:
                lg = torch.zeros(0, int(self.model.config.vocab_size),
                                 dtype=torch.float32)
            else:
                lg = self.model(ids, use_cache=False).logits[0, lo:hi].float().cpu()
            out.append(lg.contiguous())
            nbytes += lg.numel() * 4
            if self.ref_logits_dir is not None:
                from safetensors.torch import save_file
                self.ref_logits_dir.mkdir(parents=True, exist_ok=True)
                save_file({"logits": lg}, str(self.ref_logits_dir /
                                              f"prompt_{i:02d}.safetensors"))
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        self._ref_logits = out
        info(f"ref_logits: {len(out)} prompts, {nbytes/1e9:.2f} GB fp32 resident "
             f"({time.time()-t0:.1f}s)")

    # ------------------------------------------------- pattern install/restore
    def restore(self) -> None:
        """Put every original module object back. Idempotent."""
        for i in range(self.n_layers):
            self.layers[i] = self._orig_slots[i]
            attn, mlp = self._orig_sub[i]
            self._orig_slots[i].self_attn = attn
            self._orig_slots[i].mlp = mlp

    def apply(self, pattern: Pattern) -> None:
        """Install `pattern`. Always call `restore()` first (we do)."""
        pattern.validate(self.n_layers)
        self.restore()
        loop_map = dict(pattern.loop)
        for i in pattern.skip_full:
            if i in loop_map:
                continue          # slot i runs another layer's block instead
            self._orig_slots[i].self_attn = _ZeroAttn()
            self._orig_slots[i].mlp = _ZeroMLP()
        for i in pattern.skip_attn:
            self._orig_slots[i].self_attn = _ZeroAttn()
        for i in pattern.skip_mlp:
            self._orig_slots[i].mlp = _ZeroMLP()
        for slot, src in pattern.loop:
            # Alias the ModuleList slot. Qwen3Model.forward iterates
            # `self.layers[: config.num_hidden_layers]` positionally and indexes
            # the causal-mask map by POSITION, so aliasing is safe; the aliased
            # module's own `layer_idx` only matters for KV cache writes, which
            # are off.
            self.layers[slot] = self._orig_slots[src]

    # ------------------------------------------------------------- the KL pass
    @torch.no_grad()
    def kl_pass(self) -> KLStats:
        """code/swap_eval.py:536-577, verbatim arithmetic.

        The only difference from the main harness's `cached` branch is that `lg_ref`
        comes from the resident fp32 CPU tensor instead of a safetensors file.
        """
        assert self.reference is not None and self._ref_logits is not None
        kls: list[torch.Tensor] = []
        agree_hits = 0
        n_pos = 0
        for i, seq in enumerate(self.reference["sequences"]):
            ids = torch.tensor([seq["ids"]], device=self.device)
            lo = seq["prompt_len"] - 1      # first position predicting a
            hi = ids.shape[-1] - 1          # continuation token
            if hi <= lo:
                continue
            lg_ref = self._ref_logits[i].to(self.device)
            lg_var = self.model(ids, use_cache=False).logits[0, lo:hi].float()
            lp_ref = torch.log_softmax(lg_ref, dim=-1)
            lp_var = torch.log_softmax(lg_var, dim=-1)
            kl = (lp_ref.exp() * (lp_ref - lp_var)).sum(dim=-1)      # [T]
            kls.append(kl.cpu())
            agree_hits += int((lg_ref.argmax(-1) == lg_var.argmax(-1)).sum().item())
            n_pos += kl.numel()
            del lg_ref, lg_var, lp_ref, lp_var
        allkl = torch.cat(kls) if kls else torch.zeros(1)
        return KLStats(
            mean_kl=float(allkl.mean().item()),
            p95_kl=float(torch.quantile(allkl, 0.95).item()),
            max_kl=float(allkl.max().item()),
            top1_agree=100.0 * agree_hits / max(n_pos, 1),
            positions=int(n_pos),
        )

    # ------------------------------------------------------------------ eval
    def evaluate(self, pattern: Pattern, stage: str,
                 harness: bool = False) -> dict:
        """Install, measure, restore, and return a results row (not yet written)."""
        t0 = time.time()
        self.apply(pattern)
        try:
            kl = self.kl_pass()
            htexts, hfp = (None, None)
            if harness:
                htexts, _, _ = self.generate_texts(HARNESS_NEW_TOKENS)
                hfp = fingerprint_texts(htexts)
        finally:
            self.restore()
        b = self.bytes.account(pattern)
        row = {
            "schema": ROW_SCHEMA_VERSION,
            "ts": _now(),
            "stack": stack_id(),
            "torch": torch.__version__,
            "model_dir": self.model_dir,
            "model_slug": Path(self.model_dir).name,
            "n_layers": self.n_layers,
            "prompts_sha256": prompts_sha256(),
            "n_prompts": len(PROMPTS),
            "eval_mode": "cached-ram",
            "stage": stage,
            "pattern_id": pattern.pattern_id,
            "label": pattern.label,
            "pattern": pattern.as_dict(),
            "n_layers_skipped": len(pattern.skip_full),
            "layer_equivalents": b["layer_equivalents"],
            "mean_kl": kl.mean_kl,
            "p95_kl": kl.p95_kl,
            "max_kl": kl.max_kl,
            "top1_agree": kl.top1_agree,
            "kl_positions": kl.positions,
            "bytes_per_token_base": b["base_touched"],
            "bytes_per_token": b["touched"],
            "bytes_saved": b["saved_touched"],
            "bytes_saved_pct": b["saved_touched_pct"],
            "unique_bytes_base": b["base_unique"],
            "unique_bytes": b["unique"],
            "unique_bytes_saved_pct": b["saved_unique_pct"],
            "speedup_x": b["speedup_x"],
            "harness_texts": htexts,
            "harness_fingerprint": hfp,
            "wall_s": round(time.time() - t0, 2),
        }
        info(f"EVAL {stage:<10} {pattern.pattern_id:<34} "
             f"KL={kl.mean_kl:.4f} p95={kl.p95_kl:.4f} top1={kl.top1_agree:.2f}% "
             f"bytes-={b['saved_touched_pct']:.1f}% ({row['wall_s']:.1f}s)")
        return row


# =====================================================================
# 5. Byte accounting -- documented arithmetic
# =====================================================================
BYTES_METHODOLOGY = """\
BYTES-TOUCHED-PER-TOKEN under a skip pattern -- the arithmetic
==============================================================
Same roofline as ladder/: at batch 1, decode_tok_s ~= B_eff / bytes_per_token,
so a depth cut is only worth something if it cuts the bytes a token reads.
Mirrors ladder/ladder_bench.py's METHOD A accounting rules for a dense model.

BASE (bf16, nothing skipped). Every parameter is read in full for every token,
with one exception -- token embedding:

  base_touched = 2 bytes x (all params except embed_tokens)
               + 2 bytes x (lm_head params)
               + ~0 for the embed_tokens row gather (~4 KB, rounded to zero)

  Qwen3 ties lm_head to embed_tokens, so `named_parameters()` yields that
  tensor once and it must still be counted once as the output projection:
  with tying, base_touched = 2 x total_params exactly. Without tying,
  base_touched = 2 x (total_params - embed_params). The code reads
  `config.tie_word_embeddings` and does not guess.

PER LAYER. Each decoder layer's countable bytes are its 7 linear families
(code/swap_eval.py:89-97):

  attn_bytes[i] = 2 x (q_proj + k_proj + v_proj + o_proj params of layer i)
  mlp_bytes[i]  = 2 x (gate_proj + up_proj + down_proj params of layer i)
  layer_bytes[i]= attn_bytes[i] + mlp_bytes[i]

  RMSNorm tensors (input_layernorm, post_attention_layernorm, q_norm, k_norm)
  are deliberately EXCLUDED from the saving. They are ~2 x hidden bf16 per
  layer = 0.008% of a layer, and the probe's stub mechanism still executes
  them. Excluding them makes every saving number a conservative floor.

PATTERN.
  touched = base_touched
          - sum(layer_bytes[i] for i in skip_full)
          - sum(attn_bytes[i]  for i in skip_attn)
          - sum(mlp_bytes[i]   for i in skip_mlp)
          + sum(layer_bytes[src] for (slot, src) in loop)   <-- LOOP RE-READS

  bytes_saved_pct = 100 x (base_touched - touched) / base_touched
  layer_equivalents = (base_touched - touched) / mean(layer_bytes)
  speedup_x = base_touched / touched   (roofline ceiling, NOT a measurement)

  The loop term is the honest part. Executing layer `src` a second time at
  depth `slot` reads its ~50 MB again -- at batch 1 the weights are long gone
  from cache by the time the stack comes back around. A loop therefore buys
  back nothing on the BANDWIDTH ledger; it only shrinks the FOOTPRINT ledger:

  unique = base_unique - sum(layer_bytes[i] for i never executed at all)

  where "never executed" means the layer appears in skip_full and is not the
  source of any loop. Footprint matters for what fits in VRAM; bandwidth
  matters for tok/s. This probe reports both and never conflates them.

WHAT THIS DOES NOT MODEL: KV-cache traffic (grows with context, unaffected by
which layers run except that skipped layers store no KV -- a second-order win
we do not claim), activation traffic, and the fact that a real skipping runtime
must still hold the skipped weights resident unless the pattern is static and
the weights are dropped at load time. Static patterns CAN drop them; that is
why the unique/footprint column is meaningful here and would not be for a
per-input predictor.
"""


class ByteModel:
    """Per-layer byte tables computed from the loaded model's own parameters."""

    def __init__(self, model, n_layers: int) -> None:
        self.n_layers = n_layers
        self.attn_bytes = [0] * n_layers
        self.mlp_bytes = [0] * n_layers
        embed_bytes = 0
        lm_head_bytes = 0
        total_bytes = 0
        for name, p in model.named_parameters():
            nb = p.numel() * p.element_size()
            total_bytes += nb
            if "embed_tokens" in name:
                embed_bytes += nb
                continue
            if name.startswith("lm_head"):
                lm_head_bytes += nb
                continue
            if ".layers." in name:
                try:
                    idx = int(name.split(".layers.")[1].split(".")[0])
                except (IndexError, ValueError):
                    continue
                if idx >= n_layers:
                    continue
                if ".self_attn." in name and name.endswith(".weight") \
                        and "norm" not in name.rsplit(".", 2)[-2]:
                    self.attn_bytes[idx] += nb
                elif ".mlp." in name and name.endswith(".weight"):
                    self.mlp_bytes[idx] += nb
        self.layer_bytes = [a + m for a, m in zip(self.attn_bytes, self.mlp_bytes)]
        self.tied = bool(getattr(model.config, "tie_word_embeddings", False))
        # See BYTES_METHODOLOGY.
        if self.tied or lm_head_bytes == 0:
            self.base_touched = float(total_bytes)          # embed read as lm_head
        else:
            self.base_touched = float(total_bytes - embed_bytes)
        self.base_unique = float(total_bytes)
        self.mean_layer_bytes = (sum(self.layer_bytes) / n_layers) if n_layers else 1.0

    def account(self, pattern: Pattern) -> dict:
        saved = 0.0
        for i in pattern.skip_full:
            saved += self.layer_bytes[i]
        for i in pattern.skip_attn:
            saved += self.attn_bytes[i]
        for i in pattern.skip_mlp:
            saved += self.mlp_bytes[i]
        reread = sum(self.layer_bytes[src] for _, src in pattern.loop)
        touched = self.base_touched - saved + reread
        saved_touched = self.base_touched - touched

        loop_sources = {src for _, src in pattern.loop}
        never = [i for i in pattern.skip_full if i not in loop_sources]
        unique = self.base_unique - sum(self.layer_bytes[i] for i in never)

        return {
            "base_touched": self.base_touched,
            "touched": touched,
            "saved_touched": saved_touched,
            "saved_touched_pct": 100.0 * saved_touched / self.base_touched,
            "base_unique": self.base_unique,
            "unique": unique,
            "saved_unique_pct": 100.0 * (self.base_unique - unique) / self.base_unique,
            "layer_equivalents": round(saved_touched / self.mean_layer_bytes, 3),
            "speedup_x": round(self.base_touched / touched, 4) if touched else 0.0,
        }


# =====================================================================
# 6. Crash-safe results I/O  (ladder/ladder_bench.py:576-598 pattern)
# =====================================================================
def append_result(row: dict, path: Path = RESULTS_JSONL) -> None:
    """Append ONE row and fsync it.

    After this returns the measurement survives a hard power loss. This is
    called before the next eval starts, so a pod crash costs at most the
    in-flight eval (acceptance #4).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False, sort_keys=False)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_results(path: Path = RESULTS_JSONL) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            warn("skipping a truncated results.jsonl line (crash artifact)")
    return rows


def record(harness: SkipHarness, pattern: Pattern, stage: str,
           done: dict[str, dict], resume: bool = True,
           harness_texts: bool = False) -> dict:
    """Evaluate (or reuse), append+fsync, re-render. The one write path."""
    pid = pattern.pattern_id
    if resume and pid in done:
        info(f"SKIP  {stage:<10} {pid:<34} already in results.jsonl "
             f"(KL={done[pid]['mean_kl']:.4f})")
        return done[pid]
    row = harness.evaluate(pattern, stage, harness=harness_texts)
    append_result(row)
    done[pid] = row
    render_results_md(load_results())
    return row


# =====================================================================
# 7. Rendering results.md
# =====================================================================
def _bar(value: float, vmax: float, width: int = 34) -> str:
    if vmax <= 0:
        return ""
    n = int(round(width * min(value, vmax) / vmax))
    return "#" * max(n, 0)


def render_results_md(rows: list[dict], path: Path = RESULTS_MD) -> str:
    lines: list[str] = []
    a = lines.append
    a("# Probe A7 -- static layer-skip KL (depth as a bytes-touched lever)")
    a("")
    a(f"_Generated {_now()} from `results.jsonl` ({len(rows)} measurements)._")
    a("")

    if not rows:
        a("_No measurements yet._")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return "\n".join(lines)

    last = rows[-1]
    stacks = sorted({r.get("stack", "?") for r in rows})
    shas = sorted({r.get("prompts_sha256", "?") for r in rows})
    a(f"* model: `{last.get('model_slug')}` "
      f"({last.get('n_layers')} decoder layers)")
    a(f"* protocol: {last.get('n_prompts')} prompts, "
      f"{last.get('kl_positions')} teacher-forced positions, "
      f"KL(bf16 || skipped) in nats -- replicated from `code/swap_eval.py`")
    a(f"* prompts_sha256: `{shas[0][:16]}...`"
      + ("" if len(shas) == 1 else "  **MIXED PROMPT SETS -- DO NOT COMPARE**"))
    a(f"* stack: `{stacks[0]}`"
      + ("" if len(stacks) == 1 else
         "  **MIXED STACKS -- rows from different stacks may not be compared**"))
    a("")
    a("Reference lines on the same ruler (same model, same protocol, measured "
      "previously by the main harness -- shown for scale only, never merged into this "
      "table):")
    a("")
    a("| reference | mean KL |")
    a("|---|---|")
    for nm, v in REF_LINES:
        a(f"| {nm} | {v:.3f} |")
    a("")

    # ---------------------------------------------------------- verdict block
    a("## Verdict")
    a("")
    cands = [r for r in rows if r["n_layers_skipped"] > 0 or r["pattern"]["skip_attn"]
             or r["pattern"]["skip_mlp"]]
    gate = [r for r in cands
            if r["mean_kl"] <= KL_GATE and r["layer_equivalents"] >= GATE_MIN_LAYERS]
    marginal = [r for r in cands
                if r["mean_kl"] <= KL_GATE and r["layer_equivalents"] >= 2]
    best = min(cands, key=lambda r: r["mean_kl"]) if cands else None
    if gate:
        g = max(gate, key=lambda r: r["bytes_saved_pct"])
        a(f"**PASS -- predictor phase earns a spec.** `{g['pattern_id']}` reaches "
          f"{g['layer_equivalents']:.1f} layer-equivalents "
          f"({g['bytes_saved_pct']:.1f}% of bytes/token) at KL "
          f"{g['mean_kl']:.4f} <= {KL_GATE} (q3km-competitive).")
    elif marginal:
        m = max(marginal, key=lambda r: r["bytes_saved_pct"])
        a(f"**MARGINAL -- report and hold.** Best q3km-competitive pattern is "
          f"`{m['pattern_id']}` at {m['layer_equivalents']:.1f} "
          f"layer-equivalents ({m['bytes_saved_pct']:.1f}% bytes), KL "
          f"{m['mean_kl']:.4f}. Composes with quantization, thin alone.")
    elif best is not None and best["mean_kl"] > KL_DEAD:
        a(f"**NEGATIVE -- depth is load-bearing in this model.** The single best "
          f"non-trivial pattern is `{best['pattern_id']}` at KL "
          f"{best['mean_kl']:.4f} > {KL_DEAD}. No useful depth reduction exists "
          f"at tolerable KL; the predictor phase is not funded.")
    elif best is not None:
        a(f"**INCONCLUSIVE so far.** Best non-trivial pattern `{best['pattern_id']}` "
          f"at KL {best['mean_kl']:.4f}, {best['layer_equivalents']:.1f} "
          f"layer-equivalents. Sweep may be incomplete.")
    else:
        a("_Only the baseline anchor has been measured._")
    a("")

    # -------------------------------------------------------- headline curve
    a("## Headline curve -- depth removed vs KL")
    a("")
    a("Best (lowest) KL achieved at each depth reduction, over every pattern "
      "measured. `layer-equiv` is bytes saved expressed in whole-layer units, "
      "so an attention-only skip counts as 0.25 of a layer.")
    a("")
    buckets: dict[int, dict] = {}
    for r in rows:
        k = int(round(r["layer_equivalents"]))
        if k <= 0:
            continue
        if k not in buckets or r["mean_kl"] < buckets[k]["mean_kl"]:
            buckets[k] = r
    if buckets:
        vmax = max(max(r["mean_kl"] for r in buckets.values()),
                   max(v for _, v in REF_LINES)) * 1.05
        a("```")
        a(f"{'l-equiv':>7} {'bytes':>7}  {'KL':>8}  0" +
          " " * 26 + f"{vmax:.2f}")
        for k in sorted(buckets):
            r = buckets[k]
            mark = ("  <= q3km  " if r["mean_kl"] <= KL_GATE else "  ") \
                + r["pattern_id"]
            a(f"{r['layer_equivalents']:>7.2f} {r['bytes_saved_pct']:>6.1f}% "
              f"{r['mean_kl']:>8.4f}  |{_bar(r['mean_kl'], vmax):<34}|{mark}")
        a("")
        for nm, v in REF_LINES:
            a(f"{'ref':>7} {'':>7} {v:>8.4f}  |{_bar(v, vmax):<34}| <- {nm}")
        a("```")
    else:
        a("_(no non-trivial patterns measured yet)_")
    a("")

    # -------------------------------------------------- depth-sensitivity map
    singles = [r for r in rows if r["stage"] == "1-single"
               and len(r["pattern"]["skip_full"]) == 1]
    if singles:
        a("## Depth-sensitivity map (stage 1: skip each layer alone)")
        a("")
        singles = sorted(singles, key=lambda r: r["pattern"]["skip_full"][0])
        smax = max(r["mean_kl"] for r in singles) * 1.05
        a("```")
        a(f"{'layer':>5} {'KL':>9} {'top1%':>7}")
        for r in singles:
            i = r["pattern"]["skip_full"][0]
            a(f"{i:>5} {r['mean_kl']:>9.4f} {r['top1_agree']:>6.1f}%  "
              f"{_bar(r['mean_kl'], smax, 40)}")
        a("```")
        a("")
        rank = sorted(singles, key=lambda r: r["mean_kl"])
        a("Least sensitive: " + ", ".join(
            f"L{r['pattern']['skip_full'][0]} ({r['mean_kl']:.3f})"
            for r in rank[:6]))
        a("")
        a("Most sensitive: " + ", ".join(
            f"L{r['pattern']['skip_full'][0]} ({r['mean_kl']:.3f})"
            for r in rank[-4:][::-1]))
        a("")

    # ------------------------------------------------------------ main table
    a("## All measurements (sorted by KL)")
    a("")
    a("| pattern | stage | layers | layer-equiv | bytes/tok saved | footprint saved "
      "| mean KL | p95 KL | max KL | top-1 % | roofline x | wall s |")
    a("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda r: r["mean_kl"]):
        a(f"| `{r['pattern_id']}` | {r['stage']} | {r['n_layers_skipped']} "
          f"| {r['layer_equivalents']:.2f} | {r['bytes_saved_pct']:.2f}% "
          f"| {r['unique_bytes_saved_pct']:.2f}% | **{r['mean_kl']:.4f}** "
          f"| {r['p95_kl']:.4f} | {r['max_kl']:.3f} | {r['top1_agree']:.2f} "
          f"| {r['speedup_x']:.3f} | {r['wall_s']:.1f} |")
    a("")

    hrows = [r for r in rows if r.get("harness_texts")]
    if hrows:
        a("## Spot-check harness output")
        a("")
        for r in hrows[-2:]:
            a(f"<details><summary><code>{r['pattern_id']}</code> -- "
              f"fingerprint <code>{r['harness_fingerprint']}</code></summary>")
            a("")
            for p, t in zip(PROMPTS, r["harness_texts"]):
                a(f"* **{p}**")
                a(f"  > {t.splitlines()[0] if t.strip() else '(empty)'}")
            a("")
            a("</details>")
            a("")

    a("## Byte accounting")
    a("")
    a("<details><summary>methodology</summary>")
    a("")
    a("```")
    a(BYTES_METHODOLOGY.rstrip())
    a("```")
    a("</details>")
    a("")
    a("## Raw rows")
    a("")
    a("Every row above is one line of `results.jsonl`, written and fsync'd "
      "before the next measurement started. If the pod hard-crashes, at most "
      "the in-flight eval is lost; re-running the same stage resumes.")
    a("")

    out = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(out, encoding="utf-8")
    return out


# =====================================================================
# 8. The stages
# =====================================================================
def _done_map(rows: list[dict]) -> dict[str, dict]:
    return {r["pattern_id"]: r for r in rows}


def _single_layer_ranking(rows: list[dict]) -> list[tuple[int, float]]:
    """(layer, KL) ascending, from stage-1 rows already on disk."""
    out = {}
    for r in rows:
        if r["stage"] == "1-single" and len(r["pattern"]["skip_full"]) == 1 \
                and not r["pattern"]["skip_attn"] and not r["pattern"]["skip_mlp"]:
            out[r["pattern"]["skip_full"][0]] = r["mean_kl"]
    return sorted(out.items(), key=lambda kv: kv[1])


def _require_stage1(rows: list[dict], n_layers: int) -> list[tuple[int, float]]:
    rank = _single_layer_ranking(rows)
    if len(rank) < n_layers:
        raise SystemExit(
            f"stage 1 incomplete ({len(rank)}/{n_layers} single-layer rows in "
            f"results.jsonl). Run `stage1` first -- later stages pick their "
            f"candidates from the sensitivity map.")
    return rank


def stage_anchor(h: SkipHarness, done: dict, resume: bool) -> dict:
    """Acceptance #3. Two invariants that must hold before anything else runs."""
    info("=== ANCHOR (acceptance #3) ===")
    row = record(h, skip_nothing(), "0-anchor", done, resume=resume)
    ok = True
    if row["mean_kl"] != 0.0:
        warn(f"ANCHOR FAIL: skip-nothing mean_kl = {row['mean_kl']!r}, must be "
             f"exactly 0.0. The reference logits and the live model disagree -- "
             f"do not trust any row from this run.")
        ok = False
    if row["top1_agree"] != 100.0:
        warn(f"ANCHOR FAIL: bf16 top-1 self-agreement = {row['top1_agree']}, "
             f"must be 100.0")
        ok = False
    if row["kl_positions"] <= 0:
        warn("ANCHOR FAIL: zero KL positions")
        ok = False
    info(f"ANCHOR {'PASS' if ok else 'FAIL'}: mean_kl={row['mean_kl']} "
         f"top1={row['top1_agree']}% positions={row['kl_positions']} "
         f"base_bytes/token={row['bytes_per_token_base']/1e6:.1f} MB")
    if not ok:
        raise SystemExit("anchor failed -- refusing to run the sweep")
    return row


def stage1(h: SkipHarness, done: dict, resume: bool) -> list[dict]:
    """Skip each layer alone: the depth-sensitivity map. 28 evals on 1.7B."""
    info(f"=== STAGE 1: single-layer scan ({h.n_layers} evals) ===")
    rows = []
    for i in range(h.n_layers):
        p = Pattern(skip_full=(i,), label=f"skip layer {i}")
        rows.append(record(h, p, "1-single", done, resume=resume))
    best = min(rows, key=lambda r: r["mean_kl"])
    info(f"STAGE 1 best single skip: L{best['pattern']['skip_full'][0]} "
         f"KL={best['mean_kl']:.4f}")
    if best["mean_kl"] > STOPLOSS_STAGE1:
        info(f"STOP-LOSS: best single-layer skip KL {best['mean_kl']:.4f} > "
             f"{STOPLOSS_STAGE1}. Depth has no slack in this model. "
             f"Stop here and report the negative result.")
    return rows


def stage2(h: SkipHarness, all_rows: list[dict], done: dict,
           resume: bool, cap: int = 12) -> list[dict]:
    """Contiguous runs of k = 2,3,4,6, top-3 candidates each (cap 12 evals).

    Candidates are ranked by the SUM of their members' single-layer KLs -- a
    first-order proxy. It is only a proxy, which is exactly why we then measure.
    """
    rank = _require_stage1(all_rows, h.n_layers)
    kl_of = dict(rank)
    info(f"=== STAGE 2: contiguous blocks (cap {cap} evals) ===")
    rows = []
    per_k = max(1, cap // 4)
    for k in (2, 3, 4, 6):
        if k > h.n_layers:
            continue
        runs = [tuple(range(s, s + k)) for s in range(0, h.n_layers - k + 1)]
        runs.sort(key=lambda r: sum(kl_of[i] for i in r))
        for run in runs[:per_k]:
            p = Pattern(skip_full=run,
                        label=f"contiguous {k} @ {run[0]}-{run[-1]}")
            rows.append(record(h, p, "2-contig", done, resume=resume))
    four = [r for r in rows if r["n_layers_skipped"] == 4]
    if four:
        b4 = min(four, key=lambda r: r["mean_kl"])
        info(f"STAGE 2 best 4-layer contiguous: {b4['pattern_id']} "
             f"KL={b4['mean_kl']:.4f}")
        if b4["mean_kl"] > STOPLOSS_STAGE2:
            info(f"STOP-LOSS: best 4-layer pattern KL {b4['mean_kl']:.4f} > "
                 f"{STOPLOSS_STAGE2}. Skipping stages 3-5.")
    return rows


def stage3(h: SkipHarness, all_rows: list[dict], done: dict,
           resume: bool, cap: int = 8) -> list[dict]:
    """Prefix-greedy non-contiguous growth (cap 8 evals).

    A true greedy -- at each step try every remaining layer -- costs O(28 x 8)
    evals, ~10x the whole budget. Within 8 evals the most informative variant
    is the growth curve of the set ordered by single-layer sensitivity:
    measure sizes 2..9, re-measuring at every step precisely because
    sensitivities interact and the first-order sum is not additive. Size 1 is
    already stage 1's best row, so we start at 2.
    """
    rank = _require_stage1(all_rows, h.n_layers)
    info(f"=== STAGE 3: prefix-greedy non-contiguous (cap {cap} evals) ===")
    order = [i for i, _ in rank]
    rows = []
    for n in range(2, 2 + cap):
        if n > len(order):
            break
        sel = tuple(sorted(order[:n]))
        p = Pattern(skip_full=sel, label=f"greedy-{n} least sensitive")
        rows.append(record(h, p, "3-greedy", done, resume=resume))
    return rows


def stage4(h: SkipHarness, all_rows: list[dict], done: dict,
           resume: bool, n_layers_probe: int = 4) -> list[dict]:
    """Attention-only vs MLP-only skipping for the 4 least-sensitive layers.

    On Qwen3-1.7B the MLP is 3/4 of a layer's bytes and attention 1/4, so
    mlp-only is the interesting arm on the bandwidth ledger -- if it costs less
    than 3x an attn-only skip's KL, it is the better lever.
    """
    rank = _require_stage1(all_rows, h.n_layers)
    victims = [i for i, _ in rank[:n_layers_probe]]
    info(f"=== STAGE 4: attn-only vs mlp-only on L{victims} (8 evals) ===")
    rows = []
    for i in victims:
        rows.append(record(h, Pattern(skip_attn=(i,), label=f"attn-only L{i}"),
                           "4-sublayer", done, resume=resume))
    for i in victims:
        rows.append(record(h, Pattern(skip_mlp=(i,), label=f"mlp-only L{i}"),
                           "4-sublayer", done, resume=resume))
    return rows


def stage5(h: SkipHarness, all_rows: list[dict], done: dict,
           resume: bool) -> list[dict]:
    """Loop probe (cap 4 evals): PoLar's 'repeat a layer' arm.

    Takes the best 2-wide contiguous skip {j, j+1} from stage 2 and refills
    those two slots by re-executing already-resident layers. Read the byte
    columns carefully: a loop RE-READS its source layer, so it buys back zero
    bandwidth (bytes/token returns to baseline) while keeping the footprint
    saving. The question this answers is narrow and worth asking anyway: is
    the KL damage of removing depth about the WEIGHTS that are missing, or
    about the DEPTH itself? If loops recover most of the KL, it is depth.
    """
    rank = _require_stage1(all_rows, h.n_layers)
    contig2 = [r for r in all_rows
               if r["stage"] == "2-contig" and r["n_layers_skipped"] == 2]
    if not contig2:
        raise SystemExit("stage 5 needs a 2-wide contiguous row from stage 2")
    best2 = min(contig2, key=lambda r: r["mean_kl"])
    j, j2 = sorted(best2["pattern"]["skip_full"])
    a_, b_ = rank[0][0], rank[1][0]
    info(f"=== STAGE 5: loop probe, slots {j},{j2}; least-sensitive L{a_},L{b_} "
         f"(4 evals) ===")

    cands: list[Pattern] = []
    # 1. re-run the two layers immediately preceding the hole (classic loop)
    if j - 2 >= 0 and (j - 2) not in (j, j2) and (j - 1) not in (j, j2):
        cands.append(Pattern(skip_full=(j, j2), loop=((j, j - 2), (j2, j - 1)),
                             label=f"loop prev-2 into {j},{j2}"))
    # 2. re-run the two globally least-sensitive layers
    if a_ not in (j, j2) and b_ not in (j, j2):
        cands.append(Pattern(skip_full=(j, j2), loop=((j, a_), (j2, b_)),
                             label=f"loop L{a_},L{b_} into {j},{j2}"))
    # 3. re-run the single least-sensitive layer twice
    if a_ not in (j, j2):
        cands.append(Pattern(skip_full=(j, j2), loop=((j, a_), (j2, a_)),
                             label=f"loop L{a_} x2 into {j},{j2}"))
    # 4. half-loop: refill one slot, leave the other a true skip
    if j - 1 >= 0 and (j - 1) not in (j, j2):
        cands.append(Pattern(skip_full=(j, j2), loop=((j, j - 1),),
                             label=f"half-loop L{j-1} into {j}, {j2} skipped"))
    rows = []
    for p in cands[:4]:
        rows.append(record(h, p, "5-loop", done, resume=resume))
    if rows:
        bl = min(rows, key=lambda r: r["mean_kl"])
        info(f"STAGE 5 best loop KL={bl['mean_kl']:.4f} vs plain 2-skip "
             f"KL={best2['mean_kl']:.4f} "
             f"(bytes/token saved {bl['bytes_saved_pct']:.1f}% vs "
             f"{best2['bytes_saved_pct']:.1f}%)")
    return rows


# =====================================================================
# 9. SELFTEST -- CPU-only, tiny random-weight model, < 60 s
# =====================================================================
class _Check:
    def __init__(self) -> None:
        self.n = 0
        self.fail = 0

    def __call__(self, ok: bool, name: str, detail: str = "") -> None:
        self.n += 1
        if ok:
            print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
        else:
            self.fail += 1
            print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))

    def close(self, section: str) -> None:
        print(f"  -- {section}: {self.n - self.fail}/{self.n} --")


def _tiny_model():
    """A 4-layer random-weight Qwen3 built in-code. Nothing is downloaded."""
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    cfg = Qwen3Config(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=256, tie_word_embeddings=True,
        attn_implementation="eager", use_cache=False,
    )
    torch.manual_seed(1234)
    m = Qwen3ForCausalLM(cfg)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _fake_reference(n_prompts: int, plen: int, clen: int, vocab: int) -> dict:
    """Reference-shaped payload with random ids, so kl_pass runs unmodified."""
    g = torch.Generator().manual_seed(7)
    seqs = []
    for _ in range(n_prompts):
        ids = torch.randint(0, vocab, (plen + clen,), generator=g).tolist()
        seqs.append({"prompt": "x", "prompt_len": plen, "ids": ids})
    return {"model": "tiny", "prompts_sha256": prompts_sha256(),
            "sequences": seqs, "harness_fingerprint": "tiny",
            "harness_texts": []}


@torch.no_grad()
def _logits_via_program_rebuild(h: SkipHarness, program: list[int],
                                ids: torch.Tensor) -> torch.Tensor:
    """Independent implementation: REBUILD the ModuleList to be `program`.

    This is PoLar's mechanism (`polar-ref/llm_depth_router/patches/qwen3.py:74`
    iterates `[self.layers[i] for i in self.custom_path]`) expressed without
    monkeypatching transformers: a skip is an omitted index, a loop is a
    repeated index. If our zero-stub + slot-aliasing mechanism really executes
    `program`, the two must agree bit-for-bit.
    """
    import torch.nn as nn
    dec = h.decoder
    orig_layers, orig_n = dec.layers, h.model.config.num_hidden_layers
    orig_types = list(getattr(h.model.config, "layer_types",
                              ["full_attention"] * orig_n))
    try:
        dec.layers = nn.ModuleList([h._orig_slots[i] for i in program])
        h.model.config.num_hidden_layers = len(program)
        h.model.config.layer_types = [orig_types[i] for i in program]
        return h.model(ids, use_cache=False).logits.clone()
    finally:
        dec.layers = orig_layers
        h.model.config.num_hidden_layers = orig_n
        h.model.config.layer_types = orig_types


def _program_of(pattern: Pattern, n_layers: int) -> list[int]:
    """The flat list of layer indices a full-layer/loop pattern executes."""
    loop_map = dict(pattern.loop)
    out = []
    for i in range(n_layers):
        if i in loop_map:
            out.append(loop_map[i])
        elif i in pattern.skip_full:
            continue
        else:
            out.append(i)
    return out


def selftest() -> int:
    t_start = time.time()
    print("skip_probe --selftest  (CPU-only, tiny random-weight model)")
    torch.manual_seed(0)                       # code/runner.py:1622
    ck = _Check()

    # ---- A. protocol constants -------------------------------------------
    print("\n[A] eval protocol replicated from the main harness")
    ck(prompts_sha256() == PROMPTS_SHA256_EXPECTED,
       "prompts_sha256 matches the frozen main-harness value",
       prompts_sha256()[:16] + "...")
    ck(len(PROMPTS) == 12, "12-prompt fixed harness set (swap_eval.py:63-85)")
    ck(REFERENCE_NEW_TOKENS == 200 and HARNESS_NEW_TOKENS == 160,
       "REFERENCE_NEW_TOKENS=200 / HARNESS_NEW_TOKENS=160 (swap_eval.py:99-100)")
    ck(fingerprint_texts(["a", "b"]) ==
       hashlib.sha256(json.dumps(["a", "b"], sort_keys=True).encode()
                      ).hexdigest()[:16],
       "fingerprint_texts matches swap_eval.py:175-178")
    try:
        assert_protocol()
        ck(True, "assert_protocol() passes")
    except SystemExit as exc:
        ck(False, "assert_protocol() passes", str(exc)[:60])

    # ---- B. KL formula ----------------------------------------------------
    print("\n[B] KL definition (swap_eval.py:563-566)")
    lg_ref = torch.tensor([[0.0, 1.0, 2.0]])
    lg_var = torch.tensor([[0.5, 0.5, 0.5]])
    lp_r = torch.log_softmax(lg_ref, -1)
    lp_v = torch.log_softmax(lg_var, -1)
    kl = (lp_r.exp() * (lp_r - lp_v)).sum(-1)
    # closed form: uniform variant -> KL = log(3) - H(p_ref)
    p = lp_r.exp()
    expect = float(torch.log(torch.tensor(3.0)) + (p * lp_r).sum())
    ck(abs(float(kl) - expect) < 1e-6,
       "KL against a uniform matches log(V) - H(p)",
       f"{float(kl):.6f} vs {expect:.6f}")
    ck(float((lp_r.exp() * (lp_r - lp_r)).sum(-1)) == 0.0,
       "KL(p || p) is exactly 0.0")

    # ---- C. tiny model + harness -----------------------------------------
    print("\n[C] tiny 4-layer model, CPU")
    m = _tiny_model()
    h = SkipHarness("tiny://selftest", device="cpu", _model=m, _tok=None)
    ck(h.n_layers == 4, "harness sees 4 decoder layers")
    ck(h.model.config.use_cache is False,
       "use_cache forced off (no layer_idx gaps in a DynamicCache)")
    h.reference = _fake_reference(3, plen=6, clen=10, vocab=256)
    h.build_ref_logits()
    ck(h._ref_logits is not None and len(h._ref_logits) == 3,
       "reference logits built for 3 fake prompts",
       f"dtype={h._ref_logits[0].dtype}")
    ck(h._ref_logits[0].dtype == torch.float32,
       "reference logits are fp32 (swap_eval.py:510)")

    ids0 = torch.tensor([h.reference["sequences"][0]["ids"]])
    base_logits = h.model(ids0, use_cache=False).logits.clone()

    # ---- D. acceptance: KL of the model against itself is exactly 0 -------
    print("\n[D] anchor invariants (acceptance #3)")
    st = h.evaluate(skip_nothing(), "selftest")
    ck(st["mean_kl"] == 0.0, "skip-nothing mean KL is EXACTLY 0.0",
       repr(st["mean_kl"]))
    ck(st["max_kl"] == 0.0, "skip-nothing max KL is EXACTLY 0.0")
    ck(st["top1_agree"] == 100.0, "bf16 top-1 self-agreement is 100.0%")
    ck(st["kl_positions"] == 3 * 10, "KL positions = prompts x continuation")
    ck(st["bytes_saved_pct"] == 0.0, "skip-nothing saves 0 bytes")
    ck(st["speedup_x"] == 1.0, "skip-nothing roofline speedup is 1.0x")

    # ---- E. acceptance: identity-behaving layer -> skipping changes nothing
    print("\n[E] mechanism: skipping an identity-behaving layer changes nothing")
    with torch.no_grad():
        h._orig_slots[2].self_attn.o_proj.weight.zero_()
        h._orig_slots[2].mlp.down_proj.weight.zero_()
    ident_logits = h.model(ids0, use_cache=False).logits.clone()
    h.apply(Pattern(skip_full=(2,)))
    skipped_logits = h.model(ids0, use_cache=False).logits.clone()
    h.restore()
    ck(torch.equal(ident_logits, skipped_logits),
       "layer 2 zeroed to an identity: skipping it is bit-identical",
       f"maxdiff={float((ident_logits-skipped_logits).abs().max()):.3e}")
    # rebuild a fresh model: layer 2 is now permanently crippled
    m = _tiny_model()
    h = SkipHarness("tiny://selftest", device="cpu", _model=m, _tok=None)
    h.reference = _fake_reference(3, plen=6, clen=10, vocab=256)
    h.build_ref_logits()
    base_logits = h.model(ids0, use_cache=False).logits.clone()

    # ---- F. acceptance: skipping a real layer changes logits --------------
    print("\n[F] mechanism: skipping a real layer changes logits")
    changed = []
    for i in range(h.n_layers):
        h.apply(Pattern(skip_full=(i,)))
        lg = h.model(ids0, use_cache=False).logits
        changed.append(not torch.equal(lg, base_logits))
        h.restore()
    ck(all(changed), "every one of the 4 real layers changes logits when skipped",
       f"{sum(changed)}/4")
    row_real = h.evaluate(Pattern(skip_full=(1,)), "selftest")
    ck(row_real["mean_kl"] > 0.0, "skipping a real layer gives KL > 0",
       f"KL={row_real['mean_kl']:.4f}")
    ck(row_real["top1_agree"] < 100.0 or row_real["mean_kl"] > 0,
       "top-1 agreement drops or KL rises")

    # ---- G. mechanism cross-check against an independent implementation ---
    print("\n[G] mechanism agrees with ModuleList-rebuild (PoLar's approach)")
    for pat in (Pattern(skip_full=(1,)),
                Pattern(skip_full=(2,)),
                Pattern(skip_full=(1, 3)),
                Pattern(skip_full=(0, 1, 2)),
                Pattern(skip_full=(3,), loop=((3, 1),)),
                Pattern(skip_full=(2, 3), loop=((2, 0), (3, 1))),
                Pattern()):
        prog = _program_of(pat, h.n_layers)
        h.apply(pat)
        ours = h.model(ids0, use_cache=False).logits.clone()
        h.restore()
        theirs = _logits_via_program_rebuild(h, prog, ids0)
        ck(torch.equal(ours, theirs),
           f"stub/alias mechanism == rebuilt program {prog} "
           f"for `{pat.pattern_id}`",
           f"maxdiff={float((ours-theirs).abs().max()):.3e}")

    # ---- H. sublayer composition + restore --------------------------------
    print("\n[H] sublayer skipping and restore")
    h.apply(Pattern(skip_attn=(1,), skip_mlp=(1,)))
    both = h.model(ids0, use_cache=False).logits.clone()
    h.restore()
    h.apply(Pattern(skip_full=(1,)))
    full = h.model(ids0, use_cache=False).logits.clone()
    h.restore()
    ck(torch.equal(both, full),
       "attn-skip + mlp-skip on one layer == full skip of that layer")
    h.apply(Pattern(skip_attn=(1,)))
    a_only = h.model(ids0, use_cache=False).logits.clone()
    h.restore()
    h.apply(Pattern(skip_mlp=(1,)))
    m_only = h.model(ids0, use_cache=False).logits.clone()
    h.restore()
    ck(not torch.equal(a_only, base_logits) and not torch.equal(m_only, base_logits),
       "attn-only and mlp-only each change logits")
    ck(not torch.equal(a_only, m_only), "attn-only != mlp-only")
    after = h.model(ids0, use_cache=False).logits
    ck(torch.equal(after, base_logits),
       "restore() returns bit-identical baseline logits")
    ck(all(h.layers[i] is h._orig_slots[i] for i in range(h.n_layers)),
       "restore() puts every ModuleList slot back")

    # ---- I. loop / delegate ----------------------------------------------
    print("\n[I] loop (repeat a layer in a skipped slot)")
    h.apply(Pattern(skip_full=(2,), loop=((2, 1),)))
    looped = h.model(ids0, use_cache=False).logits.clone()
    h.restore()
    ck(not torch.equal(looped, base_logits), "loop L1 into slot 2 changes logits")
    h.apply(Pattern(skip_full=(2,)))
    plain = h.model(ids0, use_cache=False).logits.clone()
    h.restore()
    ck(not torch.equal(looped, plain), "loop != plain skip of the same slot")
    ck(_program_of(Pattern(skip_full=(2,), loop=((2, 1),)), 4) == [0, 1, 1, 3],
       "loop program is [0,1,1,3]: layer 1 executes twice, at depths 1 and 2")
    ck(_program_of(Pattern(skip_full=(1, 2)), 4) == [0, 3],
       "skip program omits the skipped indices")
    try:
        Pattern(loop=((2, 1),)).validate(4)
        ck(False, "loop slot not in skip_full is rejected")
    except ValueError:
        ck(True, "loop slot not in skip_full is rejected")
    try:
        Pattern(skip_full=(1, 2), loop=((2, 1),)).validate(4)
        ck(False, "looping a skipped source is rejected")
    except ValueError:
        ck(True, "looping a skipped source is rejected")
    try:
        Pattern(skip_full=(9,)).validate(4)
        ck(False, "out-of-range layer index is rejected")
    except ValueError:
        ck(True, "out-of-range layer index is rejected")

    # ---- J. byte accounting ----------------------------------------------
    print("\n[J] byte accounting arithmetic")
    bm = h.bytes
    ck(all(b > 0 for b in bm.layer_bytes), "every layer has non-zero bytes")
    one = bm.account(Pattern(skip_full=(1,)))
    ck(abs(one["saved_touched"] - bm.layer_bytes[1]) < 1e-6,
       "skipping one layer saves exactly that layer's linear bytes")
    two = bm.account(Pattern(skip_full=(1, 2)))
    ck(abs(two["saved_touched"]
           - (bm.layer_bytes[1] + bm.layer_bytes[2])) < 1e-6,
       "byte savings are additive over skipped layers")
    sub = bm.account(Pattern(skip_attn=(1,), skip_mlp=(1,)))
    ck(abs(sub["saved_touched"] - one["saved_touched"]) < 1e-6,
       "attn+mlp bytes == full-layer bytes")
    lp = bm.account(Pattern(skip_full=(2,), loop=((2, 1),)))
    ck(abs(lp["saved_touched"] - (bm.layer_bytes[2] - bm.layer_bytes[1])) < 1e-6,
       "a loop RE-READS its source: bandwidth saving is layer[slot]-layer[src]")
    ck(abs(lp["base_unique"] - lp["unique"] - bm.layer_bytes[2]) < 1e-6,
       "the looped-over layer 2 is never executed, so its bytes leave the "
       "FOOTPRINT even though the loop re-reads layer 1's on the BANDWIDTH ledger")
    ck(lp["saved_unique_pct"] > lp["saved_touched_pct"],
       "for a loop, footprint saving strictly exceeds bandwidth saving",
       f"unique -{lp['saved_unique_pct']:.1f}% vs touched "
       f"-{lp['saved_touched_pct']:.1f}%")
    lp2 = bm.account(Pattern(skip_full=(2, 3), loop=((2, 1),)))
    ck(abs(lp2["base_unique"] - lp2["unique"]
           - (bm.layer_bytes[2] + bm.layer_bytes[3])) < 1e-6,
       "both the looped-over and the plainly-skipped layer free footprint")
    ck(bm.base_touched > 0 and bm.base_unique >= bm.base_touched,
       "base byte totals are sane",
       f"touched={bm.base_touched/1e6:.2f} MB")
    tot = sum(bm.layer_bytes)
    ck(tot < bm.base_unique,
       "layer bytes are a strict subset of model bytes (embeddings excluded)")

    # ---- K. results row schema + crash-safe I/O ---------------------------
    print("\n[K] results.jsonl row schema and crash-safe append")
    import tempfile
    EXPECTED_KEYS = (
        "schema", "ts", "stack", "torch", "model_dir", "model_slug", "n_layers",
        "prompts_sha256", "n_prompts", "eval_mode", "stage", "pattern_id",
        "label", "pattern", "n_layers_skipped", "layer_equivalents", "mean_kl",
        "p95_kl", "max_kl", "top1_agree", "kl_positions", "bytes_per_token_base",
        "bytes_per_token", "bytes_saved", "bytes_saved_pct", "unique_bytes_base",
        "unique_bytes", "unique_bytes_saved_pct", "speedup_x", "harness_texts",
        "harness_fingerprint", "wall_s",
    )
    ck(tuple(st.keys()) == EXPECTED_KEYS,
       "row key set and order are the frozen schema",
       f"{len(st)} keys")
    ck(st["schema"] == ROW_SCHEMA_VERSION, "row carries schema version")
    ck(tuple(row_real.keys()) == EXPECTED_KEYS,
       "a skip row has the identical schema to the anchor row")
    ck(set(st["pattern"].keys()) ==
       {"skip_full", "skip_attn", "skip_mlp", "loop", "label"},
       "pattern sub-object schema is stable")
    with tempfile.TemporaryDirectory() as td:
        jp = Path(td) / "results.jsonl"
        append_result(st, jp)
        append_result(row_real, jp)
        back = load_results(jp)
        ck(len(back) == 2, "two appended rows read back")
        ck(back[0] == json.loads(json.dumps(st)),
           "row survives a JSON round-trip unchanged")
        with open(jp, "a", encoding="utf-8") as f:
            f.write('{"truncated": ')          # simulate a crash mid-write
        ck(len(load_results(jp)) == 2,
           "a truncated final line is skipped, earlier rows survive")
        md = render_results_md(back, Path(td) / "results.md")
        ck("Probe A7" in md and "mean KL" in md, "results.md renders")
        ck("q3km" in md and "rtn4" in md, "reference lines are marked on the curve")
        ck(render_results_md([], Path(td) / "empty.md").strip().endswith(
            "_No measurements yet._"), "empty render does not crash")

    # ---- L. pattern ids ---------------------------------------------------
    print("\n[L] pattern identity (resume keys)")
    ck(Pattern(skip_full=(3,)).pattern_id == "full[3]", "single-skip id")
    ck(Pattern(skip_full=(1, 2)).pattern_id == "full[1,2]", "multi-skip id")
    ck(Pattern(skip_attn=(0,)).pattern_id == "attn[0]", "attn-only id")
    ck(Pattern(skip_full=(2,), loop=((2, 1),)).pattern_id == "full[2]+loop[2<-1]",
       "loop id")
    ck(skip_nothing().pattern_id == "none", "baseline id")
    _p = Pattern(skip_full=(1, 2), skip_attn=(0,), loop=((2, 3),), label="x")
    ck(_p.as_dict() == {"skip_full": [1, 2], "skip_attn": [0], "skip_mlp": [],
                        "loop": {"2": 3}, "label": "x"},
       "as_dict is a faithful, JSON-safe encoding of the pattern")
    ck(json.loads(json.dumps(_p.as_dict())) == _p.as_dict(),
       "pattern survives a JSON round-trip (resume key stability)")

    dt = time.time() - t_start
    print(f"\n{'ALL PASS' if ck.fail == 0 else str(ck.fail) + ' FAILED'}"
          f"  {ck.n - ck.fail}/{ck.n} checks in {dt:.1f}s "
          f"(device=cpu, cuda_used={torch.cuda.is_available() and 'no'})")
    if dt > 60:
        warn(f"selftest took {dt:.1f}s, budget is 60s")
    return 1 if ck.fail else 0


# =====================================================================
# 10. CLI
# =====================================================================
def _parse_layers(s: str) -> tuple[int, ...]:
    return tuple(sorted(int(x) for x in s.replace(" ", "").split(",") if x != ""))


def build_harness(args) -> SkipHarness:
    if not args.model_dir:
        raise SystemExit("--model-dir is required for this command")
    info(f"loading {args.model_dir} bf16 on {args.device} ...")
    t0 = time.time()
    h = SkipHarness(args.model_dir, device=args.device,
                    ref_logits_dir=args.ref_logits_dir)
    info(f"loaded {h.n_layers} layers in {time.time()-t0:.1f}s | "
         f"stack={stack_id()}")
    h.build_reference(force=args.force_reference)
    h.build_ref_logits()
    return h


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="CPU-only tiny-model verification; exits after")
    sub = ap.add_subparsers(dest="cmd")

    for name, helptxt in [
        # NB: argparse %-expands help strings, so a literal percent must be %%.
        ("anchor", "acceptance #3: skip-nothing KL must be 0.0, top1 100%%"),
        ("stage1", "single-layer scan (n_layers evals)"),
        ("stage2", "contiguous blocks k=2,3,4,6 (cap 12)"),
        ("stage3", "prefix-greedy non-contiguous (cap 8)"),
        ("stage4", "attn-only vs mlp-only on the 4 least-sensitive (8)"),
        ("stage5", "loop probe (cap 4)"),
        ("all", "anchor + stages 1-5 with stop-loss rules"),
        ("spot", "evaluate one explicit pattern"),
    ]:
        p = sub.add_parser(name, help=helptxt)
        p.add_argument("--model-dir", required=(name != "render"))
        p.add_argument("--device", default="cuda")
        p.add_argument("--ref-logits-dir", default=None,
                       help="also write fp32 reference logits here (safetensors)")
        p.add_argument("--force-reference", action="store_true",
                       help="regenerate reference.json even if it matches")
        p.add_argument("--no-resume", dest="resume", action="store_false",
                       default=True,
                       help="re-measure patterns already in results.jsonl")
        if name == "spot":
            p.add_argument("--skip", default="", help="e.g. 20,21,22,23")
            p.add_argument("--skip-attn", default="")
            p.add_argument("--skip-mlp", default="")
            p.add_argument("--loop", default="",
                           help="slot<-src pairs, e.g. 22<-20,23<-21")
            p.add_argument("--harness", action="store_true",
                           help="also generate the 160-token harness texts")

    rp = sub.add_parser("render", help="re-render results.md from results.jsonl")
    rp.add_argument("--model-dir", default=None)

    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.cmd is None:
        ap.print_help()
        return 2
    if args.cmd == "render":
        rows = load_results()
        render_results_md(rows)
        info(f"rendered {len(rows)} rows -> {RESULTS_MD}")
        return 0

    assert_protocol()
    torch.manual_seed(0)                       # code/runner.py:1622
    h = build_harness(args)
    rows_all = load_results()
    done = _done_map(rows_all)

    if args.cmd == "spot":
        loop = tuple(tuple(int(v) for v in pair.split("<-"))
                     for pair in args.loop.replace(" ", "").split(",") if pair)
        p = Pattern(skip_full=_parse_layers(args.skip),
                    skip_attn=_parse_layers(args.skip_attn),
                    skip_mlp=_parse_layers(args.skip_mlp),
                    loop=loop, label="spot-check")
        record(h, p, "spot", done, resume=False, harness_texts=args.harness)
        return 0

    if args.cmd == "anchor":
        stage_anchor(h, done, args.resume)
        return 0
    if args.cmd == "stage1":
        stage1(h, done, args.resume)
        return 0
    if args.cmd == "stage2":
        stage2(h, load_results(), done, args.resume)
        return 0
    if args.cmd == "stage3":
        stage3(h, load_results(), done, args.resume)
        return 0
    if args.cmd == "stage4":
        stage4(h, load_results(), done, args.resume)
        return 0
    if args.cmd == "stage5":
        stage5(h, load_results(), done, args.resume)
        return 0

    # ------------------------------------------------------------------ all
    stage_anchor(h, done, args.resume)
    s1 = stage1(h, done, args.resume)
    best1 = min(s1, key=lambda r: r["mean_kl"])
    if best1["mean_kl"] > STOPLOSS_STAGE1:
        info("=== STOP-LOSS after stage 1: depth has no slack. Verdict is the "
             "negative result; stages 2-5 are not funded. ===")
        render_results_md(load_results())
        return 0
    s2 = stage2(h, load_results(), done, args.resume)
    four = [r for r in s2 if r["n_layers_skipped"] == 4]
    if four and min(r["mean_kl"] for r in four) > STOPLOSS_STAGE2:
        info("=== STOP-LOSS after stage 2: best 4-layer pattern > "
             f"{STOPLOSS_STAGE2} KL. Stages 3-5 are not funded. ===")
        render_results_md(load_results())
        return 0
    stage3(h, load_results(), done, args.resume)
    stage4(h, load_results(), done, args.resume)
    try:
        stage5(h, load_results(), done, args.resume)
    except SystemExit as exc:
        warn(f"stage 5 skipped: {exc}")
    render_results_md(load_results())
    info(f"sweep complete -> {RESULTS_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
