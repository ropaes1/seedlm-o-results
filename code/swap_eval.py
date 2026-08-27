"""Model-level machinery: swap, teacher-forced KL, harness, controls.

The measurement contract:

* One reference pass over the bf16 model produces, per prompt, the greedy
  200-token continuation (token ids + text) and a 160-token harness
  fingerprint.  Everything downstream is scored against that frozen reference.
* KL is computed in fp32 from raw logits with the natural log, at every
  continuation position of every prompt.  Two ways to get the reference logits:

  - ``eval_mode="dual"`` (Phase 2): hold **both** models in VRAM (2 x 1.2 GB for
    0.6B) and run the same reference sequence through each.
  - ``eval_mode="cached"`` (Phase 3): hold **one** model, and stream the
    reference logits from ``ref_logits/prompt_NN.safetensors`` written once by
    :meth:`Evaluator.write_ref_logits`.  Stored fp32, i.e. bit-identical to what
    the dual path computes, so the two modes agree exactly — the redesign
    changes the memory profile, not the measurement.  This is
    what makes a 16 GB model evaluable on a 24 GB card.

  Both modes run the *same* arithmetic on the same tensors; only the origin of
  ``lg_ref`` differs, and that is deliberately the only branch in
  :meth:`Evaluator.kl_pass`.
* Weight swapping is in-memory only: the original target tensors live in a CPU
  copy and are copied back before the next variant.  No model directory is ever
  written unless ``--export`` is asked for, and never under ``models/originals``.

Model-agnostic: every shape is read from the loaded model's own config, so the
same evaluator serves Qwen3-0.6B / 1.7B / 8B without edits.

Determinism: greedy decode (``do_sample=False``, ``pad_token_id=eos``), chat
template with ``enable_thinking=False``, one process, one stack, no
``torch.compile``.  Comparisons are only ever made within a single run *and*
within a single stack (see ``stack_id``; the same-stack determinism rule).
"""

from __future__ import annotations

import hashlib
import json
import socket
import time
from dataclasses import dataclass
from pathlib import Path

import torch

__all__ = [
    "PROMPTS",
    "TARGET_FAMILIES",
    "CALIB_MAX_TOKENS",
    "CALIB_MIN_CHARS",
    "prompts_sha256",
    "corpus_sha256",
    "load_calib_corpus",
    "target_tensor_names",
    "fingerprint_texts",
    "stack_id",
    "Evaluator",
]

# The fixed 12-prompt harness set, verbatim.
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

# The 7 linear families that get compressed.  Everything else (embed_tokens,
# norms, q_norm/k_norm, biases) stays bf16 and is never touched.
TARGET_FAMILIES: list[tuple[str, str]] = [
    ("self_attn", "q_proj"),
    ("self_attn", "k_proj"),
    ("self_attn", "v_proj"),
    ("self_attn", "o_proj"),
    ("mlp", "gate_proj"),
    ("mlp", "up_proj"),
    ("mlp", "down_proj"),
]

REFERENCE_NEW_TOKENS: int = 200
HARNESS_NEW_TOKENS: int = 160

# --- larger calibration corpora ---
# The 12 prompts above are the *behavioural* harness and are frozen forever: KL
# is measured against them and every result in the project depends on that.
# They are also, historically, the set the AWQ activation scales were captured
# from — a separate job with no reason to use the same 12 short chat turns.  A
# calibration corpus is plain documents, tokenized raw (no chat template, which
# would prepend the same ~20 template tokens to all 256 of them and bias the
# early-layer scales toward template channels), truncated to a fixed length so
# no single long document dominates the mean.
CALIB_MAX_TOKENS: int = 256
CALIB_MIN_CHARS: int = 200


def prompts_sha256() -> str:
    """Stable hash of the prompt set — recorded in every results file."""
    blob = json.dumps(PROMPTS, ensure_ascii=False, sort_keys=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def corpus_sha256(texts: list[str]) -> str:
    """Stable hash of an ordered calibration corpus.

    Order-sensitive on purpose: the capture is a mean over tokens, so two
    corpora with the same documents in a different order give (fp-)different
    scales and must not be allowed to share a cache entry.
    """
    blob = json.dumps(list(texts), ensure_ascii=False, sort_keys=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_calib_corpus(path: str | Path, n_docs: int,
                      min_chars: int = CALIB_MIN_CHARS) -> list[str]:
    """Read the first ``n_docs`` usable paragraphs of a raw text corpus.

    Written for ``wikitext-2-raw/wiki.test.raw`` — the corpus
    ``run_phase3_pod.sh`` already downloads for the llama.cpp imatrix, so the
    seed fit and the comparators end up calibrated on the same text — but the
    rule is generic and deterministic:

    * one paragraph per line (wikitext-raw's own format), whitespace stripped;
    * section headings (``= Title =``) dropped — they are a handful of tokens
      of markup and would pull the mean toward punctuation channels;
    * paragraphs shorter than ``min_chars`` dropped, so a 256-document corpus
      is 256 real paragraphs and not 200 stubs;
    * first ``n_docs`` kept, in file order.

    Deterministic in and only in that rule: same file + same n_docs => same
    list => same :func:`corpus_sha256` => same cache entry.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the file yields fewer than ``n_docs`` usable paragraphs
            (silently calibrating on 40 documents when 256 were asked for is
            exactly the kind of quiet degradation this project bans).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"calibration corpus not found: {p}")
    out: list[str] = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if len(s) < min_chars or s.startswith("="):
            continue
        out.append(s)
        if len(out) >= n_docs:
            break
    if len(out) < n_docs:
        raise ValueError(
            f"{p} yielded only {len(out)} paragraphs of >= {min_chars} chars, "
            f"need {n_docs}; pass a bigger corpus or a smaller --calib-prompts")
    return out


def fingerprint_texts(texts: list[str]) -> str:
    """Harness fingerprint: sha256 of the sorted output list."""
    blob = json.dumps(texts, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def target_tensor_names(n_layers: int, layers_limit: int | None = None) -> list[str]:
    """Parameter names of the compressed tensors, in layer-then-family order.

    Args:
        n_layers: total decoder layers in the model.
        layers_limit: if set, restrict to the first N layers (smoke tests).

    Returns:
        List of ``model.layers.{i}.{block}.{family}.weight`` names.
    """
    hi = n_layers if layers_limit is None else min(n_layers, layers_limit)
    return [f"model.layers.{i}.{block}.{fam}.weight"
            for i in range(hi) for block, fam in TARGET_FAMILIES]


def family_of(name: str) -> str:
    """``model.layers.3.mlp.up_proj.weight`` -> ``up_proj``."""
    return name.split(".")[-2]


def layer_of(name: str) -> int:
    """``model.layers.3.mlp.up_proj.weight`` -> ``3``."""
    parts = name.split(".")
    return int(parts[parts.index("layers") + 1])


def stack_id() -> str:
    """Identity of the measurement stack: ``host|gpu|torch``.

    The same-stack determinism rule says numbers from different stacks may never be
    compared.  Phase 3 splits work across a laptop and a rented pod, so every
    results row carries this string and the gate computation refuses to run
    across more than one distinct value.
    """
    if torch.cuda.is_available():
        try:
            gpu = torch.cuda.get_device_name(0)
        except Exception:                        # driver hiccup -> still identify
            gpu = "cuda-unknown"
    else:
        gpu = "cpu"
    return f"{socket.gethostname()}|{gpu}|torch{torch.__version__}"


def assert_targets_present(available: set[str] | list[str],
                           targets: list[str]) -> None:
    """Fail fast unless every expected target tensor exists in the checkpoint.

    Architecture surprises (a family renamed, a fused QKV, a missing layer) must
    stop the run before hours of fitting, not after.

    Args:
        available: tensor names the checkpoint actually offers.
        targets: names this run intends to compress.

    Raises:
        KeyError: naming the missing tensors.
    """
    have = set(available)
    missing = [t for t in targets if t not in have]
    if missing:
        raise KeyError(
            f"checkpoint is missing {len(missing)}/{len(targets)} expected target "
            f"tensors; first few: {missing[:5]}")


@dataclass
class KLStats:
    """Teacher-forced divergence of one variant against the bf16 reference."""

    mean_kl: float
    p95_kl: float
    max_kl: float
    top1_agree: float
    positions: int


class Evaluator:
    """Owns the tokenizer, the reference model, the swap model and the originals.

    Args:
        model_dir: path to an immutable model directory (any Qwen3 size).
        device: compute device for the model(s).
        layers_limit: restrict the compressed tensor set to the first N layers.
        dual: build the swap machinery at all.  ``False`` is used by the
            act-scale capture pass, which needs one pristine model and no swaps.
        eval_mode: ``"dual"`` loads a second model so reference logits can be
            recomputed on the fly (Phase 2 behaviour, the default);
            ``"cached"`` keeps a single resident model and reads the reference
            logits from disk (Phase 3).
        ref_logits_dir: where cached reference logits live; required by
            :meth:`write_ref_logits` and by ``kl_pass`` in cached mode.

    Raises:
        KeyError: if the checkpoint does not contain every expected target
            tensor (fail fast on architecture surprises).
    """

    def __init__(self, model_dir: str | Path, device: str = "cuda",
                 layers_limit: int | None = None, dual: bool = True,
                 eval_mode: str = "dual",
                 ref_logits_dir: str | Path | None = None) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if eval_mode not in ("dual", "cached"):
            raise ValueError(f"eval_mode must be dual|cached, got {eval_mode!r}")
        self.model_dir = str(model_dir)
        self.device = device
        self.eval_mode = eval_mode
        self.ref_logits_dir = Path(ref_logits_dir) if ref_logits_dir else None
        self.tok = AutoTokenizer.from_pretrained(self.model_dir)
        self.model_ref = AutoModelForCausalLM.from_pretrained(
            self.model_dir, dtype=torch.bfloat16, device_map=device)
        self.model_ref.eval()
        self.n_layers = self.model_ref.config.num_hidden_layers
        self.targets = target_tensor_names(self.n_layers, layers_limit)
        assert_targets_present(dict(self.model_ref.named_parameters()).keys(),
                               self.targets)

        self.model_var = None
        self._var_params: dict[str, torch.nn.Parameter] = {}
        if dual:
            if eval_mode == "cached":
                # One resident model: it *is* the reference model, so every
                # reference quantity must be captured before the first swap and
                # restored after the last one.  This is the 24 GB unlock.
                self.model_var = self.model_ref
            else:
                self.model_var = AutoModelForCausalLM.from_pretrained(
                    self.model_dir, dtype=torch.bfloat16, device_map=device)
                self.model_var.eval()
            self._var_params = dict(self.model_var.named_parameters())
            # CPU copy of the pristine targets — the authoritative restore path.
            self.originals = {n: self._var_params[n].detach().to("cpu").clone()
                              for n in self.targets}
        else:
            self.originals = {}

    # ------------------------------------------------------------- helpers
    def close(self) -> None:
        """Drop both models and free VRAM (used between stages)."""
        self.model_ref = None
        self.model_var = None
        self._var_params = {}
        self.originals = {}
        torch.cuda.empty_cache()

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Chat-templated prompt ids, [1, T] on the compute device."""
        enc = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt",
            enable_thinking=False,
        )
        # transformers <5 returns a tensor, >=5 returns a BatchEncoding
        # (a UserDict, so `isinstance(enc, dict)` is False — check for a tensor).
        ids = enc if torch.is_tensor(enc) else enc["input_ids"]
        return ids.to(self.device)

    def original_tensor(self, name: str) -> torch.Tensor:
        """Pristine bf16 weights for ``name``.

        Served from the CPU copy when it exists, because in ``cached`` mode the
        reference model and the swap model are the same object and reading its
        live parameters would return whatever variant is currently swapped in.
        """
        if name in self.originals:
            return self.originals[name]
        return dict(self.model_ref.named_parameters())[name].detach()

    def encode_raw(self, text: str,
                   max_tokens: int = CALIB_MAX_TOKENS) -> torch.Tensor:
        """Plain (un-templated) token ids, truncated, [1, T] on the device.

        The calibration path, not the evaluation path: a corpus document is not
        a chat turn and templating it would prepend the same control tokens to
        every document in the corpus.
        """
        enc = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=max_tokens)
        return enc["input_ids"].to(self.device)

    # ------------------------------------------------------------ act scales
    @torch.no_grad()
    def capture_act_scales(self, prompts: list[str] | None = None,
                           chat_template: bool = True,
                           max_tokens: int = CALIB_MAX_TOKENS
                           ) -> dict[str, torch.Tensor]:
        """Mean |input| per input channel for every target linear.

        Runs the calibration texts through the bf16 model once with forward
        pre-hooks on the target ``nn.Linear`` modules.

        Args:
            prompts: calibration texts.  ``None`` (the default) means the 12
                chat prompts of :data:`PROMPTS` — bit-for-bit the historical
                behaviour, and the identity every existing ``act_scales``
                cache entry was captured under.
            chat_template: apply the chat template to each text.  True for the
                12-prompt set, False for a document corpus.
            max_tokens: truncation length when ``chat_template`` is False.

        Returns:
            ``{tensor_name: float32 tensor of shape [in_features]}``.
        """
        texts = PROMPTS if prompts is None else list(prompts)
        sums: dict[str, torch.Tensor] = {}
        counts: dict[str, int] = {}
        handles = []

        def make_hook(name: str):
            def hook(_module, args):
                x = args[0].detach().float().reshape(-1, args[0].shape[-1])
                s = x.abs().sum(dim=0)
                if name in sums:
                    sums[name] += s
                    counts[name] += x.shape[0]
                else:
                    sums[name] = s
                    counts[name] = x.shape[0]
            return hook

        for name in self.targets:
            mod = self.model_ref.get_submodule(name[: -len(".weight")])
            handles.append(mod.register_forward_pre_hook(make_hook(name)))
        try:
            for p in texts:
                self.model_ref(self.encode_prompt(p) if chat_template
                               else self.encode_raw(p, max_tokens))
        finally:
            for h in handles:
                h.remove()
        return {n: (sums[n] / max(counts[n], 1)).cpu() for n in sums}

    # -------------------------------------------------------------- passes
    @torch.no_grad()
    def generate_texts(self, model, max_new_tokens: int
                       ) -> tuple[list[str], list[list[int]], list[int]]:
        """Greedy generation over all 12 prompts.

        Returns:
            (decoded texts, full token sequences, prompt lengths).
        """
        texts, seqs, plens = [], [], []
        for p in PROMPTS:
            ids = self.encode_prompt(p)
            gen = model.generate(ids, max_new_tokens=max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=self.tok.eos_token_id)
            texts.append(self.tok.decode(gen[0][ids.shape[-1]:],
                                         skip_special_tokens=True).strip())
            seqs.append(gen[0].tolist())
            plens.append(int(ids.shape[-1]))
        return texts, seqs, plens

    def reference_pass(self, out_path: str | Path) -> dict:
        """Freeze the bf16 reference: KL sequences + harness fingerprint.

        Args:
            out_path: where ``reference.json`` is written.

        Returns:
            The reference payload dict.
        """
        t0 = time.time()
        self.restore()          # in cached mode model_ref is the swap model
        _, seqs, plens = self.generate_texts(self.model_ref, REFERENCE_NEW_TOKENS)
        texts, _, _ = self.generate_texts(self.model_ref, HARNESS_NEW_TOKENS)
        payload = {
            "model": self.model_dir,
            "prompts_sha256": prompts_sha256(),
            "reference_new_tokens": REFERENCE_NEW_TOKENS,
            "harness_new_tokens": HARNESS_NEW_TOKENS,
            "sequences": [
                {"prompt": p, "prompt_len": pl, "ids": s}
                for p, pl, s in zip(PROMPTS, plens, seqs)
            ],
            "harness_texts": texts,
            "harness_fingerprint": fingerprint_texts(texts),
            "elapsed_s": round(time.time() - t0, 1),
        }
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        return payload

    # ------------------------------------------------- cached reference logits
    def _ref_logits_path(self, i: int) -> Path:
        if self.ref_logits_dir is None:
            raise ValueError("cached eval mode needs ref_logits_dir")
        return self.ref_logits_dir / f"prompt_{i:02d}.safetensors"

    @torch.no_grad()
    def write_ref_logits(self, reference: dict, resume: bool = True) -> dict:
        """Freeze the bf16 teacher-forced logits to disk, fp32, once per model.

        Must be called while the model is pristine (in cached mode the swap
        model and the reference model are the same object).  Written per prompt
        so a killed run resumes at prompt granularity.

        Args:
            reference: the payload from :meth:`reference_pass`.
            resume: skip prompts whose file already exists.

        Returns:
            ``{"files": n, "bytes": total, "elapsed_s": s, "skipped": n}``.
        """
        from safetensors.torch import save_file

        if self.ref_logits_dir is None:
            raise ValueError("write_ref_logits needs ref_logits_dir")
        self.ref_logits_dir.mkdir(parents=True, exist_ok=True)
        self.restore()                      # paranoia: never cache swapped logits
        t0 = time.time()
        total_bytes = 0
        skipped = 0
        for i, seq in enumerate(reference["sequences"]):
            path = self._ref_logits_path(i)
            if resume and path.exists():
                skipped += 1
                total_bytes += path.stat().st_size
                continue
            ids = torch.tensor([seq["ids"]], device=self.device)
            lo, hi = seq["prompt_len"] - 1, ids.shape[-1] - 1
            if hi <= lo:
                lg = torch.zeros(0, self.model_ref.config.vocab_size,
                                 dtype=torch.float32)
            else:
                lg = self.model_ref(ids).logits[0, lo:hi].float().cpu()
            tmp = path.with_suffix(".tmp")
            save_file({"logits": lg.contiguous()}, str(tmp),
                      metadata={"meta": json.dumps(
                          {"prompt_index": i, "lo": lo, "hi": hi,
                           "prompts_sha256": prompts_sha256()})})
            tmp.replace(path)
            total_bytes += path.stat().st_size
            del lg
            torch.cuda.empty_cache()
        return {"files": len(reference["sequences"]), "bytes": total_bytes,
                "elapsed_s": round(time.time() - t0, 1), "skipped": skipped}

    def ref_logits_complete(self, reference: dict) -> bool:
        """True when every prompt's cached logits file is present."""
        if self.ref_logits_dir is None:
            return False
        return all(self._ref_logits_path(i).exists()
                   for i in range(len(reference["sequences"])))

    def _load_ref_logits(self, i: int) -> torch.Tensor:
        from safetensors import safe_open

        with safe_open(str(self._ref_logits_path(i)), framework="pt") as f:
            return f.get_tensor("logits").to(self.device)

    @torch.no_grad()
    def kl_pass(self, reference: dict) -> KLStats:
        """Teacher-forced KL(P_ref || P_var) over every continuation position.

        Reference and variant see the identical frozen reference sequence, so
        the comparison is per-position and prefix-matched.  Softmax and the KL
        sum are done in fp32 from raw logits; the log is natural.

        The only difference between ``dual`` and ``cached`` mode is where
        ``lg_ref`` comes from — a second resident model, or the fp32 tensor that
        second model wrote earlier.  Everything after that line is shared code,
        which is why the two modes agree to the last bit (AC-2).
        """
        kls: list[torch.Tensor] = []
        agree_hits = 0
        n_pos = 0
        for i, seq in enumerate(reference["sequences"]):
            ids = torch.tensor([seq["ids"]], device=self.device)
            lo = seq["prompt_len"] - 1          # first position predicting a
            hi = ids.shape[-1] - 1             # continuation token
            if hi <= lo:
                continue
            if self.eval_mode == "cached":
                lg_ref = self._load_ref_logits(i)
            else:
                lg_ref = self.model_ref(ids).logits[0, lo:hi].float()
            lg_var = self.model_var(ids).logits[0, lo:hi].float()
            lp_ref = torch.log_softmax(lg_ref, dim=-1)
            lp_var = torch.log_softmax(lg_var, dim=-1)
            kl = (lp_ref.exp() * (lp_ref - lp_var)).sum(dim=-1)   # [T]
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

    def harness_pass(self) -> tuple[list[str], str]:
        """160-token greedy harness on the variant model + harness fingerprint."""
        texts, _, _ = self.generate_texts(self.model_var, HARNESS_NEW_TOKENS)
        return texts, fingerprint_texts(texts)

    # ---------------------------------------------------------- swap / undo
    def swap_in(self, tensors: dict[str, torch.Tensor]) -> int:
        """Copy dequantized weights into the variant model in place.

        Args:
            tensors: ``{tensor_name: dequantized tensor}``; names must be a
                subset of ``self.targets``.

        Returns:
            Number of tensors swapped.
        """
        params = self._var_params
        for name, t in tensors.items():
            p = params[name]
            if tuple(t.shape) != tuple(p.shape):
                raise ValueError(f"shape mismatch for {name}: "
                                 f"{tuple(t.shape)} vs {tuple(p.shape)}")
            p.data.copy_(t.to(device=p.device, dtype=p.dtype))
        return len(tensors)

    def restore(self) -> None:
        """Restore all 196 targets from the pristine CPU copy."""
        for name, t in self.originals.items():
            p = self._var_params[name]
            p.data.copy_(t.to(p.device))

    def export(self, out_dir: str | Path) -> str:
        """Save the currently-swapped variant model as a full directory."""
        out_dir = Path(out_dir)
        if "models/originals" in out_dir.as_posix():
            raise ValueError("refusing to write inside models/originals")
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model_var.save_pretrained(out_dir)
        self.tok.save_pretrained(out_dir)
        return str(out_dir)
