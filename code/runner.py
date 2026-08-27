"""SeedLM+O experiment runner — matrix, cache/resume, gates, results.

Phase 2 and Phase 3 share this one runner.  Which
one you get is decided by a single flag:

* **no ``--model-dir``** -> *legacy layout*: exactly the Phase 2 behaviour, the
  Phase 2 variant matrix, Phase 2 gates, un-namespaced ``cache/`` and
  ``results/``, and the original ``sha1(tensor|C|P|n_seeds|rule|p|seed)`` cache
  keys.  The Phase 2 commands keep working unchanged.
* **``--model-dir M``** -> *Phase 3 layout*: ``MODEL_SLUG = basename(M)``,
  cache in ``cache/{slug}/``, results in ``results/{slug}/``, cache keys gain the
  model slug and the config name (Qwen3-1.7B has the same tensor *names* as
  Qwen3-0.6B, so without the slug the keys would collide catastrophically),
  sharded checkpoint loading, the ``c8p3``/``c12p4`` configs, the lean Phase 3
  variant matrix, real comparators, cached-reference-logits evaluation, and the
  Phase 3 gates.

    python runner.py --selftest
    python runner.py --stage fit|eval|refit-full|all --model-dir M --config c12p4
    python runner.py --stage comparators --model-dir M
    python runner.py --stage lmeval      --model-dir M          # pod only
    python runner.py --stage verdict     --model-dir M
    python runner.py --stage equivalence --model-dir M          # AC-2 harness

Every stage is idempotent and resumable: fits are cached per tensor and a rerun
prints ``SKIP cached <name>`` and recomputes nothing.  Nothing under
``models/originals/`` is ever opened for writing (every shard's sha256 is
recorded on first run and re-verified at the end of every stage).

Cross-stack safety (the same-stack determinism rule): every results
row records the ``stack`` it was measured on, and the gate computation refuses
to run over rows from more than one stack.  Laptop stage-1 numbers may pick a
winner; only pod-measured rows may be in the pod's verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import lfsr_core
import salience
import swap_eval
from lfsr_core import C_DEFAULT, GENERATOR_SEED, P_DEFAULT

# ------------------------------------------------------------------ layout
# SEEDLM_ROOT overrides the data root (cache/, results/, comparators/, default
# model dir). Needed on the pod, where the harness code lives in
# /workspace/harness/ but data must live on the volume root /workspace/ — the
# __file__-anchored default would silently write under harness/ and the pod's
# upload step would ship an empty results/ (found in review, 2026-08-16).
ROOT = Path(os.environ.get("SEEDLM_ROOT") or Path(__file__).resolve().parent)
DEFAULT_MODEL_DIR = ROOT / "models" / "originals" / "Qwen3-0.6B"

# Reconfigured by :func:`configure_layout` before anything else runs.
LAYOUT: str = "legacy"                  # "legacy" | "phase3"
MODEL_DIR: Path = DEFAULT_MODEL_DIR
MODEL_SLUG: str = DEFAULT_MODEL_DIR.name
CONFIG_NAME: str = "c8p3"
FIT_WEIGHTING: str = "none"             # "none" | "awq"  (W1)
COEFF_ROUNDING: str = "nearest"         # "nearest" | "weighted"  (W2)
INCOHERENCE: str = "none"               # "none" | "had"          (W3)

# Calibration-set identity.  The
# activation scales that drive BOTH the awq side channel and the W1/W3 fit
# weighting are a *measurement*, taken from a prompt set, and until now that
# set has silently been the fixed 12-prompt harness.  `p12` names that set.
# Any other calibration set gets its own id, its own act_scales file, its own
# cache namespace and its own results rows — the same discipline W1 applied to
# the fit objective, applied to the objective's inputs.
DEFAULT_CALIB_ID: str = "p12"
CALIB_ID: str = DEFAULT_CALIB_ID
CALIB_CORPUS: Path | None = None        # required for any non-default calib id
CALIB_N_PROMPTS: int = 256
CALIB_MAX_TOKENS: int = 256
CACHE: Path = ROOT / "cache"
RESULTS: Path = ROOT / "results"
RUN_LOG: Path = RESULTS / "run.log"
HASH_RECORD: Path = CACHE / "original_sha256.json"
ACT_SCALES: Path = CACHE / "act_scales.safetensors"
REF_LOGITS: Path = RESULTS / "ref_logits"
COMPARATOR_DIR: Path = ROOT / "comparators"
RESULTS_JSON: str = "phase2_results.json"

STAGE1_SEEDS: int = 16384
STAGE2_SEEDS: int = 65535
ETA_WARN_HOURS: float = 8.0
ETA_STALL_GAP_S: float = 300.0          # gaps > 5 min are stalls, not work

# Seed-fit configurations.  (16 seed + 4 shared-exponent + 4*P coefficient) / C.
CONFIGS: dict[str, tuple[int, int]] = {"c8p3": (8, 3), "c12p4": (12, 4)}

# Rules whose reconstruction does not depend on (C, P) — cached once per model
# rather than once per model per config.
CONFIG_FREE_RULES: frozenset[str] = frozenset({"rtn", "q4km", "q3km", "awq_w4"})
COMPARATOR_RULES: frozenset[str] = frozenset({"q4km", "q3km", "awq_w4"})

# Rules that run a seed fit and are therefore affected by --fit-weighting.
# Everything else (rtn, rtn+mag, the real comparators, bf16) is bit-for-bit the
# same object under either objective and keeps sharing one cache entry.
WEIGHTABLE_RULES: frozenset[str] = frozenset({"none", "mag", "awq", "spike"})

# Golden LFSR vector: lfsr_states([1, 42, 65535], 8), taps 0xB400 (Gate 1a).
LFSR_GOLDEN: list[list[int]] = [
    [46080, 23040, 11520, 5760, 2880, 1440, 720, 360],
    [21, 46090, 23045, 39170, 19585, 37440, 18720, 9360],
    [52223, 53759, 56575, 55935, 55615, 55455, 55375, 55335],
]


def configure_layout(model_dir: Path | None, config_name: str = "c8p3",
                     fit_weighting: str = "none",
                     coeff_rounding: str = "nearest",
                     incoherence: str = "none",
                     calib_id: str = DEFAULT_CALIB_ID,
                     calib_corpus: Path | None = None,
                     calib_n_prompts: int = CALIB_N_PROMPTS,
                     calib_max_tokens: int = CALIB_MAX_TOKENS) -> None:
    """Point every path global at the right namespace for this run.

    Args:
        model_dir: explicit model directory (Phase 3 layout), or ``None`` to
            reproduce the Phase 2 layout against the default Qwen3-0.6B.
        config_name: ``c8p3`` or ``c12p4``; part of the Phase 3 cache key.
        fit_weighting: ``none`` (plain L2, the historical objective) or ``awq``
            (activation-weighted, W1).  Weighted runs get
            their own ``w``-suffixed config slug, hence their own cache
            namespace and their own results rows.
        coeff_rounding: ``nearest`` (historical) or ``weighted`` (W2's exact
            coefficient search).  Adds an ``r`` suffix.
        incoherence: ``none`` (historical) or ``had`` (W3 design A's seeded
            orthogonal transform).  Adds an ``h`` suffix.
        calib_id: identity of the activation-scale calibration set.  ``p12``
            (the default) is the fixed 12-prompt harness every existing
            result was calibrated on, and reproduces today's paths, cache keys
            and row names byte-for-byte.  Any other id adds an ``@id`` suffix
            to the config slug of every rule that consumes activation scales,
            and sends the capture to its own ``act_scales@id.safetensors``.
        calib_corpus: text file the non-default calibration set is read from;
            required whenever ``calib_id`` is not the default.
        calib_n_prompts: documents to take from that corpus.
        calib_max_tokens: per-document truncation length.

    Side effects:
        Rebinds ``LAYOUT``, ``MODEL_DIR``, ``MODEL_SLUG``, ``CONFIG_NAME``,
        ``FIT_WEIGHTING``, ``COEFF_ROUNDING``, ``INCOHERENCE``, ``CALIB_ID``,
        ``CALIB_CORPUS``, ``CALIB_N_PROMPTS``, ``CALIB_MAX_TOKENS``, ``CACHE``,
        ``RESULTS``, ``RUN_LOG``, ``HASH_RECORD``, ``ACT_SCALES``,
        ``REF_LOGITS``, ``COMPARATOR_DIR`` and ``RESULTS_JSON``.
    """
    global LAYOUT, MODEL_DIR, MODEL_SLUG, CONFIG_NAME, CACHE, RESULTS
    global RUN_LOG, HASH_RECORD, ACT_SCALES, REF_LOGITS, COMPARATOR_DIR
    global RESULTS_JSON, FIT_WEIGHTING, COEFF_ROUNDING, INCOHERENCE
    global CALIB_ID, CALIB_CORPUS, CALIB_N_PROMPTS, CALIB_MAX_TOKENS

    if config_name not in CONFIGS:
        raise ValueError(f"unknown config {config_name!r}, want {sorted(CONFIGS)}")
    # The id ends up in file names, cache-key blobs and results-row names, and
    # `@` is its own separator, so keep it to a short unambiguous token.
    if not (calib_id and calib_id.isascii() and calib_id.isalnum()
            and calib_id.islower() and len(calib_id) <= 12):
        raise ValueError(
            f"bad --calib-id {calib_id!r}: want 1-12 lowercase alphanumeric "
            f"characters (it becomes a file name and a results-row suffix)")
    if calib_id != DEFAULT_CALIB_ID and calib_corpus is None:
        raise ValueError(
            f"--calib-id {calib_id} needs --calib-corpus: a calibration set "
            f"that is not the built-in 12 prompts has to say what it is, and "
            f"its sha256 is recorded with the captured scales")
    if fit_weighting not in lfsr_core.FIT_WEIGHTINGS:
        raise ValueError(f"unknown fit weighting {fit_weighting!r}, want "
                         f"{list(lfsr_core.FIT_WEIGHTINGS)}")
    if coeff_rounding not in lfsr_core.COEFF_ROUNDINGS:
        raise ValueError(f"unknown coeff rounding {coeff_rounding!r}, want "
                         f"{list(lfsr_core.COEFF_ROUNDINGS)}")
    if incoherence not in lfsr_core.INCOHERENCE_MODES:
        raise ValueError(f"unknown incoherence {incoherence!r}, want "
                         f"{list(lfsr_core.INCOHERENCE_MODES)}")
    # Same reason for all three: the legacy layout has no config slug to
    # namespace a non-default fit with, so its rows would collide with Phase 2's.
    for flag, val in (("--fit-weighting", fit_weighting),
                      ("--coeff-rounding", coeff_rounding != "nearest"),
                      ("--incoherence", incoherence),
                      ("--calib-id", calib_id != DEFAULT_CALIB_ID)):
        non_default = val not in ("none", "nearest", False)
        if non_default and model_dir is None:
            raise ValueError(f"{flag} needs --model-dir: the Phase 2 legacy "
                             f"layout has no config slug to namespace "
                             f"non-default fits with, so their rows would "
                             f"collide with Phase 2's")
    CONFIG_NAME = config_name
    FIT_WEIGHTING = fit_weighting
    COEFF_ROUNDING = coeff_rounding
    INCOHERENCE = incoherence
    CALIB_ID = calib_id
    CALIB_CORPUS = Path(calib_corpus) if calib_corpus is not None else None
    CALIB_N_PROMPTS = int(calib_n_prompts)
    CALIB_MAX_TOKENS = int(calib_max_tokens)
    if model_dir is None:
        LAYOUT = "legacy"
        MODEL_DIR = DEFAULT_MODEL_DIR
        MODEL_SLUG = MODEL_DIR.name
        CACHE = ROOT / "cache"
        RESULTS = ROOT / "results"
        RESULTS_JSON = "phase2_results.json"
    else:
        LAYOUT = "phase3"
        MODEL_DIR = Path(model_dir).resolve()
        MODEL_SLUG = MODEL_DIR.name
        CACHE = ROOT / "cache" / MODEL_SLUG
        RESULTS = ROOT / "results" / MODEL_SLUG
        RESULTS_JSON = "phase3_results.json"
    RUN_LOG = RESULTS / "run.log"
    HASH_RECORD = CACHE / "original_sha256.json"
    ACT_SCALES = act_scales_path()
    REF_LOGITS = RESULTS / "ref_logits"
    COMPARATOR_DIR = ROOT / "comparators" / MODEL_SLUG


# ------------------------------------------------------------ shard reading
class ShardReader:
    """Tensor-level reader over a single- or multi-shard safetensors model.

    Resolves ``tensor -> shard`` through ``model.safetensors.index.json`` when
    that file is present (Qwen3-1.7B and Qwen3-8B are sharded), otherwise falls
    back to the single ``model.safetensors``.  Shard handles are opened lazily
    and kept open, and tensors are fetched one at a time — a 16 GB shard set is
    never materialised in RAM (streaming pattern).

    Args:
        model_dir: directory holding the ``*.safetensors`` files.

    Raises:
        FileNotFoundError: if the directory holds no safetensors at all.
    """

    def __init__(self, model_dir: str | Path) -> None:
        self.dir = Path(model_dir)
        index = self.dir / "model.safetensors.index.json"
        self.index: dict[str, Path] = {}
        if index.exists():
            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            for tensor, filename in weight_map.items():
                self.index[tensor] = self.dir / filename
            self.files = sorted({p for p in self.index.values()})
        else:
            self.files = sorted(self.dir.glob("*.safetensors"))
            if not self.files:
                raise FileNotFoundError(f"no safetensors under {self.dir}")
            for path in self.files:
                with safe_open(str(path), framework="pt") as fh:
                    for key in fh.keys():
                        self.index[key] = path
        if not self.files:
            raise FileNotFoundError(f"no safetensors under {self.dir}")
        self._open: dict[Path, object] = {}

    def _handle(self, path: Path):
        h = self._open.get(path)
        if h is None:
            h = safe_open(str(path), framework="pt")
            h.__enter__()
            self._open[path] = h
        return h

    def keys(self) -> set[str]:
        """Every tensor name the checkpoint offers."""
        return set(self.index)

    def get_tensor(self, name: str) -> torch.Tensor:
        """Fetch one tensor from whichever shard holds it."""
        path = self.index.get(name)
        if path is None:
            raise KeyError(f"{name} not present in {self.dir}")
        return self._handle(path).get_tensor(name)

    def close(self) -> None:
        """Release every open shard handle."""
        for h in self._open.values():
            try:
                h.__exit__(None, None, None)
            except Exception:
                pass
        self._open = {}

    def __enter__(self) -> "ShardReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------------------------------------------------------------- variants
@dataclass(frozen=True)
class Variant:
    """One row of the experiment matrix.

    Attributes:
        name: base results key, e.g. ``awq_p1.0``.
        rule: cache-key rule token — ``none``/``mag``/``awq``/``spike`` for the
            seed method, ``rtn``/``rtn+mag`` for the proxy comparators,
            ``q4km``/``q3km``/``awq_w4`` for the real comparators, ``bf16`` for
            the identity control.  Distinct per method so cache keys can never
            collide.
        p: outlier budget in percent (0 when there is no side channel).
        base_bpw: bits/weight before the side channel.  Zero for the real
            comparators, whose entire cost is measured from their stored bytes.
        salience_rule: ``mag``/``awq``/``spike`` for the experimental rows,
            ``None`` for controls and comparators (used for ranking + winner).
        needs_fit: False only for ``bf16``.
        kind: ``seed`` (our method, config-dependent), ``proxy`` (rtn),
            ``comparator`` (real foreign quantization), ``control``.
    """

    name: str
    rule: str
    p: float
    base_bpw: float
    salience_rule: str | None
    needs_fit: bool = True
    kind: str = "seed"


def build_matrix() -> list[Variant]:
    """The 16 Phase 2 variants, in run order."""
    budgets = [0.1, 0.5, 1.0, 2.0]
    vs: list[Variant] = [Variant("seed_only", "none", 0.0, 4.0, None)]
    vs += [Variant(f"mag_p{p}", "mag", p, 4.0, "mag") for p in budgets]
    vs += [Variant(f"awq_p{p}", "awq", p, 4.0, "awq") for p in budgets]
    vs += [Variant(f"spike_p{p}", "spike", p, 4.0, "spike") for p in budgets]
    vs += [
        Variant("rtn4", "rtn", 0.0, lfsr_core.RTN_BPW, None, kind="proxy"),
        Variant("rtn4_mag_p1.0", "rtn+mag", 1.0, lfsr_core.RTN_BPW, None,
                kind="proxy"),
        Variant("bf16", "bf16", 0.0, lfsr_core.BF16_BPW, None, needs_fit=False,
                kind="control"),
    ]
    return vs


def build_matrix_phase3(C: int, P: int) -> list[Variant]:
    """The lean Phase 3 matrix for one seed config.

    Phase 2 settled the salience ranking (awq >> mag > spike), so Phase 3 runs
    ``awq`` only plus the seed-only control, and spends the saved compute on the
    real comparators instead.

    Args:
        C: seed block size (8 or 12).
        P: coefficients per block (3 or 4).

    Returns:
        seed_only + three awq budgets + rtn4 proxy + three real comparators +
        the bf16 identity control.
    """
    base = lfsr_core.bpw_seed_base(C, P)
    budgets = [0.5, 1.0, 2.0]
    vs: list[Variant] = [Variant("seed_only", "none", 0.0, base, None)]
    vs += [Variant(f"awq_p{p}", "awq", p, base, "awq") for p in budgets]
    vs += [Variant("rtn4", "rtn", 0.0, lfsr_core.RTN_BPW, None, kind="proxy")]
    vs += [Variant(nm, nm, 0.0, 0.0, None, kind="comparator")
           for nm in ("q4km", "q3km", "awq_w4")]
    vs += [Variant("bf16", "bf16", 0.0, lfsr_core.BF16_BPW, None,
                   needs_fit=False, kind="control")]
    return vs


def act_scales_path() -> Path:
    """Where this run's activation scales live.

    The default calibration set keeps the historical, un-suffixed name, so the
    ``act_scales.safetensors`` every measured result was captured under is
    never touched, never overwritten and never has to be recaptured.  A new
    calibration set writes a new file next to it.
    """
    name = ("act_scales.safetensors" if CALIB_ID == DEFAULT_CALIB_ID
            else f"act_scales@{CALIB_ID}.safetensors")
    return CACHE / name


def calib_dependent(rule: str | None) -> bool:
    """Does this rule's output depend on the activation-scale calibration set?

    Exactly two things consume ``act_scales``: the awq side channel (which
    ranks entries by ``|w| * s``) and the W1/W3 fit weighting (which is applied
    to every fitted tensor).  Everything else — a plain seed fit, ``mag``,
    ``spike``, ``rtn``, the real comparators — is bit-identical under any
    calibration set and must keep sharing one cache entry across them, for the
    same reason comparators are config-free: recomputing an identical object
    into a second namespace is pure waste and invites two names for one row.

    ``rule=None`` asks about the run as a whole, which the variant matrix
    always answers yes to (every matrix contains awq rows).
    """
    if rule is None:
        return True
    return rule in WEIGHTABLE_RULES and (FIT_WEIGHTING == "awq" or rule == "awq")


def config_slug(rule: str | None = None) -> str:
    """Config token used in cache keys and results-row names.

    ``c12p4`` for the historical fit, plus one letter per non-default fit knob,
    always in this order, plus a trailing
    ``@calib-id`` when the activation scales did not come from the default
    12-prompt set:

    ==============================  ======  ====================================
    knob                            suffix  meaning
    ==============================  ======  ====================================
    ``--fit-weighting awq``         ``w``   activation-weighted objective (W1)
    ``--coeff-rounding weighted``   ``r``   exact coefficient search (W2)
    ``--incoherence had``           ``h``   seeded orthogonal transform (W3)
    ==============================  ======  ====================================

    So ``c12p4`` / ``c12p4w`` / ``c12p4r`` / ``c12p4h`` / ``c12p4wr`` /
    ``c12p4wh`` / ``c12p4rh`` / ``c12p4wrh`` — eight collision-free names for
    the eight combinations, each its own cache namespace and its own results
    rows.  Suffixes apply only to rules that actually run a seed fit: rtn4 and
    the real comparators are the same object under every combination and keep
    sharing one cache entry.

    The calibration id is a ninth dimension and behaves the same way, with two
    differences: it is a word rather than a letter (``c12p4w@wt256``), and it
    is applied only to rules that actually consume activation scales — see
    :func:`calib_dependent`.  At the default ``p12`` it contributes nothing, so
    every key, path and row name in the project's history is reproduced
    character for character.

    Args:
        rule: variant rule token, or None to ask about the run as a whole.
    """
    if rule is not None and rule not in WEIGHTABLE_RULES:
        return CONFIG_NAME
    return (CONFIG_NAME
            + ("w" if FIT_WEIGHTING == "awq" else "")
            + ("r" if COEFF_ROUNDING == "weighted" else "")
            + ("h" if INCOHERENCE == "had" else "")
            + (f"@{CALIB_ID}" if CALIB_ID != DEFAULT_CALIB_ID
               and calib_dependent(rule) else ""))


def row_name(v: Variant) -> str:
    """Results-row name: config-qualified for our own seed variants.

    ``c12p4_awq_p1.0`` (the config name is part of variant
    names, results rows and cache keys), ``c12p4w_awq_p1.0`` for the
    activation-weighted objective.  Comparators and controls keep plain names
    because they are config-independent and must appear once, not once per
    config, when the two configs' results are merged.
    """
    if LAYOUT == "phase3" and v.kind == "seed":
        return f"{config_slug(v.rule)}_{v.name}"
    return v.name


SALIENCE_RULES = ("mag", "awq", "spike")


# ------------------------------------------------------------------ log/io
def log(msg: str) -> None:
    """Append a timestamped structured line to results/run.log and stdout."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with RUN_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    """Streaming sha256 of a (large) file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def shard_paths() -> list[Path]:
    """Every safetensors file of the configured model, sorted."""
    index = MODEL_DIR / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        return sorted({MODEL_DIR / f for f in weight_map.values()})
    return sorted(MODEL_DIR.glob("*.safetensors"))


def verify_originals(record: bool = True) -> tuple[bool, str]:
    """Check every immutable shard's sha256 against the first-run record.

    The record is a ``{filename: sha256}`` map, so a sharded model is guarded
    shard by shard.  The Phase 2 single-shard record (``{"model.safetensors":
    ...}``) is exactly that map with one entry and keeps validating unchanged.

    Args:
        record: write the record if it does not exist yet.

    Returns:
        ``(ok, digest)``.  ``digest`` is the single shard's hash for
        single-shard models, else a sha256 over the sorted per-shard map.
    """
    files = shard_paths()
    digests = {p.name: sha256_file(p) for p in files}
    combined = (next(iter(digests.values())) if len(digests) == 1
                else hashlib.sha256(
                    json.dumps(digests, sort_keys=True).encode()).hexdigest())
    CACHE.mkdir(parents=True, exist_ok=True)
    if HASH_RECORD.exists():
        want = json.loads(HASH_RECORD.read_text())
        recorded = {k: v for k, v in want.items() if k != "recorded"}
        return (recorded == digests), combined
    if record:
        payload = dict(digests)
        payload["recorded"] = time.strftime("%Y-%m-%d %H:%M:%S")
        HASH_RECORD.write_text(json.dumps(payload, indent=2))
    return True, combined


def cache_key(tensor: str, C: int, P: int, n_seeds: int, rule: str, p: float,
              generator_seed: int, model_slug: str | None = None,
              config_name: str | None = None) -> str:
    """The per-tensor fit cache key.

    Legacy (Phase 2): ``sha1(tensor|C|P|n_seeds|rule|p|generator_seed)[:16]``.

    Phase 3 (``model_slug`` given):
    ``sha1(slug|config|tensor|C|P|n_seeds|rule|p|generator_seed)[:16]``.
    The slug is load-bearing: Qwen3-0.6B, 1.7B and 8B share
    tensor *names*, so without it a 1.7B fit would silently be served a 0.6B
    reconstruction of a different shape.

    Args:
        tensor: HF parameter name.
        C, P: seed-fit block config.
        n_seeds: candidate seeds searched (0 for non-seed methods).
        rule: variant rule token.
        p: outlier budget in percent.
        generator_seed: candidate-set RNG seed.
        model_slug: model directory basename; ``None`` selects the legacy key.
        config_name: config token (``c8p3``/``c12p4``/``c8p3w``/``c12p4w``/
            ``cmp``).  The ``w`` suffix is how an activation-weighted fit gets
            its own cache namespace — see :func:`config_slug`.

    Returns:
        16 hex characters.
    """
    if model_slug is None:
        blob = f"{tensor}|{C}|{P}|{n_seeds}|{rule}|{p}|{generator_seed}"
    else:
        blob = (f"{model_slug}|{config_name}|{tensor}|{C}|{P}|{n_seeds}|"
                f"{rule}|{p}|{generator_seed}")
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def variant_cache_key(v: Variant, tensor: str, n_seeds: int, C: int, P: int) -> str:
    """Cache key for one (variant, tensor) pair under the active layout.

    Non-seed rules (rtn, and the three real comparators) do not depend on the
    seed config, so in the Phase 3 layout they are keyed with ``config="cmp"``
    and ``C=P=n_seeds=0`` and are therefore fitted/dequantized once per model
    rather than once per model per config.  The same reasoning makes them
    weighting-free: :func:`config_slug` only adds the ``w`` suffix for rules
    that run a seed fit.
    """
    ns = 0 if v.rule in CONFIG_FREE_RULES or v.rule.startswith("rtn") else n_seeds
    if LAYOUT == "legacy":
        return cache_key(tensor, C, P, ns, v.rule, v.p, GENERATOR_SEED)
    if v.rule in CONFIG_FREE_RULES:
        return cache_key(tensor, 0, 0, 0, v.rule, v.p, GENERATOR_SEED,
                         MODEL_SLUG, "cmp")
    return cache_key(tensor, C, P, ns, v.rule, v.p, GENERATOR_SEED,
                     MODEL_SLUG, config_slug(v.rule))


def cache_path(key: str) -> Path:
    return CACHE / f"{key}.safetensors"


def cache_load(key: str, device: str = "cpu") -> tuple[torch.Tensor, dict] | None:
    """Load a cached dequantized tensor + its JSON metadata, or None."""
    path = cache_path(key)
    if not path.exists():
        return None
    try:
        with safe_open(str(path), framework="pt") as f:
            meta = json.loads(f.metadata()["meta"])
            w = f.get_tensor("w").to(device)
        return w, meta
    except Exception as exc:            # corrupt/partial file -> refit
        log(f"WARN cache unreadable {path.name}: {exc}")
        return None


def cache_meta(key: str) -> dict | None:
    """Metadata only (no tensor read) — used by the gate computation."""
    path = cache_path(key)
    if not path.exists():
        return None
    try:
        with safe_open(str(path), framework="pt") as f:
            return json.loads(f.metadata()["meta"])
    except Exception:
        return None


def cache_store(key: str, w: torch.Tensor, meta: dict) -> None:
    """Atomically write a dequantized tensor (bf16) + metadata to the cache."""
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = cache_path(key).with_suffix(".tmp")
    save_file({"w": w.to(torch.bfloat16).contiguous().cpu()}, str(tmp),
              metadata={"meta": json.dumps(meta)})
    tmp.replace(cache_path(key))


# --------------------------------------------------------- stall-aware ETA
class EtaTracker:
    """Projected time-to-finish that survives an interrupted overnight run.

    The Phase 2 sleep incident showed the naive ``elapsed / units_done`` average
    is useless once the machine has been suspended: one 6-hour gap poisons the
    rate forever.  This tracker charges each completed unit the wall time since
    the previous unit *only when that gap is plausible work* — gaps longer than
    ``stall_gap_s`` (5 min by default) are excluded from both the
    accumulated time and the unit count.

    Args:
        total_units: units the stage will complete in all.
        stall_gap_s: inter-unit gap above which the interval is treated as a
            stall rather than work.
        done_units: units already finished before this tracker started
            (resumed runs), counted for progress but not for the rate.
        start: wall clock of the first unit boundary; defaults to now.
    """

    def __init__(self, total_units: int, stall_gap_s: float = ETA_STALL_GAP_S,
                 done_units: int = 0, start: float | None = None) -> None:
        self.total = int(total_units)
        self.stall_gap_s = float(stall_gap_s)
        self.done = int(done_units)
        self.counted = 0
        self.productive_s = 0.0
        self.stalls = 0
        self.stalled_s = 0.0
        self._last = time.time() if start is None else float(start)

    def tick(self, now: float | None = None) -> None:
        """Record one completed unit at wall-clock ``now`` (default: now)."""
        now = time.time() if now is None else float(now)
        dt = now - self._last
        self.done += 1
        if 0.0 <= dt <= self.stall_gap_s:
            self.productive_s += dt
            self.counted += 1
        else:
            self.stalls += 1
            self.stalled_s += max(dt, 0.0)
        self._last = now

    @property
    def rate_s(self) -> float | None:
        """Mean seconds per unit over non-stalled intervals, or None."""
        if not self.counted:
            return None
        return self.productive_s / self.counted

    def eta_hours(self) -> float | None:
        """Projected remaining hours, or None until one clean interval exists."""
        rate = self.rate_s
        if rate is None:
            return None
        return rate * max(self.total - self.done, 0) / 3600.0


# ------------------------------------------------------------- act scales
def load_act_scales(device: str = "cpu") -> dict[str, torch.Tensor] | None:
    """Read the cached AWQ activation scales, or None if absent."""
    if not ACT_SCALES.exists():
        return None
    out = {}
    with safe_open(str(ACT_SCALES), framework="pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k).to(device)
    return out


def n_hidden_layers() -> int:
    """Decoder layer count, read from the model's own config.json."""
    return int(json.loads((MODEL_DIR / "config.json").read_text())["num_hidden_layers"])


def act_scales_meta() -> dict | None:
    """Calibration identity recorded with the cached scales, if any."""
    if not ACT_SCALES.exists():
        return None
    try:
        with safe_open(str(ACT_SCALES), framework="pt") as f:
            return json.loads(f.metadata()["meta"])
    except Exception:
        return None


def calib_texts() -> tuple[list[str] | None, dict]:
    """The calibration set for the active ``--calib-id``, plus its identity.

    Returns:
        ``(texts, identity)``.  ``texts`` is ``None`` for the default set,
        which means "let :meth:`swap_eval.Evaluator.capture_act_scales` use its
        own default" — the historical code path, unchanged, so the default
        capture stays bit-for-bit what it always was.
    """
    if CALIB_ID == DEFAULT_CALIB_ID:
        return None, {
            "calib_id": DEFAULT_CALIB_ID,
            "calib_n_prompts": len(swap_eval.PROMPTS),
            "calib_chat_template": True,
            "calib_max_tokens": None,
            "calib_source": "built-in swap_eval.PROMPTS (fixed 12-prompt harness)",
            "corpus_sha256": swap_eval.prompts_sha256(),
            "prompts_sha256": swap_eval.prompts_sha256(),
        }
    texts = swap_eval.load_calib_corpus(CALIB_CORPUS, CALIB_N_PROMPTS)
    return texts, {
        "calib_id": CALIB_ID,
        "calib_n_prompts": len(texts),
        "calib_chat_template": False,
        "calib_max_tokens": CALIB_MAX_TOKENS,
        "calib_source": str(CALIB_CORPUS),
        "corpus_sha256": swap_eval.corpus_sha256(texts),
        "prompts_sha256": swap_eval.prompts_sha256(),
    }


def assert_calib_identity(meta: dict) -> None:
    """Refuse to reuse cached scales that were captured from a different set.

    The file name already carries the calibration id, so the only way to get
    here with a mismatch is a hand-moved file or a corpus that changed under a
    stable id — which is precisely the failure the W1 cache lesson was about:
    a silent identity change behind an unchanged key.  Scales captured under a
    different corpus are refused loudly rather than reused.

    The corpus check is skipped when no corpus is on the command line, because
    a fit/eval rerun legitimately has nothing to compare against; the id check
    is not skippable.
    """
    cached_id = meta.get("calib_id", DEFAULT_CALIB_ID)
    if cached_id != CALIB_ID:
        raise SystemExit(
            f"FATAL {ACT_SCALES.name} was captured under calib_id="
            f"{cached_id!r} but this run is --calib-id {CALIB_ID!r}. Refusing "
            f"to reuse it.")
    if CALIB_ID == DEFAULT_CALIB_ID or CALIB_CORPUS is None:
        return
    want = swap_eval.corpus_sha256(
        swap_eval.load_calib_corpus(CALIB_CORPUS, CALIB_N_PROMPTS))
    got = meta.get("corpus_sha256")
    if got and got != want:
        raise SystemExit(
            f"FATAL {ACT_SCALES.name} was captured from a corpus with sha256 "
            f"{got[:16]}… but --calib-corpus {CALIB_CORPUS} now hashes to "
            f"{want[:16]}… under the same --calib-id {CALIB_ID!r}. One id, one "
            f"corpus: give the new corpus a new id, or restore the old one. "
            f"(Every fit cached under `{config_slug()}` depends on this.)")


def ensure_act_scales(device: str) -> dict[str, torch.Tensor]:
    """Capture (once) and cache mean |input| per input channel per target.

    Always captured for *all* layers regardless of ``--layers-limit``, so the
    cache stays valid for the full overnight run.  Which prompts it is captured
    from is the ``--calib-id`` question; see :func:`calib_texts`.
    """
    have = load_act_scales(device)
    if have:
        assert_calib_identity(act_scales_meta() or {})
        log(f"SKIP cached act_scales ({len(have)} tensors, "
            f"calib={CALIB_ID}, {ACT_SCALES.name})")
        return have
    texts, ident = calib_texts()
    log(f"ACT capturing activation scales (calib={CALIB_ID}, "
        f"{ident['calib_n_prompts']} prompts, "
        f"chat_template={ident['calib_chat_template']}, bf16 model)")
    t0 = time.time()
    ev = swap_eval.Evaluator(MODEL_DIR, device=device, layers_limit=None, dual=False)
    scales = ev.capture_act_scales(
        prompts=texts, chat_template=bool(ident["calib_chat_template"]),
        max_tokens=int(ident["calib_max_tokens"] or swap_eval.CALIB_MAX_TOKENS))
    ev.close()
    del ev
    torch.cuda.empty_cache()
    CACHE.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in scales.items()}, str(ACT_SCALES),
              metadata={"meta": json.dumps(ident)})
    log(f"ACT done ({len(scales)} tensors, {time.time() - t0:.1f}s) -> "
        f"{ACT_SCALES.name} corpus_sha256={ident['corpus_sha256'][:16]}…")
    return {k: v.to(device) for k, v in scales.items()}


def act_scale_drift(device: str = "cpu") -> dict | None:
    """How far this run's scales have moved from the default calibration set.

    The cheapest possible answer to "did the calibration set matter": no fit,
    no eval, seconds of CPU.  Compares the *normalised* scales (the quantity
    the fit actually uses, mean 1 per tensor) tensor by tensor.

    Returns:
        Summary dict, or None when there is nothing to compare against.
    """
    base = CACHE / "act_scales.safetensors"
    if CALIB_ID == DEFAULT_CALIB_ID or not base.exists() or not ACT_SCALES.exists():
        return None
    cos, big, floors = [], [], []
    with safe_open(str(base), framework="pt") as fb, \
            safe_open(str(ACT_SCALES), framework="pt") as fn:
        shared = sorted(set(fb.keys()) & set(fn.keys()))
        for k in shared:
            a = lfsr_core.normalized_col_scale(fb.get_tensor(k).float())
            b = lfsr_core.normalized_col_scale(fn.get_tensor(k).float())
            cos.append(float((a @ b / (a.norm() * b.norm())).item()))
            ratio = (b.clamp(min=1e-12) / a.clamp(min=1e-12))
            big.append(float(((ratio > 2.0) | (ratio < 0.5)).float().mean().item()))
            floors.append(float(b.min().item()))
    if not cos:
        return None
    cos.sort()
    big.sort()
    return {
        "tensors": len(cos),
        "cosine_median": cos[len(cos) // 2],
        "cosine_min": cos[0],
        "frac_channels_2x_median": big[len(big) // 2],
        "frac_channels_2x_max": big[-1],
        "min_normalised_scale": min(floors),
    }


# ------------------------------------------------------------- fitting one
@dataclass
class TensorCtx:
    """Per-tensor scratch shared across the variants of the same tensor."""

    name: str
    w: torch.Tensor                      # bf16 original on device
    seed_only: torch.Tensor | None = None   # flat float32 reconstruction
    svd: tuple | None = None
    act_scale: torch.Tensor | None = None
    extras: dict = field(default_factory=dict)


def _rel_err(w: torch.Tensor, deq: torch.Tensor) -> float:
    """Relative Frobenius error of the *stored* (bf16) reconstruction."""
    d = deq.to(torch.bfloat16).float()
    return float(((w.float() - d).norm() / w.float().norm()).item())


def _weighted_rel_err(w: torch.Tensor, deq: torch.Tensor,
                      act_scale: torch.Tensor | None) -> float | None:
    """Activation-weighted relative error — the quantity a weighted fit minimises.

    Reported alongside the plain ``rel_err`` so the two objectives can be
    compared on their own terms (a weighted fit is *expected* to lose on plain
    rel_err; that is the whole point of moving the residual into low-activation
    channels).

    Returns:
        ``|| (W - Ŵ) diag(s) ||_F / || W diag(s) ||_F``, or None without scales.
    """
    if act_scale is None:
        return None
    s = lfsr_core.normalized_col_scale(act_scale.to(w.device)).unsqueeze(0)
    d = deq.to(torch.bfloat16).float()
    num = ((w.float() - d) * s).norm()
    den = (w.float() * s).norm().clamp(min=1e-30)
    return float((num / den).item())


def fit_col_scale(ctx: "TensorCtx", rule: str) -> torch.Tensor | None:
    """The column weights this (tensor, rule) pair should be fitted under.

    How they are *used* depends on ``INCOHERENCE``: as the diagonal metric of
    the weighted solve (W1), or folded into the fitted tensor (W3 design A).
    Either way this decides whether the fit sees them at all.

    Args:
        ctx: per-tensor scratch, carrying the raw act_scale when captured.
        rule: variant rule token.

    Returns:
        The raw [n] activation scale for a weighted seed fit, else None.

    Raises:
        RuntimeError: weighting was requested but this tensor has no scales —
            silently falling back to the unweighted objective would produce
            rows labelled ``…w`` that are not weighted at all.
    """
    if FIT_WEIGHTING != "awq" or rule not in WEIGHTABLE_RULES:
        return None
    if ctx.act_scale is None:
        raise RuntimeError(f"--fit-weighting awq: no act_scale for {ctx.name}")
    return ctx.act_scale


def fit_modes(rule: str) -> tuple[str, str]:
    """(coeff_rounding, incoherence) for one rule — defaults for non-seed rules.

    rtn/rtn+mag/comparators do not run a seed fit, so they are unaffected by
    both knobs and must stay bit-identical (and cache-shared) across them.
    """
    if rule not in WEIGHTABLE_RULES:
        return "nearest", "none"
    return COEFF_ROUNDING, INCOHERENCE


def fit_tensor_variant(ctx: TensorCtx, v: Variant, n_seeds: int, C: int, P: int
                       ) -> tuple[torch.Tensor, dict]:
    """Produce the dequantized tensor for one (tensor, variant) pair.

    Uses refit-reuse for the scattered rules: only the C-blocks that intersect
    the held-out mask are refitted, spliced over the cached ``seed_only``
    reconstruction (legal because the seed fit is independent per block — which
    remains true under activation weighting, since a block's column weights are
    fixed by its absolute position; see ``lfsr_core.refit_blocks_over``).

    Returns:
        (dequantized float32 tensor, metadata dict).
    """
    w = ctx.w
    numel = w.numel()
    t0 = time.time()
    cs = fit_col_scale(ctx, v.rule)
    rounding, incoherence = fit_modes(v.rule)

    if v.rule == "none":
        deq, _ = lfsr_core.seed_fit_tensor(w, C, P, n_seeds, GENERATOR_SEED,
                                           col_scale=cs, rounding=rounding,
                                           incoherence=incoherence)
        side = salience.SideChannel(kind="none")

    elif v.rule in ("mag", "awq"):
        side = salience.build_side_channel(w, v.rule, v.p, act_scale=ctx.act_scale)
        masked = w.flatten().float()
        masked = masked.clone()
        masked[side.idx] = 0.0
        base = ctx.seed_only
        if base is None:
            raise RuntimeError("seed_only reconstruction unavailable for refit-reuse")
        bids = lfsr_core.refit_block_ids(side.idx, C, w.shape, incoherence)
        flat = lfsr_core.refit_blocks_over(base, masked, bids, C, P, n_seeds,
                                           GENERATOR_SEED, col_scale=cs,
                                           rounding=rounding,
                                           incoherence=incoherence,
                                           shape=w.shape)
        flat[side.idx] = side.values
        deq = flat.view(w.shape)
        side.extra["refit_blocks"] = int(bids.numel())
        side.extra["total_blocks"] = int((numel + C - 1) // C)

    elif v.rule == "spike":
        if ctx.svd is None:
            ctx.svd = salience.singular_spectrum(w)
        side = salience.build_side_channel(w, "spike", v.p, svd=ctx.svd)
        resid = w.float() - side.spike
        deq_resid, _ = lfsr_core.seed_fit_tensor(resid, C, P, n_seeds,
                                                 GENERATOR_SEED, col_scale=cs,
                                                 rounding=rounding,
                                                 incoherence=incoherence)
        deq = deq_resid + side.spike

    elif v.rule == "rtn":
        deq = lfsr_core.rtn_int4(w.float())
        side = salience.SideChannel(kind="none")

    elif v.rule == "rtn+mag":
        side = salience.build_side_channel(w, "mag", v.p)
        masked = w.flatten().float().clone()
        masked[side.idx] = 0.0
        flat = lfsr_core.rtn_int4(masked.view(w.shape)).flatten()
        flat[side.idx] = side.values
        deq = flat.view(w.shape)

    else:
        raise ValueError(f"variant {v.name} has no fit path (rule {v.rule!r})")

    # Design A is the one knob that is not free: decode divides diag(s) back
    # out, so those n fp16 scales are stored side information and are priced
    # here.  W1 (weighting without incoherence) uses s at fit time only and
    # stays at exactly the historical bpw — which is what makes a weighted row
    # comparable to its unweighted twin at equal bits.
    tbits = lfsr_core.transform_side_bits(incoherence, cs is not None, w.shape)
    meta = {
        "tensor": ctx.name,
        "variant": v.name,
        "rule": v.rule,
        "p": v.p,
        "n_seeds": n_seeds,
        "C": C,
        "P": P,
        # Fit-objective tags.  Both W1's weighting and W2's rounding leave every
        # bpw field below untouched — they change which coefficients are chosen,
        # never how many bits they cost (selftest "bit accounting invariant").
        # W3 is the exception and pays for it in `transform_bits`.
        "fit_weighting": "awq" if cs is not None else "none",
        "coeff_rounding": rounding,
        "incoherence": incoherence,
        "weighted_rel_err": _weighted_rel_err(w, deq, ctx.act_scale),
        "numel": int(numel),
        "shape": list(w.shape),
        "bpw_base": v.base_bpw,
        "side_bits": int(side.side_bits),
        "bpw_side": side.side_bits / numel,
        "transform_bits": int(tbits),
        "bpw_transform": tbits / numel,
        "bpw_total": salience.bpw_total(v.base_bpw, side.side_bits, numel)
        + tbits / numel,
        "rank": int(side.r),
        "mass_fraction": float(side.mass_fraction),
        "rel_err": _rel_err(w, deq),
        "fit_seconds": round(time.time() - t0, 3),
        **{f"x_{k}": val for k, val in side.extra.items()},
    }
    return deq, meta


# ------------------------------------------------------------- stage: fit
def stage_fit(variants: list[Variant], n_seeds: int, layers_limit: int | None,
              device: str, resume: bool, C: int, P: int,
              only: list[str] | None = None) -> None:
    """Fit every (tensor, variant) pair, caching and resuming as it goes."""
    torch.manual_seed(0)
    keep = set(only) if only is not None else None
    if keep is not None and any(v.rule in ("mag", "awq") for v in variants
                                if v.name in keep):
        # The scattered rules splice onto the seed_only reconstruction, so it
        # has to be in the same pass (or already cached) or refit-reuse has no
        # base to stand on.
        keep.add("seed_only")
    todo = [v for v in variants
            if v.needs_fit and v.kind != "comparator"
            and (keep is None or v.name in keep)]
    if not todo:
        log("FIT nothing to do (no fittable variants selected)")
        return
    names = swap_eval.target_tensor_names(n_hidden_layers(), layers_limit)
    # Weighted fits need the scales for *every* fitted tensor, not just for the
    # awq side channel's ranking.
    need_act = FIT_WEIGHTING == "awq" or any(v.rule == "awq" for v in todo)
    need_spike = any(v.rule == "spike" for v in todo)
    act_scales = ensure_act_scales(device) if need_act else {}
    if FIT_WEIGHTING == "awq":
        missing_scales = [n for n in names if n not in act_scales]
        if missing_scales:
            raise SystemExit(
                f"--fit-weighting awq: act_scales missing for "
                f"{len(missing_scales)}/{len(names)} target tensors "
                f"(first: {missing_scales[:3]})")

    spectra_path = RESULTS / "spectra.json"
    spectra: dict[str, dict] = {}
    if need_spike and spectra_path.exists():
        try:
            spectra = json.loads(spectra_path.read_text())
        except Exception:
            spectra = {}

    total_units = len(names) * len(todo)
    done_units = 0
    eta = EtaTracker(total_units)
    t_start = time.time()
    log(f"FIT start model={MODEL_SLUG} config={config_slug()} C={C} P={P} "
        f"fit_weighting={FIT_WEIGHTING} coeff_rounding={COEFF_ROUNDING} "
        f"incoherence={INCOHERENCE} "
        f"tensors={len(names)} variants={len(todo)} n_seeds={n_seeds} "
        f"units={total_units}")

    with ShardReader(MODEL_DIR) as shard:
        swap_eval.assert_targets_present(shard.keys(), names)
        for name in names:
            # Skip the whole tensor if every variant for it is already cached.
            keys = {v.name: variant_cache_key(v, name, n_seeds, C, P) for v in todo}
            pending = [v for v in todo
                       if not (resume and cache_path(keys[v.name]).exists())]
            spectrum_needed = need_spike and name not in spectra
            if not pending and not spectrum_needed:
                done_units += len(todo)
                eta.done += len(todo)
                log(f"SKIP cached {name} (all {len(todo)} variants)")
                continue

            w = shard.get_tensor(name).to(device)
            ctx = TensorCtx(name=name, w=w,
                            act_scale=act_scales.get(name) if act_scales else None)

            if need_spike:
                ctx.svd = salience.singular_spectrum(w)
                if name not in spectra:
                    S = ctx.svd[1]
                    spectra[name] = {
                        "family": swap_eval.family_of(name),
                        "layer": swap_eval.layer_of(name),
                        "shape": list(w.shape),
                        "mp_edge": salience.mp_bulk_edge(w),
                        "s": [round(float(x), 6) for x in S.tolist()],
                    }

            for v in todo:
                key = keys[v.name]
                cached = cache_load(key, device) if resume else None
                if cached is not None:
                    if v.rule == "none":
                        ctx.seed_only = cached[0].float().flatten()
                    done_units += 1
                    eta.done += 1
                    log(f"SKIP cached {name} :: {v.name}")
                    continue
                deq, meta = fit_tensor_variant(ctx, v, n_seeds, C, P)
                if v.rule == "none":
                    # Store exactly what the cache would hand back on resume, so
                    # refit-reuse is bit-identical fresh vs. resumed.
                    ctx.seed_only = deq.to(torch.bfloat16).float().flatten().clone()
                cache_store(key, deq, meta)
                done_units += 1
                eta.tick()
                eta_h = eta.eta_hours()
                log(f"FIT {name} :: {v.name} bpw={meta['bpw_total']:.4f} "
                    f"rel_err={meta['rel_err']:.4f} "
                    + (f"wrel_err={meta['weighted_rel_err']:.4f} "
                       if meta.get("weighted_rel_err") is not None else "")
                    + f"t={meta['fit_seconds']:.2f}s "
                    f"[{done_units}/{total_units}] "
                    f"ETA={_fmt(eta_h, 2)}h"
                    + (f" (excluded {eta.stalls} stall(s), "
                       f"{eta.stalled_s / 60:.0f} min)" if eta.stalls else ""))
                if eta_h is not None and eta_h > ETA_WARN_HOURS:
                    log(f"WARN projected remaining {eta_h:.2f}h exceeds "
                        f"{ETA_WARN_HOURS:.0f}h budget")
                del deq
            del ctx, w
            torch.cuda.empty_cache()

    if need_spike and spectra:
        salience.save_spectra(spectra, spectra_path)
        try:
            png = salience.render_mp_atlas(spectra, RESULTS / "mp_atlas.png")
            log(f"ATLAS wrote {png} ({len(spectra)} tensors)")
        except Exception as exc:
            log(f"WARN mp_atlas render failed: {exc}")
    log(f"FIT done in {(time.time() - t_start) / 60:.1f} min")


# ----------------------------------------------------------- stage: calib
def stage_calib(device: str) -> dict:
    """Capture this run's activation scales and report how far they moved.

    Split out from ``--stage fit`` so a recalibration can be done, timed and
    inspected on its own — it is minutes of GPU against hours for a fit, and if
    the scales barely move there is no point paying for the fit at all.

    Returns:
        The drift summary (empty dict when there is no baseline to compare to).
    """
    t0 = time.time()
    scales = ensure_act_scales(device)
    log(f"CALIB {ACT_SCALES.name}: {len(scales)} tensors, "
        f"{time.time() - t0:.1f}s")
    drift = act_scale_drift()
    if drift is None:
        log(f"CALIB drift: no `{DEFAULT_CALIB_ID}` baseline to compare against")
        return {}
    log(f"CALIB drift vs {DEFAULT_CALIB_ID} over {drift['tensors']} tensors: "
        f"cosine median={drift['cosine_median']:.4f} min={drift['cosine_min']:.4f} "
        f"| channels moving >2x: median={drift['frac_channels_2x_median']:.3%} "
        f"max={drift['frac_channels_2x_max']:.3%} "
        f"| smallest normalised scale {drift['min_normalised_scale']:.2e}")
    log("CALIB reading: cosine ~1.00 and few channels moving means the "
        "12-prompt estimate was already converged and a refit will not move "
        "KL; a low cosine means it was not, and the refit is worth its money.")
    return drift


# ----------------------------------------------------- stage: comparators
def stage_comparators(variants: list[Variant], layers_limit: int | None,
                      device: str, resume: bool,
                      gguf_q4km: Path | None, gguf_q3km: Path | None,
                      awq_dir: Path | None) -> list[str]:
    """Dequantize the real comparators into the per-model cache.

    Each comparator's tensors are decoded back to dense bf16 (GGUF K-quants via
    the ``gguf`` package's numpy dequantizers; AWQ via its int32 unpacking) and
    stored under the same cache contract as our own fits, with ``kind:
    comparator`` and ``n_seeds=0``.  Effective bpw is taken
    from the container's own stored bytes, never from the nominal label.

    Missing artifacts are tolerated: Leg A on the laptop legitimately runs
    before the pod has produced any GGUF, and the eval stage simply drops the
    rows it has no cache for.

    Args:
        variants: full matrix (comparator entries are picked out of it).
        layers_limit: restrict to the first N layers.
        device: unused for decode (numpy/CPU), kept for signature symmetry.
        resume: skip tensors already cached.
        gguf_q4km: path to the Q4_K_M GGUF, or None to auto-discover.
        gguf_q3km: path to the Q3_K_M GGUF, or None to auto-discover.
        awq_dir: path to the AWQ checkpoint dir, or None to auto-discover.

    Returns:
        Names of the comparators that were successfully made available.
    """
    import comparators

    names = swap_eval.target_tensor_names(n_hidden_layers(), layers_limit)
    sources: dict[str, tuple[str, Path | None]] = {
        "q4km": ("gguf", gguf_q4km or COMPARATOR_DIR / "Q4_K_M.gguf"),
        "q3km": ("gguf", gguf_q3km or COMPARATOR_DIR / "Q3_K_M.gguf"),
        "awq_w4": ("awq", awq_dir or COMPARATOR_DIR / "awq"),
    }
    available: list[str] = []
    for v in [x for x in variants if x.kind == "comparator"]:
        kind, path = sources[v.rule]
        keys = {n: variant_cache_key(v, n, 0, 0, 0) for n in names}
        if resume and all(cache_path(k).exists() for k in keys.values()):
            log(f"SKIP cached comparator {v.name} (all {len(names)} tensors)")
            available.append(v.name)
            continue
        if path is None or not Path(path).exists():
            log(f"WARN comparator {v.name}: no artifact at {path} — skipping "
                f"(rows will be filled in on a later run)")
            continue
        t0 = time.time()
        try:
            if kind == "gguf":
                src = comparators.GGUFDequantizer(path)
                src.assert_covers(names)
                unmapped = src.unmapped_block_tensors()
                if unmapped:
                    log(f"NOTE {v.name}: {len(unmapped)} GGUF blk tensors outside "
                        f"the 7 compressed families (expected: norms/biases): "
                        f"{unmapped[:3]}")
                stream = src.iter_targets(names)
            else:
                src = comparators.AWQDequantizer(path)
                stream = src.iter_targets(names)
            n_done = 0
            for ct in stream:
                key = keys[ct.hf_name]
                if resume and cache_path(key).exists():
                    n_done += 1
                    continue
                numel = ct.weight.numel()
                meta = {
                    "tensor": ct.hf_name, "variant": v.name, "rule": v.rule,
                    "p": 0.0, "n_seeds": 0, "C": 0, "P": 0,
                    "numel": int(numel), "shape": list(ct.weight.shape),
                    "kind": "comparator", "source": str(path),
                    "qtype": ct.qtype,
                    "bpw_base": 0.0,
                    "side_bits": int(ct.stored_bits),
                    "bpw_side": ct.stored_bits / numel,
                    "bpw_total": ct.stored_bits / numel,
                    "rank": 0, "mass_fraction": 0.0,
                    "rel_err": 0.0,
                    "fit_seconds": 0.0,
                }
                cache_store(key, ct.weight, meta)
                n_done += 1
            log(f"CMP {v.name} dequantized {n_done}/{len(names)} tensors from "
                f"{Path(path).name} in {time.time() - t0:.1f}s")
            available.append(v.name)
            if hasattr(src, "close"):
                src.close()
        except Exception as exc:
            log(f"WARN comparator {v.name} failed: {type(exc).__name__}: {exc}")
    return available


# ------------------------------------------------------------ stage: eval
def collect_fit_stats(variants: list[Variant], names: list[str], n_seeds: int,
                      C: int, P: int) -> dict[str, dict]:
    """Aggregate per-variant fit metadata straight from the cache."""
    out: dict[str, dict] = {}
    for v in variants:
        if not v.needs_fit:
            out[v.name] = {"bpw_base": v.base_bpw, "bpw_side": 0.0,
                           "bpw_transform": 0.0, "transform_bits": 0,
                           "bpw_total": v.base_bpw, "mean_rel_err": 0.0,
                           "mean_weighted_rel_err": None,
                           "fit_seconds": 0.0, "per_tensor": {},
                           "side_bits": 0, "numel": 0, "ranks": {}}
            continue
        per, side_bits, numel, secs, ranks = {}, 0, 0, 0.0, {}
        tbits = 0
        for name in names:
            m = cache_meta(variant_cache_key(v, name, n_seeds, C, P))
            if m is None:
                continue
            per[name] = m
            side_bits += m["side_bits"]
            # Absent in pre-3.6 cache entries, which by construction stored none.
            tbits += int(m.get("transform_bits", 0))
            numel += m["numel"]
            secs += m["fit_seconds"]
            if m["rank"]:
                ranks.setdefault(swap_eval.family_of(name), []).append(m["rank"])
        rel = [m["rel_err"] for m in per.values()]
        wrel = [m["weighted_rel_err"] for m in per.values()
                if m.get("weighted_rel_err") is not None]
        out[v.name] = {
            "bpw_base": v.base_bpw,
            "bpw_side": (side_bits / numel) if numel else 0.0,
            "bpw_transform": (tbits / numel) if numel else 0.0,
            "transform_bits": tbits,
            "bpw_total": v.base_bpw + (((side_bits + tbits) / numel)
                                       if numel else 0.0),
            "mean_rel_err": sum(rel) / len(rel) if rel else 0.0,
            "mean_weighted_rel_err": (sum(wrel) / len(wrel)) if wrel else None,
            "fit_seconds": secs,
            "per_tensor": per,
            "side_bits": side_bits,
            "numel": numel,
            "ranks": ranks,
        }
    return out


def outlier_heavy_reduction(stats: dict[str, dict], vname: str, top_k: int = 10
                            ) -> float | None:
    """Mean rel_err reduction vs seed_only on the most outlier-heavy tensors.

    "Outlier-heavy" is scored mechanically as the share of the tensor's squared
    Frobenius norm carried by the variant's own side channel (``mass_fraction``
    in the cache metadata).  Used by the Phase 2 WEAK gate.

    Returns:
        Mean fractional reduction in [0, 1], or None if unavailable.
    """
    base = stats.get("seed_only", {}).get("per_tensor", {})
    per = stats.get(vname, {}).get("per_tensor", {})
    if not base or not per:
        return None
    ranked = sorted(per.items(), key=lambda kv: -kv[1].get("mass_fraction", 0.0))
    picks = [k for k, _ in ranked[:top_k] if k in base]
    if not picks:
        return None
    reds = []
    for k in picks:
        b, a = base[k]["rel_err"], per[k]["rel_err"]
        if b > 0:
            reds.append((b - a) / b)
    return sum(reds) / len(reds) if reds else None


def gate_verdict(rows: list[dict]) -> dict:
    """Mechanical Phase 2 verdict from the per-variant rows.

    Args:
        rows: dicts with at least ``name``, ``salience_rule``, ``mean_kl``,
            ``bpw_total`` and (optionally) ``outlier_rel_err_reduction``.

    Returns:
        The ``gates`` block of the results schema.

    Rules:
        * ``gap_closure(v) = (KL[seed_only] - KL[v]) / (KL[seed_only] - KL[rtn4])``
        * STRONG: best salience variant has KL reduction >= 30% AND
          gap_closure >= 50%.
        * WEAK: KL reduction >= 15% AND rel_err reduction on the 10 most
          outlier-heavy tensors >= 25%.
        * FAIL otherwise (this includes the "some improvement but under
          threshold" band; the schema admits only three verdicts).
    """
    by = {r.get("base_name", r["name"]): r for r in rows}
    if "seed_only" not in by:
        raise ValueError("gate_verdict requires a seed_only row")
    kl_seed = by["seed_only"]["mean_kl"]
    kl_rtn = by["rtn4"]["mean_kl"] if "rtn4" in by else None

    cands = [r for r in rows if r.get("salience_rule") in SALIENCE_RULES]
    if not cands:
        raise ValueError("gate_verdict requires at least one salience variant")
    # Budget matching (the matched-effective-bpw rule): a variant may not
    # claim the verdict while spending MORE bits than the rtn4 comparator it
    # is credited with approaching.  The unconstrained best is still reported
    # as an informational line, but never decides the verdict.
    rtn_bpw = by.get("rtn4", {}).get("bpw_total")
    if rtn_bpw is not None:
        eligible = [r for r in cands
                    if r.get("bpw_total") is not None
                    and r["bpw_total"] <= rtn_bpw + 1e-6]
    else:
        eligible = cands
    if not eligible:
        eligible = cands  # degenerate config; verdict still computes, flagged below
    best = min(eligible, key=lambda r: r["mean_kl"])
    best_any = min(cands, key=lambda r: r["mean_kl"])

    def reduction(r: dict) -> float:
        return (kl_seed - r["mean_kl"]) / kl_seed if kl_seed > 0 else 0.0

    kl_red = reduction(best)
    gap = None
    if kl_rtn is not None and (kl_seed - kl_rtn) > 0:
        gap = (kl_seed - best["mean_kl"]) / (kl_seed - kl_rtn)

    oer = best.get("outlier_rel_err_reduction")
    if kl_red >= 0.30 and gap is not None and gap >= 0.50:
        verdict = "STRONG"
    elif kl_red >= 0.15 and oer is not None and oer >= 0.25:
        verdict = "WEAK"
    else:
        verdict = "FAIL"

    ranking = []
    for rule in SALIENCE_RULES:
        rs = [r for r in cands if r["salience_rule"] == rule]
        if rs:
            b = min(rs, key=lambda r: r["mean_kl"])
            ranking.append((rule, b["mean_kl"], b["name"], reduction(b)))
    ranking.sort(key=lambda t: t[1])

    return {
        "phase2_verdict": verdict,
        "best_variant": best["name"],
        "best_variant_bpw": best.get("bpw_total"),
        "budget_cap_bpw": rtn_bpw,
        "unconstrained_best_variant": best_any["name"],
        "unconstrained_best_bpw": best_any.get("bpw_total"),
        "unconstrained_best_kl": best_any["mean_kl"],
        "seed_only_bpw": by["seed_only"].get("bpw_total"),
        "rtn4_bpw": by.get("rtn4", {}).get("bpw_total"),
        "kl_seed_only": kl_seed,
        "kl_rtn4": kl_rtn,
        "kl_best": best["mean_kl"],
        "kl_reduction_pct": 100.0 * kl_red,
        "gap_closure_pct": (100.0 * gap) if gap is not None else None,
        "outlier_rel_err_reduction_pct": (100.0 * oer) if oer is not None else None,
        "all_below_10pct": all(reduction(r) < 0.10 for r in cands),
        "salience_ranking": [r[0] for r in ranking],
        "salience_ranking_detail": [
            {"rule": r[0], "best_variant": r[2], "mean_kl": r[1],
             "kl_reduction_pct": 100.0 * r[3]} for r in ranking],
    }


# --------------------------------------------------------- Phase 3 gates
SUB4_BPW_CAP: float = 3.6
PASS_A_BPW_RATIO: float = 0.92          # >= 8% fewer bits than Q3_K_M
PASS_B_KL_RATIO: float = 1.10           # within 10% of Q4_K_M quality
PASS_B_BPW_RATIO: float = 0.85          # >= 15% fewer bits than Q4_K_M
LMEVAL_CONSISTENCY_POINTS: float = 3.0


def assert_single_stack(rows: list[dict]) -> str | None:
    """Refuse to compare rows measured on different stacks.

    The same-stack determinism rule: laptop-measured
    KL and pod-measured KL are not the same quantity.  Mixing them inside one
    gate computation would produce a number that looks precise and means
    nothing, so this raises instead.

    Args:
        rows: results rows, each optionally carrying a ``stack`` string.

    Returns:
        The single stack id, or ``None`` when no row declares one.

    Raises:
        ValueError: if two or more distinct stack ids are present.
    """
    stacks = sorted({r["stack"] for r in rows if r.get("stack")})
    if len(stacks) > 1:
        raise ValueError(
            "refusing to compute gates across mixed measurement stacks "
            f"({len(stacks)} present): {stacks}. Re-evaluate every verdict row "
            "on one stack (same-stack determinism rule).")
    return stacks[0] if stacks else None


def gate_verdict_phase3(rows: list[dict]) -> dict:
    """Mechanical Phase 3 verdict.

    Let ``KL(v)`` be mean teacher-forced KL and ``bpw(v)`` true effective bpw.

    * **PASS (a) — the sub-4 claim**: some seed variant ``v`` with
      ``bpw(v) <= 3.6`` has ``KL(v) <= KL(q3km)`` and
      ``bpw(v) <= 0.92 * bpw(q3km)``.
    * **PASS (b) — the parity claim**: some seed variant ``v`` has
      ``KL(v) <= 1.10 * KL(q4km)`` and ``bpw(v) <= 0.85 * bpw(q4km)``.

    Either clause passes Phase 3.  When the corresponding comparator row is
    absent the clause is ``None`` (unavailable), never a pass; if *no*
    comparator is available at all the verdict is ``INCOMPLETE`` rather than
    ``FAIL``, so a Leg-A-only run cannot be misread as a negative result.

    Args:
        rows: results rows; seed rows are identified by ``kind == "seed"``.

    Returns:
        The Phase 3 ``gates`` block.

    Raises:
        ValueError: on mixed measurement stacks, or with no seed rows.
    """
    stack = assert_single_stack(rows)
    by = {r["name"]: r for r in rows}
    cands = [r for r in rows
             if r.get("kind") == "seed" and r.get("mean_kl") is not None
             and r.get("bpw_total") is not None]
    if not cands:
        raise ValueError("gate_verdict_phase3 requires at least one seed row")

    q3 = by.get("q3km")
    q4 = by.get("q4km")

    def clause_a(r: dict) -> bool:
        return (r["bpw_total"] <= SUB4_BPW_CAP + 1e-9
                and r["mean_kl"] <= q3["mean_kl"] + 1e-12
                and r["bpw_total"] <= PASS_A_BPW_RATIO * q3["bpw_total"] + 1e-9)

    def clause_b(r: dict) -> bool:
        return (r["mean_kl"] <= PASS_B_KL_RATIO * q4["mean_kl"] + 1e-12
                and r["bpw_total"] <= PASS_B_BPW_RATIO * q4["bpw_total"] + 1e-9)

    a_hits = [r for r in cands if clause_a(r)] if q3 else []
    b_hits = [r for r in cands if clause_b(r)] if q4 else []
    a_best = min(a_hits, key=lambda r: r["mean_kl"]) if a_hits else None
    b_best = min(b_hits, key=lambda r: r["mean_kl"]) if b_hits else None

    pass_a = None if q3 is None else bool(a_hits)
    pass_b = None if q4 is None else bool(b_hits)
    if q3 is None and q4 is None:
        verdict = "INCOMPLETE"
    elif pass_a or pass_b:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    best_seed = min(cands, key=lambda r: r["mean_kl"])
    sub4 = [r for r in cands if r["bpw_total"] <= SUB4_BPW_CAP + 1e-9]
    best_sub4 = min(sub4, key=lambda r: r["mean_kl"]) if sub4 else None

    # lm-eval sanity clause (reported, never gating).
    consistent, lm_detail = None, {}
    winner = (a_best or b_best or best_seed)
    if q4 is not None and winner is not None:
        wl, ql = winner.get("lmeval") or {}, q4.get("lmeval") or {}
        deltas = {t: abs(wl[t] - ql[t]) for t in sorted(set(wl) & set(ql))
                  if isinstance(wl.get(t), (int, float))
                  and isinstance(ql.get(t), (int, float))}
        if deltas:
            consistent = all(d <= LMEVAL_CONSISTENCY_POINTS for d in deltas.values())
            lm_detail = {t: round(100.0 * d, 2) for t, d in deltas.items()}

    return {
        "phase3_verdict": verdict,
        "stack": stack,
        "pass_a": pass_a,
        "pass_b": pass_b,
        "pass_a_variant": a_best["name"] if a_best else None,
        "pass_a_bpw": a_best["bpw_total"] if a_best else None,
        "pass_a_kl": a_best["mean_kl"] if a_best else None,
        "pass_b_variant": b_best["name"] if b_best else None,
        "pass_b_bpw": b_best["bpw_total"] if b_best else None,
        "pass_b_kl": b_best["mean_kl"] if b_best else None,
        "best_seed_variant": best_seed["name"],
        "best_seed_kl": best_seed["mean_kl"],
        "best_seed_bpw": best_seed["bpw_total"],
        "best_sub4_variant": best_sub4["name"] if best_sub4 else None,
        "best_sub4_kl": best_sub4["mean_kl"] if best_sub4 else None,
        "best_sub4_bpw": best_sub4["bpw_total"] if best_sub4 else None,
        "q3km_kl": q3["mean_kl"] if q3 else None,
        "q3km_bpw": q3["bpw_total"] if q3 else None,
        "q4km_kl": q4["mean_kl"] if q4 else None,
        "q4km_bpw": q4["bpw_total"] if q4 else None,
        "thresholds": {
            "sub4_bpw_cap": SUB4_BPW_CAP,
            "pass_a_bpw_ratio": PASS_A_BPW_RATIO,
            "pass_b_kl_ratio": PASS_B_KL_RATIO,
            "pass_b_bpw_ratio": PASS_B_BPW_RATIO,
        },
        "benchmark_consistent": consistent,
        "benchmark_deltas_points": lm_detail,
        "comparators_available": sorted(
            n for n in ("q4km", "q3km", "awq_w4") if n in by),
    }


def stage_eval(variants: list[Variant], n_seeds: int, layers_limit: int | None,
               device: str, C: int, P: int, export: str | None = None,
               extra_rows: list[dict] | None = None,
               label_suffix: dict[str, str] | None = None,
               eval_mode: str = "dual") -> dict:
    """Swap every fitted variant into the model and measure KL + harness.

    Args:
        variants: the variants to evaluate (any not fully cached are skipped).
        n_seeds: seed count whose cache entries to read.
        layers_limit: restrict to the first N layers.
        device: compute device.
        C, P: seed config (part of the cache key).
        export: variant name to additionally save as a model directory.
        extra_rows: rows carried in from an earlier stage.
        label_suffix: per-variant suffix appended to the results row name.
        eval_mode: ``dual`` (two resident models) or ``cached`` (one resident
            model + reference logits streamed from disk).

    Returns:
        ``{"rows": [...], "reference": {...}}``.
    """
    torch.manual_seed(0)
    t_stage = time.time()
    ev = swap_eval.Evaluator(MODEL_DIR, device=device, layers_limit=layers_limit,
                             eval_mode=eval_mode, ref_logits_dir=REF_LOGITS)
    names = ev.targets
    log(f"EVAL model={MODEL_SLUG} loaded, {len(names)} target tensors "
        f"(layers_limit={layers_limit}, eval_mode={eval_mode})")

    ref_path = RESULTS / "reference.json"
    reference = None
    if ref_path.exists():
        try:
            reference = json.loads(ref_path.read_text(encoding="utf-8"))
            if reference.get("prompts_sha256") != swap_eval.prompts_sha256():
                reference = None
        except Exception:
            reference = None
    if reference is None:
        log("REF generating bf16 reference pass (200 tok + 160 tok harness)")
        reference = ev.reference_pass(ref_path)
        log(f"REF fingerprint={reference['harness_fingerprint']} "
            f"({reference['elapsed_s']}s)")
    else:
        log(f"SKIP cached reference.json fp={reference['harness_fingerprint']}")

    if eval_mode == "cached":
        info = ev.write_ref_logits(reference)
        log(f"REFLOGITS {info['files']} prompts "
            f"({info['bytes'] / 1e9:.2f} GB, {info['skipped']} cached, "
            f"{info['elapsed_s']}s) -> {REF_LOGITS}")

    stats = collect_fit_stats(variants, names, n_seeds, C, P)
    rows: list[dict] = []
    stack = swap_eval.stack_id()
    total_params = sum(p.numel() for _, p in ev.model_ref.named_parameters())
    target_params = sum(dict(ev.model_ref.named_parameters())[n].numel()
                        for n in names)

    for v in variants:
        t0 = time.time()
        st = stats[v.name]
        missing = []
        if v.needs_fit:
            for name in names:
                got = cache_load(variant_cache_key(v, name, n_seeds, C, P), device)
                if got is None:
                    missing.append(name)
                    continue
                ev.swap_in({name: got[0]})
            if missing:
                log(f"WARN {v.name}: {len(missing)} tensors missing from cache "
                    f"-> skipping variant")
                ev.restore()
                continue
        kl = ev.kl_pass(reference)
        texts, fp = ev.harness_pass()
        ev.restore()
        eff_bits = target_params * st["bpw_total"] + \
            (total_params - target_params) * lfsr_core.BF16_BPW
        row = {
            "name": row_name(v) + (label_suffix or {}).get(v.name, ""),
            "base_name": v.name,
            "rule": v.rule,
            "kind": v.kind,
            "config": config_slug(v.rule) if v.kind == "seed" else "-",
            # The bare config, always a valid --config value: --print-winner
            # hands this to the pod launcher, which would choke on "c12p4w".
            "config_base": CONFIG_NAME if v.kind == "seed" else "-",
            "fit_weighting": (FIT_WEIGHTING if v.kind == "seed"
                              and v.rule in WEIGHTABLE_RULES else "none"),
            "coeff_rounding": fit_modes(v.rule)[0] if v.kind == "seed" else "nearest",
            "incoherence": fit_modes(v.rule)[1] if v.kind == "seed" else "none",
            # "-" means "this row does not read activation scales at all", so
            # it is the same object under every calibration set (comparators,
            # rtn, and an unweighted seed_only/mag/spike fit).
            "calib_id": (CALIB_ID if v.kind == "seed" and calib_dependent(v.rule)
                         else "-"),
            "model_slug": MODEL_SLUG,
            "stack": stack,
            "eval_mode": eval_mode,
            "salience_rule": v.salience_rule,
            "p": v.p,
            "n_seeds": 0 if not v.needs_fit else (
                0 if v.rule in CONFIG_FREE_RULES else n_seeds),
            "bpw_seed_base": st["bpw_base"],
            "bpw_side": st["bpw_side"],
            "bpw_transform": st.get("bpw_transform", 0.0),
            "bpw_total": st["bpw_total"],
            "side_bits_total": st["side_bits"],
            "transform_bits_total": st.get("transform_bits", 0),
            "whole_model_effective_bytes": eff_bits / 8.0,
            "mean_kl": kl.mean_kl,
            "p95_kl": kl.p95_kl,
            "max_kl": kl.max_kl,
            "top1_agree": kl.top1_agree,
            "kl_positions": kl.positions,
            "harness_fingerprint": fp,
            "harness_matches_reference": fp == reference["harness_fingerprint"],
            "fit_seconds": round(st["fit_seconds"], 2),
            "eval_seconds": round(time.time() - t0, 2),
            "mean_rel_err": st["mean_rel_err"],
            "mean_weighted_rel_err": st.get("mean_weighted_rel_err"),
            "spike_ranks": {k: sorted(set(vv)) for k, vv in st["ranks"].items()},
            "outlier_rel_err_reduction": outlier_heavy_reduction(stats, v.name),
            "harness_texts": texts,
        }
        rows.append(row)
        log(f"EVAL {row['name']:<20} bpw={row['bpw_total']:.4f} "
            f"mean_kl={kl.mean_kl:.6f} p95={kl.p95_kl:.6f} max={kl.max_kl:.4f} "
            f"top1={kl.top1_agree:.2f}% fp={fp} ({row['eval_seconds']:.1f}s)")

    if export:
        match = [v for v in variants if v.name == export and v.needs_fit]
        if match:
            v = match[0]
            for name in names:
                got = cache_load(variant_cache_key(v, name, n_seeds, C, P), device)
                if got is not None:
                    ev.swap_in({name: got[0]})
            out = ev.export(RESULTS / f"export_{export}")
            log(f"EXPORT wrote {out}")
            ev.restore()

    ev.close()
    del ev
    torch.cuda.empty_cache()
    log(f"EVAL done in {(time.time() - t_stage) / 60:.1f} min")
    return {"rows": rows + (extra_rows or []), "reference": reference}


# ----------------------------------------------------- stage: equivalence
def stage_equivalence(layers_limit: int | None, device: str, C: int, P: int,
                      n_seeds: int, variant: str = "rtn4",
                      tol: float = 1e-6) -> bool:
    """AC-2: prove ``cached`` and ``dual`` measure the same thing.

    Runs one variant through both evaluation modes back to back and compares the
    mean KL.  The two paths share every line of arithmetic; only the origin of
    the reference logits differs, so any disagreement above ``tol`` is a real
    bug in the caching, not numerical drift.

    Args:
        layers_limit: restrict to the first N layers (keeps the check quick).
        device: compute device.
        C, P: seed config.
        n_seeds: seed count (only matters if ``variant`` is a seed variant).
        variant: which variant to compare; ``rtn4`` per the spec.
        tol: maximum tolerated absolute difference in mean KL.

    Returns:
        True when the modes agree within ``tol``.
    """
    variants = build_matrix_phase3(C, P) if LAYOUT == "phase3" else build_matrix()
    picked = [v for v in variants if v.name == variant]
    if not picked:
        raise SystemExit(f"unknown variant {variant!r} for equivalence check")
    log(f"EQUIV comparing dual vs cached on {variant} "
        f"(layers_limit={layers_limit})")
    out_d = stage_eval(picked, n_seeds, layers_limit, device, C, P,
                       eval_mode="dual")
    out_c = stage_eval(picked, n_seeds, layers_limit, device, C, P,
                       eval_mode="cached")
    if not out_d["rows"] or not out_c["rows"]:
        raise SystemExit(f"{variant} has no cached fit — run --stage fit first")
    kd = out_d["rows"][0]["mean_kl"]
    kc = out_c["rows"][0]["mean_kl"]
    delta = abs(kd - kc)
    ok = delta <= tol
    log(f"EQUIV {variant}: dual mean_kl={kd!r} cached mean_kl={kc!r} "
        f"|delta|={delta:.3e} tol={tol:.0e} -> {'PASS' if ok else 'FAIL'}")
    (RESULTS / "equivalence.json").write_text(json.dumps({
        "variant": variant, "dual_mean_kl": kd, "cached_mean_kl": kc,
        "abs_delta": delta, "tol": tol, "pass": ok,
        "layers_limit": layers_limit, "stack": swap_eval.stack_id(),
        "dual_top1": out_d["rows"][0]["top1_agree"],
        "cached_top1": out_c["rows"][0]["top1_agree"],
    }, indent=2), encoding="utf-8")
    return ok


# ---------------------------------------------------------- stage: lm-eval
def stage_lmeval(rows: list[dict], layers_limit: int | None, device: str,
                 C: int, P: int, n_seeds: int, tasks: list[str], limit: int,
                 variants: list[Variant]) -> list[dict]:
    """Run the small lm-eval suite on bf16, q4km and the Phase 3 winner.

    Uses the documented lm-eval Python API against the **in-memory swapped
    model** (``lm_eval.models.huggingface.HFLM(pretrained=<model object>,
    tokenizer=<tok>)``) so the benchmark measures exactly the weights the KL
    table measured — no re-serialisation, no second stack.

    Args:
        rows: existing results rows (used to pick the winner and to attach
            scores back onto).
        layers_limit: restrict to the first N layers.
        device: compute device.
        C, P: seed config.
        n_seeds: seed count whose cache entries to swap in.
        tasks: lm-eval task names (``gsm8k``, ``ifeval``).
        limit: per-task example cap.
        variants: the matrix, to resolve names back to Variant objects.

    Returns:
        Rows updated in place with an ``lmeval`` dict of ``{task: score}``.

    Note:
        **Pod-tested only.**  ``lm-eval`` is deliberately not installed on the
        laptop (the suite is pod-only); this function raises a
        clear message rather than silently doing nothing if the import fails.
    """
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError as exc:               # pragma: no cover - pod only
        raise SystemExit(
            "stage lmeval needs lm-eval: pip install lm-eval  "
            f"(pod only; not installed on the laptop by design) [{exc}]") from exc

    winner = pick_winner_phase3(rows)
    wanted = ["bf16"] + (["q4km"] if any(r["name"] == "q4km" for r in rows) else [])
    if winner:
        wanted.append(winner)
    log(f"LMEVAL tasks={tasks} limit={limit} rows={wanted}")

    ev = swap_eval.Evaluator(MODEL_DIR, device=device, layers_limit=layers_limit,
                             eval_mode="cached", ref_logits_dir=REF_LOGITS)
    by_row = {r["name"]: r for r in rows}
    out: list[dict] = []
    for target in wanted:
        row = by_row.get(target)
        base = row["base_name"] if row else target
        v = next((x for x in variants if x.name == base), None)
        ev.restore()
        if v is not None and v.needs_fit:
            n_missing = 0
            for name in ev.targets:
                got = cache_load(variant_cache_key(v, name, n_seeds, C, P), device)
                if got is None:
                    n_missing += 1
                else:
                    ev.swap_in({name: got[0]})
            if n_missing:
                log(f"WARN lmeval {target}: {n_missing} tensors missing -> skip")
                continue
        t0 = time.time()
        lm = HFLM(pretrained=ev.model_var, tokenizer=ev.tok, batch_size=1)
        res = lm_eval.simple_evaluate(model=lm, tasks=tasks, limit=limit,
                                      bootstrap_iters=0)
        scores: dict[str, float] = {}
        for task, metrics in (res.get("results") or {}).items():
            for key, val in metrics.items():
                if isinstance(val, (int, float)) and not key.endswith("_stderr") \
                        and "," in key:
                    scores[f"{task}/{key.split(',')[0]}"] = float(val)
        if row is not None:
            row["lmeval"] = scores
            row["lmeval_seconds"] = round(time.time() - t0, 1)
        out.append({"name": target, "lmeval": scores})
        log(f"LMEVAL {target}: {scores} ({time.time() - t0:.0f}s)")
    ev.restore()
    ev.close()
    del ev
    torch.cuda.empty_cache()
    return rows


# ------------------------------------------------------------- reporting
def _fmt(x, nd=6, dash="—"):
    return dash if x is None else f"{x:.{nd}f}"


def merge_rows(old: list[dict], new: list[dict]) -> list[dict]:
    """Merge results rows by ``(name, stack)``, newest wins, order preserved.

    The key includes the stack so an optional cross-machine replication (the
    same 1.7B variants measured on the laptop *and* on the pod) accumulates two
    rows instead of one silently overwriting the other.  Re-running a variant on
    the *same* stack still replaces its row, which is what resume needs.
    """
    out: list[dict] = []
    index: dict[tuple[str, str | None], int] = {}
    for r in old + new:
        key = (r["name"], r.get("stack"))
        if key in index:
            out[index[key]] = r
        else:
            index[key] = len(out)
            out.append(r)
    return out


def stacks_in(rows: list[dict]) -> list[str]:
    """Distinct measurement stacks present in ``rows``, sorted."""
    return sorted({r["stack"] for r in rows if r.get("stack")})


def cross_stack_drift(rows: list[dict]) -> dict | None:
    """Compare two complete, independently-measured verdicts (addendum #2).

    :func:`assert_single_stack` forbids mixing stacks *inside* one gate
    computation.  This does the legitimate opposite: it computes one verdict per
    stack and then reports how far the two stacks' numbers drifted for every
    variant they share.  That is a measurement about reproducibility, not a
    quality comparison, so it never feeds a gate.

    Args:
        rows: all results rows, possibly spanning several stacks.

    Returns:
        ``None`` when fewer than two stacks are present, else a dict with
        ``reference`` stack, per-variant ``deltas``, per-stack ``verdicts`` and
        an ``agreement`` of ``REPLICATED`` / ``DIVERGED``.
    """
    stacks = stacks_in(rows)
    if len(stacks) < 2:
        return None
    by_stack: dict[str, dict[str, dict]] = {s: {} for s in stacks}
    for r in rows:
        if r.get("stack") in by_stack:
            by_stack[r["stack"]][r["name"]] = r
    ref = stacks[0]
    shared = [n for n in by_stack[ref]
              if all(n in by_stack[s] for s in stacks[1:])]
    deltas = []
    for name in sorted(shared):
        entry: dict = {"name": name, "per_stack": {}}
        base = by_stack[ref][name]
        for s in stacks:
            r = by_stack[s][name]
            entry["per_stack"][s] = {
                "mean_kl": r.get("mean_kl"),
                "top1_agree": r.get("top1_agree"),
                "bpw_total": r.get("bpw_total"),
                "abs_delta_kl": (None if s == ref or r.get("mean_kl") is None
                                 or base.get("mean_kl") is None
                                 else r["mean_kl"] - base["mean_kl"]),
                "rel_delta_kl": (None if s == ref or not base.get("mean_kl")
                                 else (r["mean_kl"] - base["mean_kl"])
                                 / base["mean_kl"]),
                "delta_top1": (None if s == ref or r.get("top1_agree") is None
                               or base.get("top1_agree") is None
                               else r["top1_agree"] - base["top1_agree"]),
            }
        deltas.append(entry)

    verdicts: dict[str, str] = {}
    for s in stacks:
        try:
            verdicts[s] = gate_verdict_phase3(
                [r for r in rows if r.get("stack") == s])["phase3_verdict"]
        except ValueError as exc:
            verdicts[s] = f"UNCOMPUTABLE ({exc.__class__.__name__})"
    agreement = ("REPLICATED" if len(set(verdicts.values())) == 1
                 else "DIVERGED")
    return {"stacks": stacks, "reference": ref, "shared_variants": len(shared),
            "deltas": deltas, "verdicts": verdicts, "agreement": agreement}


def gates_for_stacks(rows: list[dict], prefer_stack: str | None = None) -> dict:
    """Phase 3 gates for one stack, plus the cross-stack drift report.

    The verdict itself is always computed inside a single stack (same-stack rule): the
    current one when it has rows, otherwise the stack with the most rows.  Any
    additional stacks show up only in the ``cross_stack`` block.

    Args:
        rows: all results rows.
        prefer_stack: stack whose rows should decide the verdict; defaults to
            the stack this process is running on.

    Returns:
        The Phase 3 ``gates`` block, with ``cross_stack`` attached when a second
        stack is present.
    """
    stacks = stacks_in(rows)
    prefer = prefer_stack or swap_eval.stack_id()
    if len(stacks) <= 1:
        gates = gate_verdict_phase3(rows)
    else:
        pick = prefer if prefer in stacks else max(
            stacks, key=lambda s: sum(1 for r in rows if r.get("stack") == s))
        gates = gate_verdict_phase3([r for r in rows if r.get("stack") == pick])
        gates["stack"] = pick
    drift = cross_stack_drift(rows)
    if drift:
        gates["cross_stack"] = drift
    return gates


def load_results() -> dict | None:
    """Read the results JSON for the active layout, or None."""
    path = RESULTS / RESULTS_JSON
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_results(rows: list[dict], gates: dict, cfg: dict, run_id: str) -> Path:
    """Write the results JSON in the SPEC schema for the active layout."""
    payload = {
        "run_id": run_id,
        "model_slug": MODEL_SLUG,
        "config": cfg,
        "variants": [{k: v for k, v in r.items() if k != "harness_texts"}
                     for r in rows],
        "gates": gates,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / RESULTS_JSON
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    (RESULTS / "harness_outputs.json").write_text(
        json.dumps({r["name"]: r.get("harness_texts", []) for r in rows},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_summary(rows: list[dict], gates: dict, cfg: dict, run_id: str) -> Path:
    """Write results/summary.md: decomposed bpw, KL table, gate verdict."""
    L: list[str] = []
    L.append(f"# Phase 2 results — {run_id}")
    L.append("")
    L.append(f"- config: C={cfg['C']} P={cfg['P']} stage1_seeds={cfg['stage1_seeds']} "
             f"stage2_seeds={cfg['stage2_seeds']} generator_seed={cfg['generator_seed']}")
    L.append(f"- torch {cfg['torch']} | transformers {cfg['transformers']} | "
             f"device {cfg['device']}")
    L.append(f"- layers compressed: {cfg['layers']} | target tensors: "
             f"{cfg['n_target_tensors']} | prompts sha256 {cfg['prompts_sha256'][:16]}…")
    L.append(f"- originals sha256 {cfg['originals_sha256'][:16]}… "
             f"(verified: {cfg['originals_verified']})")
    L.append("")
    L.append("## bits/weight decomposition (compressed tensors only)")
    L.append("")
    L.append("| variant | base bpw | + side bits / numel | = total bpw | "
             "side bits | whole-model MB @bf16 rest |")
    L.append("|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| `{r['name']}` | {r['bpw_seed_base']:.2f} | "
                 f"{r['bpw_side']:.4f} | **{r['bpw_total']:.4f}** | "
                 f"{r['side_bits_total']:,} | "
                 f"{r['whole_model_effective_bytes'] / 1e6:.1f} |")
    L.append("")
    L.append("## behavioural damage (teacher-forced vs bf16 reference)")
    L.append("")
    L.append("| variant | bpw | mean KL | p95 KL | max KL | top-1 agree | "
             "mean rel_err | fingerprint | fit s | eval s |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| `{r['name']}` | {r['bpw_total']:.4f} | {r['mean_kl']:.6f} | "
                 f"{r['p95_kl']:.6f} | {r['max_kl']:.4f} | {r['top1_agree']:.2f}% | "
                 f"{r['mean_rel_err']:.4f} | `{r['harness_fingerprint']}` | "
                 f"{r['fit_seconds']:.0f} | {r['eval_seconds']:.0f} |")
    L.append("")
    spike_rows = [r for r in rows if r["salience_rule"] == "spike"]
    if spike_rows:
        L.append("## spike variants — achieved rank per family "
                 "(budget-matched, achieved not target)")
        L.append("")
        L.append("| variant | achieved bpw | r per family |")
        L.append("|---|---|---|")
        for r in spike_rows:
            fam = ", ".join(f"{k}:{v}" for k, v in sorted(r["spike_ranks"].items()))
            L.append(f"| `{r['name']}` | {r['bpw_total']:.4f} | {fam or '—'} |")
        L.append("")
    L.append("## controls")
    L.append("")
    for nm, cond in (("bf16", "mean_kl == 0.0 and fingerprint == reference"),
                     ("rtn4", "mean_kl > 0")):
        r = next((x for x in rows if x["name"] == nm), None)
        if r is None:
            L.append(f"- `{nm}`: MISSING")
            continue
        if nm == "bf16":
            ok = (r["mean_kl"] == 0.0) and r["harness_matches_reference"]
        else:
            ok = r["mean_kl"] > 0
        L.append(f"- `{nm}` ({cond}): **{'PASS' if ok else 'FAIL'}** — "
                 f"mean_kl={r['mean_kl']!r}, fp={r['harness_fingerprint']}, "
                 f"matches_reference={r['harness_matches_reference']}")
    L.append("")
    L.append("## gate verdict (Phase 2)")
    L.append("")
    L.append(f"**{gates['phase2_verdict']}**  — best salience variant "
             f"`{gates['best_variant']}` @ {_fmt(gates['best_variant_bpw'], 4)} bpw "
             f"(budget-matched: bpw <= rtn4's {_fmt(gates.get('budget_cap_bpw'), 4)})")
    L.append("")
    if gates.get("unconstrained_best_variant") != gates["best_variant"]:
        L.append(f"- unconstrained best (info only, over budget): "
                 f"`{gates['unconstrained_best_variant']}` @ "
                 f"{_fmt(gates.get('unconstrained_best_bpw'), 4)} bpw, "
                 f"mean KL {_fmt(gates.get('unconstrained_best_kl'))}")
    L.append(f"- KL seed_only = {_fmt(gates['kl_seed_only'])} "
             f"@ {_fmt(gates['seed_only_bpw'], 4)} bpw")
    L.append(f"- KL best      = {_fmt(gates['kl_best'])} "
             f"@ {_fmt(gates['best_variant_bpw'], 4)} bpw")
    L.append(f"- KL rtn4      = {_fmt(gates['kl_rtn4'])} "
             f"@ {_fmt(gates['rtn4_bpw'], 4)} bpw")
    L.append(f"- KL reduction vs seed_only = "
             f"{_fmt(gates['kl_reduction_pct'], 2)}%  (STRONG needs >= 30%, "
             f"WEAK needs >= 15%)")
    L.append(f"- gap closure to rtn4 = {_fmt(gates['gap_closure_pct'], 2)}%  "
             f"(STRONG needs >= 50%)")
    L.append(f"- rel_err reduction on 10 most outlier-heavy tensors = "
             f"{_fmt(gates['outlier_rel_err_reduction_pct'], 2)}%  "
             f"(WEAK needs >= 25%)")
    L.append(f"- all salience variants below 10% KL reduction: "
             f"{gates['all_below_10pct']}")
    L.append("")
    if gates.get("stage"):
        L.append(f"- verdict computed on: {gates['stage']}")
        L.append("")
    L.append("### side finding — salience rule ranking (best variant per rule)")
    if gates.get("salience_ranking_stage"):
        L.append(f"_ranked on {gates['salience_ranking_stage']} "
                 f"(only the winner is refit at stage 2)_")
    L.append("")
    L.append("| rank | rule | best variant | mean KL | KL reduction |")
    L.append("|---|---|---|---|---|")
    for i, d in enumerate(gates["salience_ranking_detail"], 1):
        L.append(f"| {i} | **{d['rule']}** | `{d['best_variant']}` | "
                 f"{d['mean_kl']:.6f} | {d['kl_reduction_pct']:.2f}% |")
    L.append("")
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "summary.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def write_summary_phase3(rows: list[dict], gates: dict, cfg: dict,
                         run_id: str) -> Path:
    """Write results/{slug}/summary.md for Phase 3.

    Every bpw figure is decomposed exactly as in Phase 2, and the comparator
    rows show the bpw derived from their own stored bytes next to the nominal
    label they are usually quoted with.
    """
    all_rows = rows
    primary = gates.get("stack")
    if primary and len(stacks_in(rows)) > 1:
        rows = [r for r in rows if r.get("stack") == primary]

    L: list[str] = []
    L.append(f"# Phase 3 results — {cfg['model_slug']} — {run_id}")
    L.append("")
    L.append(f"- stack: `{primary or cfg.get('stack')}`  (all rows below were "
             f"measured on this one stack; same-stack determinism rule)")
    L.append(f"- seed configs present: {', '.join(cfg.get('configs_present', []))} "
             f"| eval mode: {cfg.get('eval_mode')}")
    L.append(f"- stage1_seeds={cfg['stage1_seeds']} stage2_seeds={cfg['stage2_seeds']} "
             f"generator_seed={cfg['generator_seed']}")
    L.append(f"- torch {cfg['torch']} | transformers {cfg['transformers']} | "
             f"device {cfg['device']}")
    L.append(f"- layers compressed: {cfg['layers']} | target tensors: "
             f"{cfg['n_target_tensors']} | prompts sha256 {cfg['prompts_sha256'][:16]}…")
    L.append(f"- act-scale calibration: `{cfg.get('calib_id', DEFAULT_CALIB_ID)}` "
             f"({cfg.get('calib_n_prompts', len(swap_eval.PROMPTS))} prompts"
             + (f", corpus {cfg['calib_corpus']}" if cfg.get("calib_corpus") else "")
             + (f", sha256 {cfg['calib_corpus_sha256'][:16]}…"
                if cfg.get("calib_corpus_sha256") else "") + ")")
    L.append(f"- originals sha256 {cfg['originals_sha256'][:16]}… "
             f"(verified: {cfg['originals_verified']})")
    L.append("")
    L.append("## bits/weight decomposition")
    L.append("")
    L.append("| variant | kind | config | fit obj | rounding | incoh | calib | "
             "base bpw | + side bits / numel | + scales | = total bpw | "
             "side bits | whole-model MB @bf16 rest |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| `{r['name']}` | {r.get('kind', '-')} | {r.get('config', '-')} | "
                 f"{r.get('fit_weighting', 'none')} | "
                 f"{r.get('coeff_rounding', 'nearest')} | "
                 f"{r.get('incoherence', 'none')} | "
                 f"{r.get('calib_id', '-')} | "
                 f"{r['bpw_seed_base']:.2f} | {r['bpw_side']:.4f} | "
                 f"{r.get('bpw_transform', 0.0):.4f} | "
                 f"**{r['bpw_total']:.4f}** | {r['side_bits_total']:,} | "
                 f"{r['whole_model_effective_bytes'] / 1e6:.1f} |")
    L.append("")
    L.append("_Comparator rows carry base 0.00: their entire cost is the "
             "side-bits column, read from the GGUF/AWQ container's own stored "
             "bytes rather than from a nominal label._")
    L.append("")
    if any(r.get("fit_weighting", "none") != "none" for r in rows):
        L.append("_`fit obj = awq` rows (config slug `…w`) were fitted against "
                 "the activation-weighted objective (W1). "
                 "The stored format is unchanged — same seed, same shared "
                 "exponent, same P 4-bit coefficients — so their bpw is "
                 "identical to the matching unweighted row by construction, and "
                 "the comparison is at exactly equal bits._")
        L.append("")
    if any(r.get("coeff_rounding", "nearest") != "nearest" for r in rows):
        L.append("_`rounding = weighted` rows (slug `…r`) chose each block's "
                 "coefficients by exact search over the 3 x 2^P grid points "
                 "instead of rounding each coefficient to its own nearest one "
                 "(W2). Same seed, same exponent field, same P "
                 "4-bit coefficients: the `+ scales` and `= total bpw` columns "
                 "are unchanged, so this comparison is also at exactly equal "
                 "bits._")
        L.append("")
    if any(r.get("calib_id", "-") not in ("-", DEFAULT_CALIB_ID) for r in rows):
        L.append(f"_`calib` names the prompt set the activation scales were "
                 f"captured from. `{DEFAULT_CALIB_ID}` "
                 f"is the fixed 12-prompt harness every earlier result was "
                 f"calibrated on; other ids carry an `@id` in the config slug "
                 f"and were captured into their own `act_scales@id` file, so "
                 f"the 12-prompt scales and every fit that depends on them are "
                 f"untouched. `calib = -` marks a row that never reads "
                 f"activation scales and is therefore the same object under "
                 f"every calibration set. Recalibration changes no bits: these "
                 f"rows are at equal bpw with their `{DEFAULT_CALIB_ID}` "
                 f"twins._")
        L.append("")
    if any(r.get("incoherence", "none") != "none" for r in rows):
        L.append("_`incoh = had` rows (slug `…h`) fitted the seeded orthogonal "
                 "rotation `T = W diag(s) H` and decode as "
                 "`W = T H^T diag(s)^-1` (W3, design A). H is "
                 "seed-derived and free; `diag(s)` is not, and the `+ scales` "
                 "column prices it at 16 bits per input channel (= 16/m bpw). "
                 "These rows are therefore **not** at equal bits with their "
                 "twins — read the total, not the base._")
        L.append("")
    L.append("## behavioural damage (teacher-forced vs bf16 reference)")
    L.append("")
    L.append("| variant | bpw | mean KL | p95 KL | max KL | top-1 agree | "
             "mean rel_err | act-wtd rel_err | fingerprint | fit s | eval s |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: x["bpw_total"]):
        L.append(f"| `{r['name']}` | {r['bpw_total']:.4f} | {r['mean_kl']:.6f} | "
                 f"{r['p95_kl']:.6f} | {r['max_kl']:.4f} | {r['top1_agree']:.2f}% | "
                 f"{r['mean_rel_err']:.4f} | "
                 f"{_fmt(r.get('mean_weighted_rel_err'), 4)} | "
                 f"`{r['harness_fingerprint']}` | "
                 f"{r['fit_seconds']:.0f} | {r['eval_seconds']:.0f} |")
    L.append("")
    lm_rows = [r for r in rows if r.get("lmeval")]
    if lm_rows:
        tasks = sorted({t for r in lm_rows for t in r["lmeval"]})
        L.append("## lm-eval (reported, not gating)")
        L.append("")
        L.append("| variant | " + " | ".join(tasks) + " |")
        L.append("|---" * (len(tasks) + 1) + "|")
        for r in lm_rows:
            L.append(f"| `{r['name']}` | "
                     + " | ".join(_fmt(r["lmeval"].get(t), 4) for t in tasks) + " |")
        L.append("")
    L.append("## controls")
    L.append("")
    for nm, cond in (("bf16", "mean_kl == 0.0 and fingerprint == reference"),
                     ("rtn4", "mean_kl > 0")):
        r = next((x for x in rows if x["name"] == nm), None)
        if r is None:
            L.append(f"- `{nm}`: MISSING")
            continue
        ok = ((r["mean_kl"] == 0.0 and r["harness_matches_reference"])
              if nm == "bf16" else r["mean_kl"] > 0)
        L.append(f"- `{nm}` ({cond}): **{'PASS' if ok else 'FAIL'}** — "
                 f"mean_kl={r['mean_kl']!r}, fp={r['harness_fingerprint']}, "
                 f"matches_reference={r['harness_matches_reference']}")
    L.append("")
    L.append("## gate verdict (Phase 3)")
    L.append("")
    th = gates["thresholds"]
    L.append(f"**{gates['phase3_verdict']}** — pass_a={gates['pass_a']} "
             f"pass_b={gates['pass_b']} "
             f"(comparators available: "
             f"{', '.join(gates['comparators_available']) or 'none'})")
    L.append("")
    L.append(f"- **PASS (a), the sub-4 claim**: a seed variant at bpw <= "
             f"{th['sub4_bpw_cap']} with KL <= KL(q3km) and bpw <= "
             f"{th['pass_a_bpw_ratio']}·bpw(q3km).")
    L.append(f"  - q3km: KL {_fmt(gates['q3km_kl'])} @ {_fmt(gates['q3km_bpw'], 4)} bpw")
    L.append(f"  - best qualifying: `{gates['pass_a_variant']}` KL "
             f"{_fmt(gates['pass_a_kl'])} @ {_fmt(gates['pass_a_bpw'], 4)} bpw")
    L.append(f"- **PASS (b), the parity claim**: a seed variant with KL <= "
             f"{th['pass_b_kl_ratio']}·KL(q4km) and bpw <= "
             f"{th['pass_b_bpw_ratio']}·bpw(q4km).")
    L.append(f"  - q4km: KL {_fmt(gates['q4km_kl'])} @ {_fmt(gates['q4km_bpw'], 4)} bpw")
    L.append(f"  - best qualifying: `{gates['pass_b_variant']}` KL "
             f"{_fmt(gates['pass_b_kl'])} @ {_fmt(gates['pass_b_bpw'], 4)} bpw")
    L.append(f"- best seed variant overall: `{gates['best_seed_variant']}` KL "
             f"{_fmt(gates['best_seed_kl'])} @ {_fmt(gates['best_seed_bpw'], 4)} bpw")
    L.append(f"- best sub-4-bpw seed variant: `{gates['best_sub4_variant']}` KL "
             f"{_fmt(gates['best_sub4_kl'])} @ {_fmt(gates['best_sub4_bpw'], 4)} bpw")
    L.append(f"- benchmark-consistent (winner within "
             f"{LMEVAL_CONSISTENCY_POINTS:.0f} points of q4km, reported not "
             f"gating): {gates['benchmark_consistent']} "
             f"{gates['benchmark_deltas_points'] or ''}")
    L.append("")
    if gates["phase3_verdict"] == "INCOMPLETE":
        L.append("> `INCOMPLETE` means no real comparator row was available in "
                 "this run (Leg A before the pod produced any GGUF/AWQ "
                 "artifacts). It is **not** a negative result — rerun "
                 "`--stage comparators` once the artifacts exist.")
        L.append("")

    drift = gates.get("cross_stack")
    if drift:
        L.append("## cross-stack drift")
        L.append("")
        L.append(f"Two complete verdicts were computed independently, one per "
                 f"stack; **no gate ever mixes them** (same-stack rule). This table only "
                 f"measures how reproducible the numbers are across machines. "
                 f"Reference stack: `{drift['reference']}`.")
        L.append("")
        for i, s in enumerate(drift["stacks"]):
            L.append(f"- stack {i}: `{s}` -> verdict "
                     f"**{drift['verdicts'].get(s)}**")
        L.append("")
        L.append(f"**{drift['agreement']}** — the {len(drift['stacks'])} "
                 f"same-stack verdicts "
                 f"{'agree' if drift['agreement'] == 'REPLICATED' else 'disagree'} "
                 f"({drift['shared_variants']} shared variants compared).")
        L.append("")
        others = drift["stacks"][1:]
        L.append("| variant | "
                 + " | ".join(f"mean KL @stack{i}"
                              for i in range(len(drift["stacks"])))
                 + " | abs ΔKL | rel ΔKL | Δtop-1 pp |")
        L.append("|---" * (len(drift["stacks"]) + 4) + "|")
        for e in drift["deltas"]:
            per = e["per_stack"]
            cells = [f"`{e['name']}`"]
            cells += [_fmt(per[s]["mean_kl"]) for s in drift["stacks"]]
            last = per[others[-1]] if others else {}
            cells.append(_fmt(last.get("abs_delta_kl"), 6))
            cells.append("—" if last.get("rel_delta_kl") is None
                         else f"{100.0 * last['rel_delta_kl']:+.2f}%")
            cells.append("—" if last.get("delta_top1") is None
                         else f"{last['delta_top1']:+.2f}")
            L.append("| " + " | ".join(cells) + " |")
        L.append("")
        L.append(f"_rows in the JSON: {len(all_rows)} across "
                 f"{len(drift['stacks'])} stacks._")
        L.append("")
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "summary.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def make_config(n_seeds: int, stage2_seeds: int, layers_limit: int | None,
                device: str, C: int, P: int, digest: str, ok: bool,
                eval_mode: str = "dual",
                configs_present: list[str] | None = None) -> dict:
    import transformers
    return {
        "C": C,
        "P": P,
        "config_name": CONFIG_NAME,
        "config_slug": config_slug(),
        "fit_weighting": FIT_WEIGHTING,
        "coeff_rounding": COEFF_ROUNDING,
        "incoherence": INCOHERENCE,
        "calib_id": CALIB_ID,
        "calib_n_prompts": (len(swap_eval.PROMPTS)
                            if CALIB_ID == DEFAULT_CALIB_ID else CALIB_N_PROMPTS),
        "calib_corpus": str(CALIB_CORPUS) if CALIB_CORPUS else None,
        "calib_corpus_sha256": (act_scales_meta() or {}).get("corpus_sha256"),
        "model_slug": MODEL_SLUG,
        "model_dir": str(MODEL_DIR),
        "layout": LAYOUT,
        "stack": swap_eval.stack_id(),
        "eval_mode": eval_mode,
        "configs_present": configs_present or [config_slug()],
        "stage1_seeds": n_seeds,
        "stage2_seeds": stage2_seeds,
        "generator_seed": GENERATOR_SEED,
        "prompts_sha256": swap_eval.prompts_sha256(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": device,
        "layers": n_hidden_layers() if layers_limit is None else layers_limit,
        "n_target_tensors": len(swap_eval.target_tensor_names(n_hidden_layers(),
                                                             layers_limit)),
        "originals_sha256": digest,
        "originals_verified": ok,
    }


# ------------------------------------------------------------- self-tests
def _synthetic_gate_rows(kind: str) -> list[dict]:
    """Three hand-built Phase 2 result sets, one per verdict (AC-1 check 4)."""
    base = [
        {"name": "seed_only", "salience_rule": None, "mean_kl": 1.0, "bpw_total": 4.0},
        {"name": "rtn4", "salience_rule": None, "mean_kl": 0.2, "bpw_total": 4.5},
    ]
    if kind == "STRONG":       # 60% reduction, gap closure 0.6/0.8 = 75%
        return base + [{"name": "awq_p1.0", "salience_rule": "awq", "mean_kl": 0.4,
                        "bpw_total": 4.48, "outlier_rel_err_reduction": 0.40},
                       {"name": "mag_p1.0", "salience_rule": "mag", "mean_kl": 0.7,
                        "bpw_total": 4.48, "outlier_rel_err_reduction": 0.20},
                       {"name": "spike_p1.0", "salience_rule": "spike", "mean_kl": 0.9,
                        "bpw_total": 4.47, "outlier_rel_err_reduction": 0.05}]
    if kind == "WEAK":         # 20% reduction (<30), gap closure 25% (<50), oer 30%
        return base + [{"name": "awq_p1.0", "salience_rule": "awq", "mean_kl": 0.8,
                        "bpw_total": 4.48, "outlier_rel_err_reduction": 0.30},
                       {"name": "mag_p1.0", "salience_rule": "mag", "mean_kl": 0.95,
                        "bpw_total": 4.48, "outlier_rel_err_reduction": 0.10},
                       {"name": "spike_p1.0", "salience_rule": "spike", "mean_kl": 0.99,
                        "bpw_total": 4.47, "outlier_rel_err_reduction": 0.02}]
    return base + [{"name": "awq_p1.0", "salience_rule": "awq", "mean_kl": 0.95,
                    "bpw_total": 4.48, "outlier_rel_err_reduction": 0.50},
                   {"name": "mag_p1.0", "salience_rule": "mag", "mean_kl": 0.97,
                    "bpw_total": 4.48, "outlier_rel_err_reduction": 0.40},
                   {"name": "spike_p1.0", "salience_rule": "spike", "mean_kl": 0.99,
                    "bpw_total": 4.47, "outlier_rel_err_reduction": 0.30}]


def _synthetic_gate_rows_p3(kind: str, stack: str = "hostA|gpuA|torchX"
                            ) -> list[dict]:
    """Phase 3 result sets exercising PASS-a / PASS-b / both / FAIL / mixed stack.

    Comparator anchors are the spec's own numbers: Q3_K_M ~3.9 bpw and Q4_K_M
    ~4.85 bpw.  The seed row is moved around them to hit exactly one clause at a
    time.
    """
    cmp_rows = [
        {"name": "q3km", "kind": "comparator", "mean_kl": 0.30, "bpw_total": 3.90,
         "stack": stack},
        {"name": "q4km", "kind": "comparator", "mean_kl": 0.20, "bpw_total": 4.85,
         "stack": stack},
    ]
    seed = {"name": "c12p4_awq_p1.0", "kind": "seed", "stack": stack}
    if kind == "PASS_A":       # sub-4 win only: cheap + not worse than q3km
        seed |= {"mean_kl": 0.28, "bpw_total": 3.48}
    elif kind == "PASS_B":     # parity with q4km at >=15% fewer bits, but >3.6 bpw
        seed |= {"mean_kl": 0.21, "bpw_total": 4.00}
    elif kind == "BOTH":
        seed |= {"mean_kl": 0.20, "bpw_total": 3.40}
    elif kind == "FAIL":
        seed |= {"mean_kl": 0.45, "bpw_total": 3.90}
    elif kind == "MIXED_STACK":
        seed |= {"mean_kl": 0.20, "bpw_total": 3.40, "stack": "hostB|gpuB|torchY"}
    else:
        raise ValueError(kind)
    return cmp_rows + [seed]


def _selftest_shard_dir(tmp: Path) -> Path:
    """Build a synthetic two-shard safetensors model dir with an index.json."""
    tmp.mkdir(parents=True, exist_ok=True)
    a = {"model.layers.0.self_attn.q_proj.weight": torch.arange(12,
                                                                dtype=torch.float32
                                                                ).view(3, 4)}
    b = {"model.layers.1.mlp.up_proj.weight": torch.full((2, 5), 7.0)}
    save_file(a, str(tmp / "model-00001-of-00002.safetensors"))
    save_file(b, str(tmp / "model-00002-of-00002.safetensors"))
    (tmp / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 68},
        "weight_map": {
            "model.layers.0.self_attn.q_proj.weight":
                "model-00001-of-00002.safetensors",
            "model.layers.1.mlp.up_proj.weight":
                "model-00002-of-00002.safetensors",
        }}), encoding="utf-8")
    return tmp


def selftest(device: str, C: int, P: int) -> bool:
    """AC-1: all Phase 2 checks plus the six Phase 3 units, target < 5 min."""
    t0 = time.time()
    ok_all = True

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal ok_all
        ok_all &= bool(ok)
        log(f"SELFTEST {'PASS' if ok else 'FAIL'}  {label}"
            + (f"  {detail}" if detail else ""))

    torch.manual_seed(0)

    # ---------------------------------------------------- Phase 2 checks
    # 1. LFSR determinism + golden vector
    s = torch.tensor([1, 42, 65535], dtype=torch.int64)
    a, b = lfsr_core.lfsr_states(s, 8), lfsr_core.lfsr_states(s, 8)
    check("LFSR determinism", torch.equal(a, b))
    check("LFSR golden vector", a.tolist() == LFSR_GOLDEN,
          f"got {a.tolist()[1]}")

    # 2. mask -> block-id mapping
    idx = torch.tensor([0, 7, 8, 9, 31, 32], dtype=torch.int64)
    bids = lfsr_core.block_ids_for_indices(idx, 8)
    check("mask -> block ids", bids.tolist() == [0, 1, 3, 4], f"{bids.tolist()}")

    # 3. bpw accounting, scattered + spike, on a synthetic [64, 128]
    m, n = 64, 128
    numel = m * n                                   # 8192
    k = salience.scatter_count(numel, 1.0)          # int(81.92) = 81
    scat_bits = k * salience.SCATTER_BITS_PER_ENTRY  # 81 * 48 = 3888
    r = salience.spike_rank_for_budget(m, n, 1.0)   # round(3932.16/3088) = 1
    spike_bits = (m + n + 1) * r * salience.SPIKE_BITS_PER_UNIT   # 193*16 = 3088
    check("scattered accounting [64,128] p=1.0",
          k == 81 and scat_bits == 3888
          and abs(salience.bpw_total(4.0, scat_bits, numel) - 4.474609375) < 1e-12,
          f"k={k} bits={scat_bits} bpw={salience.bpw_total(4.0, scat_bits, numel)}")
    check("spike accounting [64,128] p=1.0",
          r == 1 and spike_bits == 3088
          and abs(salience.bpw_total(4.0, spike_bits, numel) - 4.376953125) < 1e-12,
          f"r={r} bits={spike_bits} bpw={salience.bpw_total(4.0, spike_bits, numel)}")
    wsyn = torch.randn(m, n, device=device) * 0.02
    sc = salience.build_side_channel(wsyn, "spike", 1.0)
    check("spike side channel matches formula",
          sc.side_bits == spike_bits and sc.r == r,
          f"r={sc.r} bits={sc.side_bits}")

    # 4. block-path fit == phase1_fit.seedlm_fit, on a real tensor @256 seeds
    name = "model.layers.11.self_attn.o_proj.weight"
    with ShardReader(MODEL_DIR) as rdr:
        w = rdr.get_tensor(name).to(device)
    d_ref, bpw_ref = lfsr_core.seedlm_fit(w, C, P, 256)
    d_new, bpw_new = lfsr_core.seed_fit_tensor(w, C, P, 256, GENERATOR_SEED)
    e_ref = lfsr_core.rel_err(w, d_ref)
    e_new = lfsr_core.rel_err(w, d_new)
    check("block fit @256 == phase1_fit (4 dp)",
          round(e_ref, 4) == round(e_new, 4) and abs(bpw_ref - bpw_new) < 1e-12,
          f"phase1={e_ref:.6f} lfsr_core={e_new:.6f} bpw={bpw_new:.2f}")

    # 5. refit-reuse == full masked refit (the legality of the cache shortcut)
    small = w[:128, :128].contiguous()
    d_full, bpw_full = lfsr_core.with_outliers(small, 1.0, C, P, 256)
    side = salience.build_side_channel(small, "mag", 1.0)
    base, _ = lfsr_core.seed_fit_tensor(small, C, P, 256, GENERATOR_SEED)
    masked = small.flatten().float().clone()
    masked[side.idx] = 0.0
    spliced = lfsr_core.refit_blocks_over(
        base.flatten().float(), masked,
        lfsr_core.block_ids_for_indices(side.idx, C), C, P, 256, GENERATOR_SEED)
    spliced[side.idx] = side.values
    bpw_side = salience.bpw_total(4.0, side.side_bits, small.numel())
    check("refit-reuse == full refit",
          torch.allclose(spliced.view(small.shape), d_full, atol=1e-5)
          and abs(bpw_side - bpw_full) < 1e-9,
          f"max|d|={(spliced.view(small.shape) - d_full).abs().max().item():.2e} "
          f"bpw {bpw_side:.4f} vs {bpw_full:.4f}")

    # 6. Phase 2 gate logic on three synthetic result sets
    for want in ("STRONG", "WEAK", "FAIL"):
        got = gate_verdict(_synthetic_gate_rows(want))
        check(f"gate logic (P2) -> {want}", got["phase2_verdict"] == want,
              f"got {got['phase2_verdict']} "
              f"(red={got['kl_reduction_pct']:.1f}% "
              f"gap={_fmt(got['gap_closure_pct'], 1)}%)")
    got = gate_verdict(_synthetic_gate_rows("STRONG"))
    check("salience ranking order", got["salience_ranking"] == ["awq", "mag", "spike"],
          str(got["salience_ranking"]))

    # 7. originals immutability
    ok, digest = verify_originals(record=True)
    check("originals sha256", ok, digest[:32] + "...")

    # ---------------------------------------------------- Phase 3 checks
    # 8. cache key carries the model slug (AC-1a)
    saved = (LAYOUT, MODEL_SLUG, CONFIG_NAME)
    try:
        globals()["LAYOUT"] = "phase3"
        tname = "model.layers.0.self_attn.q_proj.weight"
        v = Variant("seed_only", "none", 0.0, 4.0, None)
        globals()["MODEL_SLUG"], globals()["CONFIG_NAME"] = "Qwen3-0.6B", "c8p3"
        k06 = variant_cache_key(v, tname, 512, 8, 3)
        globals()["MODEL_SLUG"] = "Qwen3-1.7B"
        k17 = variant_cache_key(v, tname, 512, 8, 3)
        globals()["MODEL_SLUG"], globals()["CONFIG_NAME"] = "Qwen3-0.6B", "c12p4"
        k12 = variant_cache_key(v, tname, 512, 12, 4)
        globals()["LAYOUT"] = "legacy"
        klegacy = variant_cache_key(v, tname, 512, 8, 3)
        check("cache key: slug + config distinguish keys",
              len({k06, k17, k12, klegacy}) == 4
              and klegacy == cache_key(tname, 8, 3, 512, "none", 0.0, GENERATOR_SEED),
              f"0.6B={k06} 1.7B={k17} c12p4={k12} legacy={klegacy}")
        # comparator keys must be config-free (one dequant serves both configs)
        globals()["LAYOUT"] = "phase3"
        cv = Variant("q4km", "q4km", 0.0, 0.0, None, kind="comparator")
        globals()["CONFIG_NAME"] = "c8p3"
        c1 = variant_cache_key(cv, tname, 512, 8, 3)
        globals()["CONFIG_NAME"] = "c12p4"
        c2 = variant_cache_key(cv, tname, 512, 12, 4)
        check("comparator cache key is config-free", c1 == c2, f"{c1} == {c2}")
        # W1: the fit objective is part of the namespace, for seed rules only
        globals()["CONFIG_NAME"] = "c12p4"
        globals()["FIT_WEIGHTING"] = "none"
        k_plain = variant_cache_key(v, tname, 512, 12, 4)
        k_cmp_plain = variant_cache_key(cv, tname, 512, 12, 4)
        n_plain, s_plain = row_name(v), config_slug(v.rule)
        globals()["FIT_WEIGHTING"] = "awq"
        k_wtd = variant_cache_key(v, tname, 512, 12, 4)
        k_cmp_wtd = variant_cache_key(cv, tname, 512, 12, 4)
        n_wtd, s_wtd = row_name(v), config_slug(v.rule)
        check("cache key: fit weighting distinguishes keys",
              k_plain != k_wtd and k_wtd == cache_key(
                  tname, 12, 4, 512, "none", 0.0, GENERATOR_SEED,
                  "Qwen3-0.6B", "c12p4w")
              and k_plain == k12 and k_cmp_plain == k_cmp_wtd
              and s_plain == "c12p4" and s_wtd == "c12p4w"
              and n_plain == "c12p4_seed_only" and n_wtd == "c12p4w_seed_only",
              f"{k_plain} != {k_wtd}, comparator shared={k_cmp_plain == k_cmp_wtd}, "
              f"rows {n_plain} / {n_wtd}")
        # 8c (W2+W3). Every one of the eight (weighting, rounding, incoherence)
        # combinations must own a distinct cache namespace and a distinct row
        # name, while the comparators keep sharing exactly one entry across all
        # eight.  This is the check that stops a c12p4wrh run from silently
        # being served a c12p4w reconstruction.
        combos = [(w, r, h) for w in ("none", "awq")
                  for r in ("nearest", "weighted") for h in ("none", "had")]
        seed_keys, cmp_keys, slugs = [], set(), []
        for wm, rm, hm in combos:
            globals()["FIT_WEIGHTING"] = wm
            globals()["COEFF_ROUNDING"] = rm
            globals()["INCOHERENCE"] = hm
            seed_keys.append(variant_cache_key(v, tname, 512, 12, 4))
            cmp_keys.add(variant_cache_key(cv, tname, 512, 12, 4))
            slugs.append(config_slug(v.rule))
        expect = ["c12p4", "c12p4h", "c12p4r", "c12p4rh",
                  "c12p4w", "c12p4wh", "c12p4wr", "c12p4wrh"]
        check("cache key: all 8 fit-knob combinations are distinct",
              len(set(seed_keys)) == 8 and len(cmp_keys) == 1
              and slugs == expect and seed_keys[0] == k12,
              f"{len(set(seed_keys))}/8 distinct, comparators share "
              f"{len(cmp_keys)}, slugs {slugs}")
    finally:
        globals()["FIT_WEIGHTING"] = "none"
        globals()["COEFF_ROUNDING"] = "nearest"
        globals()["INCOHERENCE"] = "none"
        globals()["LAYOUT"], globals()["MODEL_SLUG"], globals()["CONFIG_NAME"] = saved

    # 8b. weighting is refused in the legacy layout (no slug to namespace with).
    # configure_layout validates before it mutates, so a raise leaves the
    # process's layout untouched and nothing needs restoring.
    refused = []
    for kwargs in ({"fit_weighting": "awq"}, {"coeff_rounding": "weighted"},
                   {"incoherence": "had"}):
        try:
            configure_layout(None, "c8p3", **kwargs)
        except ValueError:
            refused.append(list(kwargs)[0])
    check("non-default fits refused in legacy layout", len(refused) == 3,
          f"refused {refused}")

    # 9. c12p4 accounting is exactly 3.0 bpw, and a [64,132] tensor round-trips
    b12 = lfsr_core.bpw_seed_base(12, 4)
    exact = ((16 + 4 + 4 * 4) / 12) == 3.0
    w12 = torch.randn(64, 132, device=device) * 0.02
    d12, bpw12 = lfsr_core.seed_fit_tensor(w12, 12, 4, 256, GENERATOR_SEED)
    e12 = lfsr_core.rel_err(w12, d12)
    # 64*133 = 8512 is NOT divisible by 12 -> exercises the block padding path
    w12b = torch.randn(64, 133, device=device) * 0.02
    d12b, _ = lfsr_core.seed_fit_tensor(w12b, 12, 4, 256, GENERATOR_SEED)
    check("c12p4 accounting == 3.0 bpw exactly",
          exact and b12 == 3.0 and bpw12 == 3.0,
          f"(16+4+16)/12={b12!r} fit_bpw={bpw12!r}")
    check("c12p4 [64,132] round-trip (+ [64,133] padding path)",
          d12.shape == w12.shape and torch.isfinite(d12).all()
          and 0.0 < e12 < 1.0 and d12b.shape == w12b.shape
          and torch.isfinite(d12b).all(),
          f"rel_err={e12:.4f} shapes {tuple(d12.shape)}/{tuple(d12b.shape)}")

    # 10. sharded loading resolves tensors via index.json (AC-1c)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        sd = _selftest_shard_dir(Path(td) / "shards")
        with ShardReader(sd) as rdr:
            ok_keys = rdr.keys() == {
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.1.mlp.up_proj.weight"}
            t0v = rdr.get_tensor("model.layers.0.self_attn.q_proj.weight")
            t1v = rdr.get_tensor("model.layers.1.mlp.up_proj.weight")
            two_files = len(rdr.files) == 2
        missing_raises = False
        try:
            swap_eval.assert_targets_present(
                {"model.layers.0.self_attn.q_proj.weight"},
                ["model.layers.0.self_attn.q_proj.weight",
                 "model.layers.0.mlp.down_proj.weight"])
        except KeyError:
            missing_raises = True
    check("sharded index resolution (2 shards)",
          ok_keys and two_files and tuple(t0v.shape) == (3, 4)
          and tuple(t1v.shape) == (2, 5) and float(t0v[0, 0]) == 0.0
          and float(t1v[1, 4]) == 7.0,
          f"keys_ok={ok_keys} files={two_files}")
    check("target coverage assertion fails loud", missing_raises)

    # 11. Phase 3 gate logic: 4 verdict cases + the mixed-stack refusal
    for want, expect in (("PASS_A", ("PASS", True, False)),
                         ("PASS_B", ("PASS", False, True)),
                         ("BOTH", ("PASS", True, True)),
                         ("FAIL", ("FAIL", False, False))):
        g = gate_verdict_phase3(_synthetic_gate_rows_p3(want))
        got = (g["phase3_verdict"], g["pass_a"], g["pass_b"])
        check(f"gate logic (P3) -> {want}", got == expect, f"got {got}")
    mixed_raises = False
    try:
        gate_verdict_phase3(_synthetic_gate_rows_p3("MIXED_STACK"))
    except ValueError:
        mixed_raises = True
    check("gate logic (P3) refuses mixed stacks", mixed_raises)
    g_inc = gate_verdict_phase3([
        {"name": "c12p4_seed_only", "kind": "seed", "mean_kl": 0.5,
         "bpw_total": 3.0, "stack": "hostA|gpuA|torchX"}])
    check("gate logic (P3) -> INCOMPLETE without comparators",
          g_inc["phase3_verdict"] == "INCOMPLETE",
          f"got {g_inc['phase3_verdict']}")

    # 11b. cross-stack drift: two COMPLETE verdicts compared without raising
    lap = _synthetic_gate_rows_p3("BOTH", stack="laptop|RTX5070|torchX")
    pod = [dict(r, stack="pod|RTX4090|torchX") for r in
           _synthetic_gate_rows_p3("BOTH")]
    for r in lap + pod:                    # give the drift table a top-1 column
        r["top1_agree"] = 90.0 if r["stack"].startswith("laptop") else 89.5
    pod[-1]["mean_kl"] = 0.202             # small drift on the shared seed row
    both = merge_rows(lap, pod)
    drift = cross_stack_drift(both)
    g_multi = gates_for_stacks(both, prefer_stack="pod|RTX4090|torchX")
    d0 = drift["deltas"][0]["per_stack"]["pod|RTX4090|torchX"] if drift else {}
    check("cross-stack drift table + agreement line",
          len(both) == 6 and drift is not None
          and drift["agreement"] == "REPLICATED"
          and drift["shared_variants"] == 3
          and set(drift["verdicts"].values()) == {"PASS"}
          and d0.get("delta_top1") is not None
          and g_multi["stack"] == "pod|RTX4090|torchX"
          and g_multi["cross_stack"]["agreement"] == "REPLICATED",
          f"rows={len(both)} shared={drift['shared_variants']} "
          f"agreement={drift['agreement']} verdicts={list(drift['verdicts'].values())}")
    diverged = merge_rows(
        _synthetic_gate_rows_p3("BOTH", stack="laptop|RTX5070|torchX"),
        [dict(r, stack="pod|RTX4090|torchX")
         for r in _synthetic_gate_rows_p3("FAIL")])
    check("cross-stack drift flags DIVERGED",
          cross_stack_drift(diverged)["agreement"] == "DIVERGED")

    # 12. GGUF Q4_K dequant round-trip (AC-4)
    try:
        import comparators
        rt = comparators.q4k_roundtrip_check((64, 512), seed=0)
        check("GGUF Q4_K dequant round-trip",
              bool(rt["shape_ok"]) and rt["dtype"] == "float32"
              and bool(rt["in_band"]),
              f"rel_err={rt['rel_err']:.4f} (band 0.05-0.15) "
              f"dtype={rt['dtype']} stored_bpw={rt['bpw']:.3f}")
        # and the name mapping, both directions, over the 7 families
        hf = swap_eval.target_tensor_names(2)
        rt_ok = all(comparators.gguf_to_hf_name(comparators.hf_to_gguf_name(h)) == h
                    for h in hf)
        unmapped_raises = False
        try:
            comparators.hf_to_gguf_name("model.layers.0.self_attn.q_norm.weight")
        except KeyError:
            unmapped_raises = True
        check("GGUF<->HF name mapping round-trips + fails loud",
              rt_ok and unmapped_raises
              and comparators.hf_to_gguf_name(
                  "model.layers.3.mlp.up_proj.weight") == "blk.3.ffn_up.weight",
              f"{len(hf)} names")
        # ...and the whole production reader, on a real .gguf file, offline.
        with tempfile.TemporaryDirectory() as td:
            fr = comparators.gguf_file_roundtrip_check(Path(td))
        check("GGUF file -> GGUFDequantizer -> torch (no llama.cpp)",
              fr["shape_ok"] and fr["dtype"] == "torch.float32"
              and fr["qtype"] == "Q4_K" and abs(fr["bpw"] - 4.5) < 1e-9
              and bool(fr["in_band"]) and not fr["unmapped"]
              and fr["hf_name"] == "model.layers.0.self_attn.q_proj.weight",
              f"{fr['file_bytes']}B gguf -> bpw={fr['bpw']:.4f} "
              f"rel_err={fr['rel_err']:.4f} qtype={fr['qtype']}")
    except ImportError as exc:
        check("GGUF Q4_K dequant round-trip", False, f"gguf missing: {exc}")

    # 13. stall-aware ETA: a 1000 s gap must not poison the rate
    eta = EtaTracker(total_units=10, stall_gap_s=300.0, start=0.0)
    for ts in (1.0, 2.0, 3.0, 1003.0, 1004.0, 1005.0):
        eta.tick(ts)
    naive = 1005.0 / 6
    check("stall-aware ETA excludes >5 min gaps",
          eta.counted == 5 and abs(eta.rate_s - 1.0) < 1e-9
          and eta.stalls == 1 and abs(eta.eta_hours() - 4 / 3600.0) < 1e-12
          and naive > 100,
          f"rate={eta.rate_s:.4f}s/unit (naive would be {naive:.1f}) "
          f"stalls={eta.stalls} eta={eta.eta_hours() * 3600:.1f}s")

    # ------------------------------------ W1: activation-weighted fit objective
    # A synthetic tensor with two dominant activation channels: the weighted
    # objective should move residual out of them and into the cheap columns.
    torch.manual_seed(7)
    mw, nw, sw = 64, 96, 256
    ww = torch.randn(mw, nw, device=device) * 0.02
    scale_w = torch.rand(nw, device=device) * 0.5 + 0.1
    scale_w[7], scale_w[31] = 40.0, 25.0
    sn = lfsr_core.normalized_col_scale(scale_w)

    # 14a. the [B, C] weight matrix is factored exactly (this is the indexing
    # the whole weighted path rides on)
    pat, grp = lfsr_core.block_weight_layout(ww.numel(), C, sn)
    naive = lfsr_core.to_blocks(sn.repeat(mw), C)          # numel % C == 0 here
    pad_pat, pad_grp = lfsr_core.block_weight_layout(ww.numel() + 5, C, sn)
    check("weighted fit: block weight layout == naive gather",
          torch.equal(pat[grp], naive)
          and pat.shape == (nw // math.gcd(C, nw), C)
          and pad_grp.numel() == (ww.numel() + 5 + C - 1) // C
          and torch.equal(pad_grp[:grp.numel()], grp),
          f"{pat.shape[0]} patterns for n={nw} C={C}, "
          f"padded blocks {pad_grp.numel()}")

    # 14b. unweighted path untouched, and s == 1 reproduces it through the
    # weighted solver (the algebra check: diag(1) U = U, so the weighted normal
    # equations must collapse onto pinv(U))
    d_unw, _ = lfsr_core.seed_fit_tensor(ww, C, P, sw, GENERATOR_SEED)
    d_ones, _ = lfsr_core.seed_fit_tensor(ww, C, P, sw, GENERATOR_SEED,
                                          col_scale=torch.ones(nw, device=device))
    e_unw, e_ones = lfsr_core.rel_err(ww, d_unw), lfsr_core.rel_err(ww, d_ones)
    check("weighted fit: unit weights == unweighted fit",
          torch.allclose(d_unw, d_ones, atol=1e-5)
          and round(e_unw, 6) == round(e_ones, 6),
          f"rel_err {e_unw:.6f} vs {e_ones:.6f}, "
          f"max|d|={(d_unw - d_ones).abs().max().item():.2e}")

    # 14c. the objective it claims to minimise, it minimises
    d_wtd, _ = lfsr_core.seed_fit_tensor(ww, C, P, sw, GENERATOR_SEED,
                                         col_scale=scale_w)

    def _werr(d: torch.Tensor) -> float:
        num = ((ww.float() - d) * sn.unsqueeze(0)).norm()
        return float((num / (ww.float() * sn.unsqueeze(0)).norm()).item())

    w_unw, w_wtd = _werr(d_unw), _werr(d_wtd)
    e_wtd = lfsr_core.rel_err(ww, d_wtd)
    check("weighted fit beats unweighted on the weighted objective",
          w_wtd < w_unw and e_wtd > e_unw,
          f"weighted rel_err {w_unw:.4f} -> {w_wtd:.4f} "
          f"({100 * (1 - w_wtd / w_unw):.1f}% lower); plain rel_err "
          f"{e_unw:.4f} -> {e_wtd:.4f} (worse, as intended)")

    # 14d. determinism: same generator_seed + same weighting => same bits
    d_wtd2, _ = lfsr_core.seed_fit_tensor(ww, C, P, sw, GENERATOR_SEED,
                                          col_scale=scale_w)
    check("weighted fit is deterministic", torch.equal(d_wtd, d_wtd2))

    # 14e. refit-reuse stays exact under weighting (the cache shortcut's
    # legality, re-proved for the weighted objective)
    side_w = salience.build_side_channel(ww, "awq", 1.0, act_scale=scale_w)
    masked_w = ww.flatten().float().clone()
    masked_w[side_w.idx] = 0.0
    full_w, _ = lfsr_core.seed_fit_tensor(masked_w.view(ww.shape), C, P, sw,
                                          GENERATOR_SEED, col_scale=scale_w)
    full_w = full_w.flatten()
    full_w[side_w.idx] = side_w.values
    spliced_w = lfsr_core.refit_blocks_over(
        d_wtd.flatten().float(), masked_w,
        lfsr_core.block_ids_for_indices(side_w.idx, C), C, P, sw,
        GENERATOR_SEED, col_scale=scale_w)
    spliced_w[side_w.idx] = side_w.values
    check("weighted refit-reuse == weighted full refit",
          torch.equal(spliced_w, full_w),
          f"max|d|={(spliced_w - full_w).abs().max().item():.2e}")

    # 14f. bit accounting is provably identical under either objective: fit the
    # same variant both ways and diff every bpw field of the cache metadata.
    saved_w = (LAYOUT, FIT_WEIGHTING)
    try:
        globals()["LAYOUT"] = "phase3"
        v_seed = Variant("seed_only", "none", 0.0,
                         lfsr_core.bpw_seed_base(C, P), None)
        v_awq = Variant("awq_p1.0", "awq", 1.0,
                        lfsr_core.bpw_seed_base(C, P), "awq")
        metas: dict[str, dict] = {}
        for mode in ("none", "awq"):
            globals()["FIT_WEIGHTING"] = mode
            ctx_w = TensorCtx(name="synthetic", w=ww, act_scale=scale_w)
            base_deq, m_seed = fit_tensor_variant(ctx_w, v_seed, sw, C, P)
            ctx_w.seed_only = base_deq.to(torch.bfloat16).float().flatten().clone()
            _, m_awq = fit_tensor_variant(ctx_w, v_awq, sw, C, P)
            metas[mode] = {"seed_only": m_seed, "awq_p1.0": m_awq}
        bit_fields = ("bpw_base", "side_bits", "bpw_side", "bpw_total", "C", "P",
                      "numel", "rank", "n_seeds")
        same_bits = all(metas["none"][vn][f] == metas["awq"][vn][f]
                        for vn in metas["none"] for f in bit_fields)
        tagged = (metas["awq"]["seed_only"]["fit_weighting"] == "awq"
                  and metas["none"]["seed_only"]["fit_weighting"] == "none")
        moved = (metas["awq"]["awq_p1.0"]["rel_err"]
                 != metas["none"]["awq_p1.0"]["rel_err"])
        check("bit accounting invariant under fit weighting",
              same_bits and tagged and moved,
              f"bpw_total {metas['none']['awq_p1.0']['bpw_total']:.6f} == "
              f"{metas['awq']['awq_p1.0']['bpw_total']:.6f}, "
              f"side_bits {metas['none']['awq_p1.0']['side_bits']}, "
              f"rel_err {metas['none']['awq_p1.0']['rel_err']:.4f} -> "
              f"{metas['awq']['awq_p1.0']['rel_err']:.4f}")
    finally:
        globals()["LAYOUT"], globals()["FIT_WEIGHTING"] = saved_w

    # ------------------------------- W2: weighted coefficient rounding (3.6)
    # 15a. the search can never lose: its candidate set contains the nearest
    # point, so per-block error is monotone non-increasing everywhere, and
    # strictly lower somewhere.  Checked on both metrics.
    def _sse(d: torch.Tensor, s: torch.Tensor | None = None) -> float:
        e = (ww.float() - d).pow(2)
        return float((e if s is None else e * s.unsqueeze(0)).sum().item())

    d_near, _ = lfsr_core.seed_fit_tensor(ww, C, P, sw, GENERATOR_SEED)
    d_srch, _ = lfsr_core.seed_fit_tensor(ww, C, P, sw, GENERATOR_SEED,
                                          rounding="weighted")
    b_near = lfsr_core.to_blocks(ww.flatten().float(), C)
    per_near = (lfsr_core.to_blocks(d_near.flatten().float(), C) - b_near).pow(2).sum(-1)
    per_srch = (lfsr_core.to_blocks(d_srch.flatten().float(), C) - b_near).pow(2).sum(-1)
    check("W2 rounding never loses a block, wins some (plain metric)",
          bool((per_srch <= per_near + 1e-9).all()) and _sse(d_srch) < _sse(d_near)
          and int((per_srch < per_near - 1e-12).sum()) > 0,
          f"SSE {_sse(d_near):.6e} -> {_sse(d_srch):.6e} "
          f"({int((per_srch < per_near - 1e-12).sum())}/{per_near.numel()} "
          f"blocks improved, 0 worsened)")

    # 15b. and the same under W1's weighted metric, which is the metric the
    # search is actually pointed at when both knobs are on
    d_w_near, _ = lfsr_core.seed_fit_tensor(ww, C, P, sw, GENERATOR_SEED,
                                            col_scale=scale_w)
    d_w_srch, _ = lfsr_core.seed_fit_tensor(ww, C, P, sw, GENERATOR_SEED,
                                            col_scale=scale_w,
                                            rounding="weighted")
    s2_full = sn.pow(2)
    check("W2 rounding lowers the weighted objective too",
          _sse(d_w_srch, s2_full) < _sse(d_w_near, s2_full),
          f"weighted SSE {_sse(d_w_near, s2_full):.6e} -> "
          f"{_sse(d_w_srch, s2_full):.6e}")

    # 15c. constructed case: a block whose independently-nearest coefficients
    # are provably not the best grid point.  Built by brute force over the same
    # grid the search uses, so it is an independent oracle, not a restatement.
    torch.manual_seed(11)
    Ub = lfsr_core.make_U(torch.tensor([1234], dtype=torch.int64), C, P, device)[0]
    found = None
    for _ in range(200):
        tgt = torch.randn(1, C, device=device)
        tt = torch.linalg.pinv(Ub) @ tgt[0]
        q, sc_ = lfsr_core.quantize_coeffs(tt)
        e_near = ((Ub @ (q * sc_)) - tgt[0]).pow(2).sum()
        best = None
        for k in range(1 << P):                       # brute force, offset 0
            m = torch.tensor([(k >> i) & 1 for i in range(P)],
                             device=device, dtype=torch.float32)
            qq = (torch.floor(tt / sc_) + m).clamp(-8, 7)
            e = ((Ub @ (qq * sc_)) - tgt[0]).pow(2).sum()
            best = e if best is None else torch.minimum(best, e)
        if float(best) < float(e_near) * (1 - 1e-6):
            found = (tgt, float(e_near), float(best))
            break
    if found is None:
        check("W2 strictly beats nearest on a constructed block", False,
              "no adversarial block found in 200 draws")
    else:
        tgt, e_near, e_brute = found
        got = lfsr_core.refine_coeffs(
            tgt, Ub.unsqueeze(0),
            (torch.linalg.pinv(Ub) @ tgt[0]).unsqueeze(0), None)
        e_got = float((got[0] - tgt[0]).pow(2).sum())
        # Oracle match is relative, not absolute: the brute-force loop and
        # refine_coeffs reduce in different orders, and fp32 at |e|~3.6 only
        # agrees to ~4e-7. An absolute 1e-9 bar passed on one GPU's kernels
        # and failed on the 4090's (2026-08-18).
        check("W2 strictly beats nearest on a constructed block",
              e_got < e_near * (1 - 1e-6) and e_got <= e_brute * (1 + 1e-5),
              f"nearest {e_near:.6e} -> brute-force {e_brute:.6e} -> "
              f"refine_coeffs {e_got:.6e}")

    # 15d. determinism, and inertness when off
    d_srch2, _ = lfsr_core.seed_fit_tensor(ww, C, P, sw, GENERATOR_SEED,
                                           rounding="weighted")
    check("W2 fit is deterministic and `nearest` is byte-identical to before",
          torch.equal(d_srch, d_srch2) and torch.equal(d_near, d_unw),
          "same bits on rerun; nearest == the pre-3.6 path")

    # ------------------------------------ W3: incoherence transform (design A)
    # 16a. H is exactly orthogonal and exactly reproducible.  Tolerance: the
    # transform is log2(L) butterfly stages of fp32 adds plus one 1/sqrt(L)
    # multiply, so error accumulates like sqrt(log2 L) * eps ~ 2e-7 at L=2048;
    # 1e-5 leaves two orders of headroom and would still catch a wrong-axis or
    # wrong-permutation bug (those are O(1) errors, not O(eps)).
    tol = 1e-5
    orth, rt, mixes = {}, {}, {}
    for nn in (8, 96, 132, 2048):
        eye = torch.eye(nn, device=device)
        H = lfsr_core.incoherence_forward(eye, GENERATOR_SEED)
        orth[nn] = float((H.T @ H - eye).abs().max().item())
        rt[nn] = float((lfsr_core.incoherence_inverse(H, GENERATOR_SEED)
                        - eye).abs().max().item())
        # a real mixer, not a signed permutation: every entry is 1/sqrt(L)
        mixes[nn] = float(H.abs().max().item())
    H1 = lfsr_core.incoherence_forward(torch.eye(96, device=device), GENERATOR_SEED)
    H2 = lfsr_core.incoherence_forward(torch.eye(96, device=device), GENERATOR_SEED)
    check("W3 H is orthogonal, invertible and reproducible",
          all(v < tol for v in orth.values()) and all(v < tol for v in rt.values())
          and torch.equal(H1, H2) and mixes[2048] < 0.03,
          f"max|H^T H - I|={max(orth.values()):.2e}, max round-trip="
          f"{max(rt.values()):.2e} (tol {tol:.0e}), max|H_ij| @2048="
          f"{mixes[2048]:.4f} = 1/sqrt(2048)")

    # 16b. THE design-A claim, checked numerically on a real fitted pair:
    #      || (W - What) diag(s) ||_F  ==  || T - That ||_F
    # If the block layout broke the identity this is where it would show.
    s_dec = lfsr_core.decode_col_scale(scale_w, nw, device)
    d_hadA, _ = lfsr_core.seed_fit_tensor(ww, C, P, sw, GENERATOR_SEED,
                                          col_scale=scale_w, incoherence="had")
    lhs = float(((ww.float() - d_hadA) * s_dec).norm().item())
    rhs = float((lfsr_core.incoherence_forward(ww.float() * s_dec, GENERATOR_SEED)
                 - lfsr_core.incoherence_forward(d_hadA * s_dec, GENERATOR_SEED)
                 ).norm().item())
    check("W3 design A: plain L2 in T space IS the weighted metric in W space",
          abs(lhs - rhs) <= 1e-4 * max(lhs, 1e-12),
          f"||(W-What)diag(s)||={lhs:.6f} vs ||T-That||={rhs:.6f} "
          f"(rel {abs(lhs - rhs) / max(lhs, 1e-12):.2e})")

    # 16c. honest accounting: design A's scales are 16 bits per input channel,
    # they show up in bpw_total, and they are the ONLY new bits.
    tb = lfsr_core.transform_side_bits("had", True, (mw, nw))
    check("W3 accounting: scales priced at 16 bits/input channel",
          tb == 16 * nw and abs(tb / (mw * nw) - 16.0 / mw) < 1e-12
          and lfsr_core.transform_side_bits("had", False, (mw, nw)) == 0
          and lfsr_core.transform_side_bits("none", True, (mw, nw)) == 0,
          f"{tb} bits over {mw * nw} weights = {tb / (mw * nw):.4f} bpw "
          f"(= 16/{mw}); unweighted-had and weighted-none both 0")

    # 16d. determinism + refit-reuse under design A.  Held-out weights dirty
    # whole rows here (H is dense), so the block set is the row footprint; the
    # splice matches a full refit to fp32 round-trip precision rather than
    # bit-exactly, because kept blocks return through forward(inverse(.)).
    d_hadB, _ = lfsr_core.seed_fit_tensor(ww, C, P, sw, GENERATOR_SEED,
                                          col_scale=scale_w, incoherence="had")
    ids_plain = lfsr_core.refit_block_ids(side_w.idx, C, ww.shape, "none")
    ids_had = lfsr_core.refit_block_ids(side_w.idx, C, ww.shape, "had")
    full_h, _ = lfsr_core.seed_fit_tensor(masked_w.view(ww.shape), C, P, sw,
                                          GENERATOR_SEED, col_scale=scale_w,
                                          incoherence="had")
    spliced_h = lfsr_core.refit_blocks_over(
        d_hadA.flatten().float(), masked_w, ids_had, C, P, sw, GENERATOR_SEED,
        col_scale=scale_w, incoherence="had", shape=ww.shape)
    dmax = float((spliced_h - full_h.flatten()).abs().max().item())
    check("W3 refit-reuse over the row footprint == full refit",
          torch.equal(d_hadA, d_hadB) and ids_had.numel() >= ids_plain.numel()
          and dmax < 1e-4,
          f"row footprint {ids_had.numel()} blocks vs W-space "
          f"{ids_plain.numel()}; max|d|={dmax:.2e}")

    # 16e. the scale floor actually bounds decode amplification (the failure
    # mode design A has and W1 does not: decode divides by s).
    dead = scale_w.clone()
    dead[3] = 1e-9
    s_floored = lfsr_core.decode_col_scale(dead, nw, device)
    check("W3 scale floor bounds the 1/s decode amplification",
          float(s_floored.min()) >= lfsr_core.INCOHERENCE_SCALE_FLOOR - 1e-9
          and float((1.0 / s_floored).max()) <= 1.0
          / lfsr_core.INCOHERENCE_SCALE_FLOOR + 1e-6
          and float(lfsr_core.normalized_col_scale(dead).min()) < 1e-8,
          f"raw min {float(lfsr_core.normalized_col_scale(dead).min()):.2e} "
          f"-> floored {float(s_floored.min()):.3f}, max amplification "
          f"{float((1.0 / s_floored).max()):.1f}x")

    # 16f. accounting through the runner: W2 must not move a single bit, W3
    # must move exactly the scale bits and nothing else.
    saved_36 = (LAYOUT, FIT_WEIGHTING, COEFF_ROUNDING, INCOHERENCE)
    try:
        globals()["LAYOUT"] = "phase3"
        v_seed = Variant("seed_only", "none", 0.0,
                         lfsr_core.bpw_seed_base(C, P), None)
        m36: dict[str, dict] = {}
        for tag, (wm, rm, hm) in {
                "base": ("awq", "nearest", "none"),
                "w2": ("awq", "weighted", "none"),
                "w3": ("awq", "nearest", "had")}.items():
            globals()["FIT_WEIGHTING"] = wm
            globals()["COEFF_ROUNDING"] = rm
            globals()["INCOHERENCE"] = hm
            ctx36 = TensorCtx(name="synthetic", w=ww, act_scale=scale_w)
            _, m36[tag] = fit_tensor_variant(ctx36, v_seed, sw, C, P)
        bit_fields = ("bpw_base", "side_bits", "bpw_side", "bpw_total",
                      "transform_bits", "C", "P", "numel")
        w2_free = all(m36["base"][f] == m36["w2"][f] for f in bit_fields)
        w3_delta = m36["w3"]["bpw_total"] - m36["base"]["bpw_total"]
        w3_honest = (m36["w3"]["transform_bits"] == 16 * nw
                     and abs(w3_delta - 16.0 / mw) < 1e-12
                     and m36["w3"]["side_bits"] == m36["base"]["side_bits"])
        tagged = (m36["w2"]["coeff_rounding"] == "weighted"
                  and m36["w3"]["incoherence"] == "had"
                  and m36["base"]["coeff_rounding"] == "nearest"
                  and m36["base"]["incoherence"] == "none")
        check("accounting: W2 is free, W3 pays exactly its scales",
              w2_free and w3_honest and tagged
              and m36["w2"]["rel_err"] != m36["base"]["rel_err"],
              f"W2 bpw_total {m36['base']['bpw_total']:.6f} == "
              f"{m36['w2']['bpw_total']:.6f}; W3 +{w3_delta:.6f} bpw "
              f"(= 16/{mw}), rel_err {m36['base']['rel_err']:.4f} -> "
              f"W2 {m36['w2']['rel_err']:.4f} / W3 {m36['w3']['rel_err']:.4f}")
    finally:
        (globals()["LAYOUT"], globals()["FIT_WEIGHTING"],
         globals()["COEFF_ROUNDING"], globals()["INCOHERENCE"]) = saved_36

    # 16g. regression guard, the whole point of the "inert when off" contract:
    # with both new knobs at their defaults every path is byte-identical to the
    # pre-3.6 code, under either fit objective.
    check("W2/W3 defaults are byte-identical to the earlier fits",
          torch.equal(d_near, d_unw) and torch.equal(d_w_near, d_wtd),
          "unweighted and weighted both unchanged at "
          "--coeff-rounding nearest --incoherence none")

    # ------------------------------ A2: calibration-set identity (--calib-id)
    # The activation scales are an input to the objective, not a constant, and
    # until now their identity was implicit.  These checks are the W1 cache
    # lesson applied to that input: a new calibration set must be unable to
    # reach any existing key, path or row name, and the default must be unable
    # to change any of them.
    saved_a2 = (LAYOUT, MODEL_SLUG, CONFIG_NAME, FIT_WEIGHTING,
                COEFF_ROUNDING, INCOHERENCE, CALIB_ID, CALIB_CORPUS, CACHE)
    try:
        globals()["LAYOUT"] = "phase3"
        globals()["MODEL_SLUG"], globals()["CONFIG_NAME"] = "Qwen3-0.6B", "c12p4"
        globals()["FIT_WEIGHTING"] = "none"
        globals()["COEFF_ROUNDING"] = "nearest"
        globals()["INCOHERENCE"] = "none"
        globals()["CACHE"] = Path("cache") / "Qwen3-0.6B"
        tname = "model.layers.0.self_attn.q_proj.weight"
        v_seed = Variant("seed_only", "none", 0.0, 3.0, None)
        v_awq = Variant("awq_p1.0", "awq", 1.0, 3.0, "awq")
        v_cmp = Variant("q4km", "q4km", 0.0, 0.0, None, kind="comparator")

        # 17a. DEFAULT CALIB IS A NO-OP.  Every artefact the project already
        # has on disk must come back character-identical, and the key must
        # equal the literal pre-A2 formula, not merely "whatever the code says".
        globals()["CALIB_ID"] = DEFAULT_CALIB_ID
        k_def = variant_cache_key(v_awq, tname, 512, 12, 4)
        path_def = act_scales_path()
        rows_def = (config_slug(None), config_slug("none"), config_slug("awq"),
                    row_name(v_seed), row_name(v_awq))
        check("calib: the default calibration set changes nothing",
              k_def == cache_key(tname, 12, 4, 512, "awq", 1.0, GENERATOR_SEED,
                                 "Qwen3-0.6B", "c12p4")
              and path_def.name == "act_scales.safetensors"
              and rows_def == ("c12p4", "c12p4", "c12p4",
                               "c12p4_seed_only", "c12p4_awq_p1.0"),
              f"key={k_def} scales={path_def.name} slugs/rows={rows_def}")

        # 17b. A NEW CALIB SET IS A NEW NAMESPACE — for the rules that read the
        # scales, and only those.  With weighting off, `seed_only` does not read
        # them and must keep sharing the historical entry; `awq_p1.0` ranks its
        # outliers by |w|*s and must not.
        globals()["CALIB_ID"] = "wt256"
        k_seed_off = variant_cache_key(v_seed, tname, 512, 12, 4)
        k_awq_off = variant_cache_key(v_awq, tname, 512, 12, 4)
        k_cmp_new = variant_cache_key(v_cmp, tname, 512, 12, 4)
        globals()["CALIB_ID"] = DEFAULT_CALIB_ID
        k_seed_p12 = variant_cache_key(v_seed, tname, 512, 12, 4)
        k_cmp_p12 = variant_cache_key(v_cmp, tname, 512, 12, 4)
        globals()["CALIB_ID"] = "wt256"
        check("calib: only scale-reading rules get a new namespace",
              k_awq_off != k_def and k_seed_off == k_seed_p12
              and k_cmp_new == k_cmp_p12
              and config_slug("awq") == "c12p4@wt256"
              and config_slug("none") == "c12p4"
              and row_name(v_awq) == "c12p4@wt256_awq_p1.0",
              f"awq {k_def}->{k_awq_off}, seed_only shared={k_seed_off == k_seed_p12}, "
              f"comparator shared={k_cmp_new == k_cmp_p12}, "
              f"row {row_name(v_awq)}")

        # ...and with weighting ON every fitted tensor reads the scales, so the
        # whole run moves namespace, including `seed_only`.
        globals()["FIT_WEIGHTING"] = "awq"
        check("calib: a weighted run moves every seed rule's namespace",
              config_slug("none") == "c12p4w@wt256"
              and config_slug(None) == "c12p4w@wt256"
              and variant_cache_key(v_cmp, tname, 512, 12, 4) == k_cmp_p12
              and act_scales_path().name == "act_scales@wt256.safetensors",
              f"slug={config_slug(None)} scales={act_scales_path().name}")

        # 17c. All 8 fit-knob combinations x 2 calibration sets = 16 distinct
        # namespaces, and the 8 at the default calib are byte-identical to the
        # 8 the selftest above pinned.  This is the check that stops
        # a wt256 fit from ever being served a p12 reconstruction.
        keys16, slugs_p12 = [], []
        for cal in (DEFAULT_CALIB_ID, "wt256"):
            globals()["CALIB_ID"] = cal
            for wm in ("none", "awq"):
                for rm in ("nearest", "weighted"):
                    for hm in ("none", "had"):
                        globals()["FIT_WEIGHTING"] = wm
                        globals()["COEFF_ROUNDING"] = rm
                        globals()["INCOHERENCE"] = hm
                        keys16.append(variant_cache_key(v_awq, tname, 512, 12, 4))
                        if cal == DEFAULT_CALIB_ID:
                            slugs_p12.append(config_slug("awq"))
        expect8 = ["c12p4", "c12p4h", "c12p4r", "c12p4rh",
                   "c12p4w", "c12p4wh", "c12p4wr", "c12p4wrh"]
        check("calib: 8 fit knobs x 2 calibration sets = 16 distinct keys",
              len(set(keys16)) == 16 and sorted(slugs_p12) == expect8,
              f"{len(set(keys16))}/16 distinct; p12 slugs {sorted(slugs_p12)}")
    finally:
        (globals()["LAYOUT"], globals()["MODEL_SLUG"], globals()["CONFIG_NAME"],
         globals()["FIT_WEIGHTING"], globals()["COEFF_ROUNDING"],
         globals()["INCOHERENCE"], globals()["CALIB_ID"],
         globals()["CALIB_CORPUS"], globals()["CACHE"]) = saved_a2

    # 17d. The corpus loader is the thing that gives a calib id its meaning, so
    # its rule has to be deterministic and its hash order-sensitive.
    with tempfile.TemporaryDirectory() as td:
        corpus = Path(td) / "wiki.test.raw"
        long_a = "Alpha " * 60          # 360 chars
        long_b = "Beta " * 60
        long_c = "Gamma " * 60
        corpus.write_text("\n".join([
            " = Heading one = ", "", "short line", long_a, " = Sub = ",
            long_b, "   ", long_c]), encoding="utf-8")
        got2 = swap_eval.load_calib_corpus(corpus, 2)
        got3 = swap_eval.load_calib_corpus(corpus, 3)
        too_many = False
        try:
            swap_eval.load_calib_corpus(corpus, 4)
        except ValueError:
            too_many = True
        h_fwd = swap_eval.corpus_sha256(got3)
        h_rev = swap_eval.corpus_sha256(list(reversed(got3)))
        check("calib: corpus loader is deterministic, filtered and ordered",
              got2 == [long_a.strip(), long_b.strip()]
              and got3 == [long_a.strip(), long_b.strip(), long_c.strip()]
              and got3[:2] == got2 and too_many
              and h_fwd == swap_eval.corpus_sha256(got3) and h_fwd != h_rev,
              f"{len(got3)} docs kept of 8 lines, headings/short dropped, "
              f"short corpus raises={too_many}, sha {h_fwd[:12]}…")

    # 17f. The drift readout is what section 5 of the runbook decides on, so
    # its arithmetic gets a test of its own: identical scales must read as no
    # drift, and a known perturbation of exactly one channel in one tensor must
    # show up in both the cosine and the >2x channel count.
    saved_drift = (CACHE, ACT_SCALES, CALIB_ID)
    with tempfile.TemporaryDirectory() as td:
        try:
            globals()["CACHE"] = Path(td)
            base = {"t.a": torch.rand(64) + 0.5, "t.b": torch.rand(32) + 0.5}
            save_file({k: v.contiguous() for k, v in base.items()},
                      str(Path(td) / "act_scales.safetensors"))
            same = {k: v.clone() for k, v in base.items()}
            save_file({k: v.contiguous() for k, v in same.items()},
                      str(Path(td) / "act_scales@same.safetensors"))
            moved = {k: v.clone() for k, v in base.items()}
            moved["t.a"][0] *= 50.0                    # one channel, 50x up
            save_file({k: v.contiguous() for k, v in moved.items()},
                      str(Path(td) / "act_scales@moved.safetensors"))
            globals()["CALIB_ID"] = "same"
            globals()["ACT_SCALES"] = act_scales_path()
            d_same = act_scale_drift()
            globals()["CALIB_ID"] = "moved"
            globals()["ACT_SCALES"] = act_scales_path()
            d_move = act_scale_drift()
            globals()["CALIB_ID"] = DEFAULT_CALIB_ID
            globals()["ACT_SCALES"] = act_scales_path()
            d_self = act_scale_drift()          # p12 vs itself -> no comparison
            check("calib: act-scale drift readout reflects real movement",
                  d_same is not None and d_move is not None and d_self is None
                  and abs(d_same["cosine_median"] - 1.0) < 1e-6
                  and d_same["frac_channels_2x_max"] == 0.0
                  and d_move["cosine_min"] < d_same["cosine_min"]
                  and d_move["frac_channels_2x_max"] > 0.0
                  and d_move["tensors"] == 2,
                  f"identical: cos={d_same['cosine_median']:.6f} "
                  f"moved>2x={d_same['frac_channels_2x_max']:.3f}; perturbed: "
                  f"cos_min={d_move['cosine_min']:.4f} "
                  f"moved>2x={d_move['frac_channels_2x_max']:.3f}")
        finally:
            (globals()["CACHE"], globals()["ACT_SCALES"],
             globals()["CALIB_ID"]) = saved_drift

    # 17e. Bad calibration identities fail before anything is written.  As with
    # the 3.6 knobs, configure_layout validates before it mutates, so a raise
    # leaves the process's layout untouched and nothing needs restoring.
    refused_calib = []
    for label, kwargs in (("bad chars", {"calib_id": "wt-256"}),
                          ("empty", {"calib_id": ""}),
                          ("too long", {"calib_id": "a" * 13}),
                          ("uppercase", {"calib_id": "WT256"}),
                          ("no corpus", {"calib_id": "wt256"})):
        try:
            configure_layout(MODEL_DIR, "c12p4", **kwargs)
        except ValueError:
            refused_calib.append(label)
    legacy_refused = False
    try:
        configure_layout(None, "c8p3", calib_id="wt256",
                         calib_corpus=Path("x.txt"))
    except ValueError:
        legacy_refused = True
    check("calib: malformed ids, missing corpus and legacy layout all refused",
          len(refused_calib) == 5 and legacy_refused,
          f"refused {refused_calib}, legacy refused={legacy_refused}")

    log(f"SELFTEST {'ALL PASS' if ok_all else 'FAILURES PRESENT'} "
        f"in {time.time() - t0:.1f}s")
    return ok_all


# ------------------------------------------------------------------- main
def pick_winner(rows: list[dict]) -> str | None:
    """Lowest mean KL among the salience variants, budget-matched.

    Mirrors :func:`gate_verdict`: variants spending more bpw than the rtn4
    comparator cannot be the winner (the matched-effective-bpw rule), because
    the stage-2 refit of the winner is what the final verdict is computed on.
    Falls back to the unconstrained pool if no variant fits the budget.
    """
    cands = [r for r in rows if r.get("salience_rule") in SALIENCE_RULES]
    if not cands:
        return None
    rtn = next((r for r in rows
                if r.get("base_name", r["name"]) == "rtn4"), None)
    rtn_bpw = rtn.get("bpw_total") if rtn else None
    if rtn_bpw is not None:
        eligible = [r for r in cands
                    if r.get("bpw_total") is not None
                    and r["bpw_total"] <= rtn_bpw + 1e-6]
        cands = eligible or cands
    best = min(cands, key=lambda r: r["mean_kl"])
    return best.get("base_name", best["name"])


def pick_winner_phase3(rows: list[dict]) -> str | None:
    """Row name of the Phase 3 stage-1 winner.

    Phase 3's gates are absolute (against q3km/q4km at their own bpw), not
    relative to a proxy, so the winner is simply the lowest-KL salience variant
    across whichever configs have been evaluated.  Ties break toward fewer bits,
    which is the direction both gate clauses reward.
    """
    cands = [r for r in rows
             if r.get("kind") == "seed"
             and r.get("salience_rule") in SALIENCE_RULES
             and r.get("mean_kl") is not None]
    if not cands:
        return None
    return min(cands, key=lambda r: (r["mean_kl"], r.get("bpw_total", 0.0)))["name"]


def main() -> None:
    ap = argparse.ArgumentParser(description="SeedLM+O Phase 2/3 runner")
    ap.add_argument("--stage", choices=["fit", "eval", "refit-full", "all",
                                        "comparators", "lmeval", "verdict",
                                        "equivalence", "calib"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model-dir", default=None,
                    help="model directory; omit for the Phase 2 legacy layout "
                         "(cache/ + results/ un-namespaced, Phase 2 matrix)")
    ap.add_argument("--config", default=None, choices=sorted(CONFIGS),
                    help="seed config: c8p3 (4.0 bpw base) or c12p4 (3.0)")
    ap.add_argument("--fit-weighting", default="none",
                    choices=list(lfsr_core.FIT_WEIGHTINGS),
                    help="fit objective: none = plain per-block L2 (default, "
                         "bit-for-bit the historical behaviour), awq = "
                         "activation-weighted least squares using the cached "
                         "act_scales. Weighted runs get a `w`-suffixed config "
                         "slug (c12p4w), hence their own cache namespace and "
                         "their own results rows at identical bpw. Needs "
                         "--model-dir.")
    ap.add_argument("--coeff-rounding", default="nearest",
                    choices=list(lfsr_core.COEFF_ROUNDINGS),
                    help="coefficient quantizer (W2): nearest = each "
                         "coefficient to its own nearest 4-bit grid point "
                         "(default, the historical behaviour), weighted = exact "
                         "search over 3 exponents x 2^P rounding patterns for "
                         "the grid point minimising the fit's own metric. Zero "
                         "bit cost, decode unchanged; adds an `r` to the config "
                         "slug (c12p4r / c12p4wr). Needs --model-dir.")
    ap.add_argument("--incoherence", default="none",
                    choices=list(lfsr_core.INCOHERENCE_MODES),
                    help="incoherence preprocessing (W3, design A): "
                         "none = fit W as it stands (default), had = fit "
                         "T = W diag(s) H for a seeded orthogonal H and decode "
                         "through H^T diag(s)^-1. NOT free: the per-input-"
                         "channel scales become stored side info (16/m bpw, "
                         "priced in every row). Adds an `h` to the config slug. "
                         "Needs --model-dir.")
    ap.add_argument("--calib-id", default=DEFAULT_CALIB_ID,
                    help=f"identity of the activation-scale calibration set. "
                         f"Default `{DEFAULT_CALIB_ID}` "
                         f"= the built-in 12 chat prompts every existing result "
                         f"was calibrated on; it reproduces today's cache keys, "
                         f"row names and act_scales path byte-for-byte. Any "
                         f"other id (1-12 lowercase alphanumerics) requires "
                         f"--calib-corpus, captures into its own "
                         f"`act_scales@id.safetensors`, and appends `@id` to "
                         f"the config slug of every rule that reads activation "
                         f"scales — so the 12-prompt scales, and every fit that "
                         f"depends on them, are never touched. Needs "
                         f"--model-dir.")
    ap.add_argument("--calib-corpus", default=None,
                    help="text file the non-default calibration set is read "
                         "from: one paragraph per line, headings and short "
                         "lines skipped (wikitext-2-raw/wiki.test.raw is the "
                         "intended input — the same corpus run_phase3_pod.sh "
                         "already fetches for the llama.cpp imatrix). Its "
                         "sha256 is stored with the captured scales and "
                         "re-checked on every reuse.")
    ap.add_argument("--calib-prompts", type=int, default=CALIB_N_PROMPTS,
                    help="documents to take from --calib-corpus (default "
                         f"{CALIB_N_PROMPTS})")
    ap.add_argument("--calib-max-tokens", type=int, default=CALIB_MAX_TOKENS,
                    help="per-document truncation length for --calib-corpus "
                         f"(default {CALIB_MAX_TOKENS}), so no single long "
                         f"document dominates the mean")
    ap.add_argument("--only", default=None,
                    help="comma-separated variant names to fit (e.g. "
                         "`seed_only,awq_p1.0`); the rest are left to whatever "
                         "is already cached and are dropped from eval with a "
                         "WARN. The cost lever for the expensive W2/W3 search "
                         "rungs. `seed_only` is added automatically when a "
                         "scattered rule needs it as a refit base.")
    ap.add_argument("--n-seeds", type=int, default=STAGE1_SEEDS,
                    help="stage-1 candidate seeds per block")
    ap.add_argument("--stage2-seeds", type=int, default=None,
                    help="winner-refit seeds (default 65535; when --n-seeds is "
                         "given below the stage-1 default, defaults to "
                         "min(65535, 4*n_seeds) so smoke runs stay bounded)")
    ap.add_argument("--layers-limit", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--export", default=None, help="variant name to save as a dir")
    ap.add_argument("--eval-mode", choices=["dual", "cached"], default=None,
                    help="reference logits: recomputed by a second resident "
                         "model (dual) or streamed from disk (cached, the "
                         "Phase 3 default)")
    ap.add_argument("--gguf-q4km", default=None, help="Q4_K_M GGUF path")
    ap.add_argument("--gguf-q3km", default=None, help="Q3_K_M GGUF path")
    ap.add_argument("--awq-dir", default=None, help="AWQ W4A16 checkpoint dir")
    ap.add_argument("--lmeval-tasks", default="gsm8k,ifeval")
    ap.add_argument("--lmeval-limit", type=int, default=200)
    ap.add_argument("--equiv-variant", default="rtn4")
    ap.add_argument("--print-winner", action="store_true",
                    help="print the stage-1 winner from the stored results as "
                         "`WINNER_ROW=… WINNER_BASE=… WINNER_CONFIG=…` and exit; "
                         "this is how the launchers hand the winner between "
                         "machines (no cache transfer needed — fits are "
                         "deterministic from generator_seed)")
    ap.add_argument("--C", type=int, default=None)
    ap.add_argument("--P", type=int, default=None)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model_dir = Path(args.model_dir) if args.model_dir else None
    if model_dir is not None and not model_dir.exists():
        raise SystemExit(f"--model-dir does not exist: {model_dir}")
    config_name = args.config or "c8p3"
    try:
        configure_layout(model_dir, config_name, args.fit_weighting,
                         args.coeff_rounding, args.incoherence,
                         args.calib_id,
                         Path(args.calib_corpus) if args.calib_corpus else None,
                         args.calib_prompts, args.calib_max_tokens)
    except ValueError as exc:
        raise SystemExit(str(exc))
    only = ([s.strip() for s in args.only.split(",") if s.strip()]
            if args.only else None)

    if args.C is not None or args.P is not None:
        if args.config:
            raise SystemExit("pass --config or --C/--P, not both")
        C = args.C if args.C is not None else C_DEFAULT
        P = args.P if args.P is not None else P_DEFAULT
    else:
        C, P = CONFIGS[config_name]
    eval_mode = args.eval_mode or ("cached" if LAYOUT == "phase3" else "dual")
    phase3 = LAYOUT == "phase3"

    RESULTS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    run_id = ("phase3-" if phase3 else "phase2-") + time.strftime("%Y%m%d-%H%M%S")
    stage2 = args.stage2_seeds
    if stage2 is None:
        stage2 = STAGE2_SEEDS if args.n_seeds >= STAGE1_SEEDS \
            else min(STAGE2_SEEDS, args.n_seeds * 4)

    if args.selftest:
        ok = selftest(args.device, C_DEFAULT, P_DEFAULT)
        raise SystemExit(0 if ok else 1)
    if args.print_winner:
        prev = load_results()
        if not prev:
            print(f"WINNER_ROW= WINNER_BASE= WINNER_CONFIG=  # no "
                  f"{RESULTS / RESULTS_JSON}")
            raise SystemExit(2)
        rows = prev["variants"]
        wrow = pick_winner_phase3(rows) if phase3 else pick_winner(rows)
        if not wrow:
            print("WINNER_ROW= WINNER_BASE= WINNER_CONFIG=  # no salience rows")
            raise SystemExit(2)
        hit = next((r for r in rows if r["name"] == wrow), None)
        # WINNER_CONFIG must stay a legal --config value (the pod launcher
        # feeds it straight back in), so it reports the *base* config even when
        # an activation-weighted row wins; the weighting travels separately.
        cfg_out = (hit or {}).get("config_base") or (hit or {}).get("config", "-")
        print(f"WINNER_ROW={wrow} "
              f"WINNER_BASE={(hit or {}).get('base_name', wrow)} "
              f"WINNER_CONFIG={cfg_out} "
              f"WINNER_WEIGHTING={(hit or {}).get('fit_weighting', 'none')}")
        raise SystemExit(0)
    if not args.stage:
        ap.error("one of --selftest or --stage is required")

    variants = build_matrix_phase3(C, P) if phase3 else build_matrix()
    resume = not args.no_resume
    log(f"RUN {run_id} layout={LAYOUT} model={MODEL_SLUG} "
        f"config={config_slug()} fit_weighting={FIT_WEIGHTING} "
        f"coeff_rounding={COEFF_ROUNDING} incoherence={INCOHERENCE} "
        f"calib_id={CALIB_ID} "
        f"C={C} P={P} stage={args.stage} n_seeds={args.n_seeds} "
        f"stage2_seeds={stage2} layers_limit={args.layers_limit} "
        f"eval_mode={eval_mode} resume={resume} stack={swap_eval.stack_id()}")
    ok0, digest = verify_originals(record=True)
    log(f"HASH originals sha256={digest[:32]}... verified={ok0}")
    if not ok0:
        raise SystemExit("FATAL: models/originals shard hash mismatch")

    def cfg_now(configs_present: list[str] | None = None) -> dict:
        return make_config(args.n_seeds, stage2, args.layers_limit, args.device,
                           C, P, digest, ok0, eval_mode, configs_present)

    def persist(rows: list[dict], gates: dict,
                configs_present: list[str] | None = None) -> None:
        cfg = cfg_now(configs_present)
        write_results(rows, gates, cfg, run_id)
        if phase3:
            write_summary_phase3(rows, gates, cfg, run_id)
        else:
            write_summary(rows, gates, cfg, run_id)

    def merged_rows(new: list[dict]) -> list[dict]:
        prev = load_results()
        return merge_rows(prev["variants"] if prev else [], new)

    def configs_present(rows: list[dict]) -> list[str]:
        return sorted({r.get("config") for r in rows
                       if r.get("config") and r["config"] != "-"})

    if args.stage == "calib":
        stage_calib(args.device)
        raise SystemExit(0)

    if args.stage == "equivalence":
        ok = stage_equivalence(args.layers_limit, args.device, C, P,
                               args.n_seeds, args.equiv_variant)
        raise SystemExit(0 if ok else 1)

    if args.stage == "comparators":
        avail = stage_comparators(
            variants, args.layers_limit, args.device, resume,
            Path(args.gguf_q4km) if args.gguf_q4km else None,
            Path(args.gguf_q3km) if args.gguf_q3km else None,
            Path(args.awq_dir) if args.awq_dir else None)
        log(f"CMP available: {avail or 'none'}")
        cmp_variants = [v for v in variants if v.kind == "comparator"]
        if cmp_variants:
            out = stage_eval(cmp_variants, args.n_seeds, args.layers_limit,
                             args.device, C, P, eval_mode=eval_mode)
            rows = merged_rows(out["rows"])
            try:
                gates = gates_for_stacks(rows)
            except ValueError as exc:
                log(f"WARN gates not computed: {exc}")
                gates = {"phase3_verdict": "INCOMPLETE", "error": str(exc)}
            persist(rows, gates, configs_present(rows))
            log(f"GATE {gates.get('phase3_verdict')}")

    elif args.stage == "lmeval":
        prev = load_results()
        if not prev:
            raise SystemExit(f"lmeval needs {RESULTS / RESULTS_JSON}")
        rows = stage_lmeval(prev["variants"], args.layers_limit, args.device,
                            C, P, args.n_seeds,
                            [t.strip() for t in args.lmeval_tasks.split(",") if t.strip()],
                            args.lmeval_limit, variants)
        gates = gates_for_stacks(rows)
        persist(rows, gates, configs_present(rows))
        log(f"GATE {gates['phase3_verdict']} benchmark_consistent="
            f"{gates['benchmark_consistent']}")

    elif args.stage == "verdict":
        prev = load_results()
        if not prev:
            raise SystemExit(f"verdict needs {RESULTS / RESULTS_JSON}")
        rows = prev["variants"]
        gates = (gates_for_stacks(rows) if phase3 else gate_verdict(rows))
        persist(rows, gates, configs_present(rows))
        log(f"GATE {gates.get('phase3_verdict') or gates.get('phase2_verdict')}")

    else:
        if args.stage in ("fit", "all"):
            stage_fit(variants, args.n_seeds, args.layers_limit, args.device,
                      resume, C, P, only=only)

        if args.stage in ("eval", "all"):
            out = stage_eval(variants, args.n_seeds, args.layers_limit,
                             args.device, C, P, export=args.export,
                             eval_mode=eval_mode)
            rows = merged_rows(out["rows"]) if phase3 else out["rows"]
            gates = gates_for_stacks(rows) if phase3 else gate_verdict(rows)
            persist(rows, gates, configs_present(rows))
            if phase3:
                log(f"GATE {gates['phase3_verdict']} "
                    f"pass_a={gates['pass_a']} pass_b={gates['pass_b']} "
                    f"best_seed={gates['best_seed_variant']} "
                    f"@{_fmt(gates['best_seed_bpw'], 4)} bpw")
            else:
                log(f"GATE {gates['phase2_verdict']} best={gates['best_variant']} "
                    f"kl_red={gates['kl_reduction_pct']:.2f}% "
                    f"gap={_fmt(gates['gap_closure_pct'], 2)}%")

            if args.stage == "all":
                winner_row = (pick_winner_phase3(rows) if phase3
                              else pick_winner(rows))
                winner = next((r["base_name"] for r in rows
                               if r["name"] == winner_row), winner_row)
                if winner:
                    log(f"WINNER {winner} -> refit @{stage2} seeds")
                    stage_fit(variants, stage2, args.layers_limit, args.device,
                              resume, C, P, only=[winner, "seed_only"])
                    keep = {winner, "seed_only", "rtn4", "bf16"}
                    if phase3:
                        keep |= {"q4km", "q3km", "awq_w4"}
                    wv = [v for v in variants if v.name in keep]
                    suffix = {v.name: f"@{stage2}"
                              for v in wv if v.kind == "seed"}
                    out2 = stage_eval(wv, stage2, args.layers_limit, args.device,
                                      C, P, label_suffix=suffix,
                                      eval_mode=eval_mode)
                    rows2 = out2["rows"]
                    if phase3:
                        merged = merge_rows(rows, rows2)
                        gates2 = gates_for_stacks(merged)
                        gates2["stage"] = f"stage2 @{stage2} seeds"
                        persist(merged, gates2, configs_present(merged))
                        log(f"GATE(stage2) {gates2['phase3_verdict']} "
                            f"pass_a={gates2['pass_a']} pass_b={gates2['pass_b']} "
                            f"best_seed={gates2['best_seed_variant']}")
                    else:
                        merged = rows + rows2
                        # Verdict from the stage-2 numbers, but keep stage 1's
                        # three-rule ranking: only the winner is refit, so rows2
                        # alone cannot rank the rules against each other.
                        if any(r.get("salience_rule") in SALIENCE_RULES
                               for r in rows2):
                            gates2 = gate_verdict(rows2)
                            gates2["stage"] = f"stage2 @{stage2} seeds"
                            gates2["salience_ranking"] = gates["salience_ranking"]
                            gates2["salience_ranking_detail"] = \
                                gates["salience_ranking_detail"]
                            gates2["salience_ranking_stage"] = \
                                f"stage1 @{args.n_seeds} seeds"
                        else:
                            gates2 = gates
                        persist(merged, gates2)
                        log(f"GATE(stage2) {gates2['phase2_verdict']} "
                            f"best={gates2['best_variant']} "
                            f"kl_red={gates2['kl_reduction_pct']:.2f}%")

        elif args.stage == "refit-full":
            prev = load_results()
            if not prev:
                raise SystemExit(f"refit-full needs {RESULTS / RESULTS_JSON} "
                                 "(run --stage eval first)")
            rows = prev["variants"]
            winner_row = pick_winner_phase3(rows) if phase3 else pick_winner(rows)
            winner = next((r["base_name"] for r in rows
                           if r["name"] == winner_row), winner_row)
            if not winner:
                raise SystemExit("no salience variant in previous results")
            log(f"WINNER {winner} -> refit @{stage2} seeds")
            stage_fit(variants, stage2, args.layers_limit, args.device, resume,
                      C, P, only=[winner, "seed_only"])

    ok1, d1 = verify_originals(record=False)
    log(f"HASH re-verify originals verified={ok1} sha256={d1[:32]}...")
    if not ok1:
        raise SystemExit("FATAL: models/originals modified during run")
    log(f"RUN {run_id} complete")


if __name__ == "__main__":
    main()
