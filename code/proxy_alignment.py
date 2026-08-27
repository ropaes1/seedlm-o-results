"""Mine every measured variant for the fit-time proxy that predicts measured KL.

Phase 3.6 rung 1 measured a Goodhart point: W2 lowered the activation-weighted L2
objective by 1.7% and KL got 11% *worse*.  Before anyone designs a "better
objective", this script asks the cheap question first — across everything the
project has already measured, which fit-time number actually ranks behavioural
damage, and where exactly does each one stop working?

Everything here is CPU-only analysis of files that already exist:

* ``results/*/phase3_results.json``               — laptop-stack rows
* ``.pod_results/results/*/phase3_results.json``  — pod-stack rows (W1 and W2)
* ``cache/{slug}/*.safetensors`` metadata         — per-tensor ``rel_err`` and
  ``weighted_rel_err``, used for (i) fit-coverage hygiene and (ii) back-filling
  the weighted proxy for rows fitted before that field existed.

No model is loaded, no fit is run, nothing outside ``docs/experiment/`` is
written.  The back-fill reads cached reconstructions and the immutable
originals and applies ``runner._weighted_rel_err`` itself, so a back-filled
number is the same quantity a fresh fit would have logged.

    python proxy_alignment.py [--out docs/experiment/PROXY-ALIGNMENT.md]
                              [--no-backfill]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open

import runner
import swap_eval

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "docs" / "experiment" / "PROXY-ALIGNMENT.md"

# Result files to mine.  ``.pod_results`` is where the pod's uploaded rows land;
# it holds the only measured W1 (``c12p4w``) and W2 (``c12p4wr``) rows.
RESULT_GLOBS: tuple[str, ...] = (
    "results/*/phase3_results.json",
    ".pod_results/results/*/phase3_results.json",
)

PROXY_LABEL = {"rel": "plain rel_err", "wrel": "act-weighted rel_err"}


# --------------------------------------------------------------- statistics
# scipy is NOT in this venv (verified 2026-08-19) and a results-mining script
# should run wherever the harness runs, so both coefficients and an *exact*
# permutation p-value are implemented here.  Every split in this document is
# small by construction, so the exact test is always affordable and no sampled
# approximation is ever silently substituted for it.
def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs)


def pearson(x: list[float], y: list[float]) -> float | None:
    """Product-moment correlation, or None if either series is constant."""
    if len(x) < 3:
        return None
    mx, my = _mean(x), _mean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def _ranks(xs: list[float]) -> list[float]:
    """Ascending ranks with ties averaged (the Spearman convention)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def spearman(x: list[float], y: list[float]) -> float | None:
    """Rank correlation (Pearson of the average ranks)."""
    if len(x) < 3:
        return None
    return pearson(_ranks(x), _ranks(y))


def perm_p(x: list[float], y: list[float], max_exact: int = 8) -> float | None:
    """Exact two-sided permutation p-value for Spearman's rho.

    Returns None above ``max_exact`` rather than sampling: an approximate
    p-value that looks exact is the kind of thing this project's own lessons
    file exists to prevent.
    """
    n = len(x)
    rho = spearman(x, y)
    if rho is None or n < 3 or n > max_exact:
        return None
    rx, ry = _ranks(x), _ranks(y)
    hits = total = 0
    for perm in itertools.permutations(range(n)):
        r = pearson(rx, [ry[i] for i in perm])
        total += 1
        if r is not None and abs(r) >= abs(rho) - 1e-12:
            hits += 1
    return hits / total


def stratified_sign_p(groups: list[tuple[list[float], list[float]]]
                      ) -> tuple[float | None, float | None]:
    """Mean within-group Spearman and its exact stratified permutation p-value.

    Each group is permuted independently (the null: inside a group of rows that
    all store the same number of bits, the proxy ordering carries no
    information about the KL ordering).  Groups here are triples, so the
    enumeration is 6**k and stays exact for k <= 6.
    """
    per_group = []
    for xs, ys in groups:
        if len(xs) < 3:
            continue
        rho = spearman(xs, ys)
        if rho is not None:
            per_group.append((xs, ys, rho))
    if not per_group:
        return None, None
    observed = _mean(r for _, _, r in per_group)
    if len(per_group) > 6 or any(len(x) > 4 for x, _, _ in per_group):
        return observed, None
    space = []
    for xs, ys, _ in per_group:
        rx, ry = _ranks(xs), _ranks(ys)
        space.append([pearson(rx, [ry[i] for i in p])
                      for p in itertools.permutations(range(len(xs)))])
    hits = total = 0
    for combo in itertools.product(*space):
        vals = [c for c in combo if c is not None]
        total += 1
        if vals and abs(_mean(vals)) >= abs(observed) - 1e-12:
            hits += 1
    return observed, (hits / total if total else None)


def elasticity(p0: float | None, p1: float | None,
               kl0: float, kl1: float) -> float | None:
    """``d log KL / d log proxy`` between two measured points.

    Units-free: "how much KL moved per unit of proxy movement".  A working
    proxy shows a moderate positive value; a *blind* proxy shows a huge
    positive one (KL moved, the proxy barely did); a Goodharted proxy shows a
    negative one (the proxy improved, KL got worse).
    """
    if p0 is None or p1 is None or min(p0, p1, kl0, kl1) <= 0:
        return None
    dp = math.log(p1 / p0)
    if abs(dp) < 1e-12:
        return None
    return math.log(kl1 / kl0) / dp


def fmt(v: float | None, nd: int = 4) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def pct(v: float | None, nd: int = 1) -> str:
    return "—" if v is None else f"{v * 100:+.{nd}f}%"


def binom_two_sided(k: int, n: int) -> float:
    """Two-sided sign-test p-value for k successes in n fair-coin trials."""
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


# --------------------------------------------------------------------- rows
@dataclass
class Row:
    """One measured variant: bits, behaviour, and both fit-time proxies."""

    model: str
    stacks: list[str]
    config: str                 # config slug, e.g. c12p4w
    variant: str                # base variant name, e.g. awq_p1.0
    kind: str
    n_seeds: int
    bpw: float
    kl: float
    rel: float | None
    wrel: float | None
    wrel_source: str            # "results" | "cache n/m" | "-"
    fit_weighting: str
    coeff_rounding: str
    incoherence: str
    row_name: str
    eff_bytes: float | None = None
    sources: list[str] = field(default_factory=list)
    coverage: str = "unknown"   # local fit cache, informational only
    fitted_tensors: int | None = None    # implied by the row's own bytes field
    excluded: str = ""          # non-empty => dropped, with the reason

    @property
    def family(self) -> str:
        """(config slug, seed budget) — the group a proxy is *meant* to rank."""
        return f"{self.config}@{self.n_seeds}"

    @property
    def label(self) -> str:
        return f"{self.config}_{self.variant}" if self.kind == "seed" else self.variant

    @property
    def host(self) -> str:
        return self.stacks[0].split("|")[0]


def load_rows() -> tuple[list[Row], int, int]:
    """Every results row from every local results file, de-duplicated.

    Two kinds of duplication have to be collapsed, and they are different:

    1. The same file lists the same measurement once per *stack label* it has
       ever carried.  The pod's container id is part of the stack string and
       changes on every restart, so one gate run shows up under three ids with
       KL agreeing to all 16 digits.  Bit-identical rows are copies, not
       replications, and counting them three times would inflate every
       correlation below.  They are merged and every stack label is recorded.
    2. The same measurement appears in two files (laptop results and the pod
       snapshot).  Same treatment.

    Returns:
        (rows, n_copies_merged, n_files).
    """
    merged: dict[tuple, Row] = {}
    copies = 0
    files = 0
    for pattern in RESULT_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            files += 1
            payload = json.loads(path.read_text(encoding="utf-8"))
            model = payload.get("model_slug") or path.parent.name
            rel_path = str(path.relative_to(ROOT)).replace("\\", "/")
            for r in payload["variants"]:
                if r.get("mean_kl") is None:
                    continue
                base = r.get("base_name", r["name"])
                cfg = r.get("config", "-")
                # KL to full precision is the identity of a measurement: two
                # rows agreeing to 16 digits ran the same arithmetic on the
                # same bits.  Anything genuinely re-measured differs by ~1e-3.
                key = (model, cfg, base, int(r.get("n_seeds", 0) or 0),
                       repr(float(r["mean_kl"])), r["name"])
                hit = merged.get(key)
                if hit is not None:
                    copies += 1
                    if r.get("stack") and r["stack"] not in hit.stacks:
                        hit.stacks.append(r["stack"])
                    if rel_path not in hit.sources:
                        hit.sources.append(rel_path)
                    continue
                merged[key] = Row(
                    model=model,
                    stacks=[r.get("stack", "?")],
                    config=cfg,
                    variant=base,
                    kind=r.get("kind", "?"),
                    n_seeds=int(r.get("n_seeds", 0) or 0),
                    bpw=float(r["bpw_total"]),
                    kl=float(r["mean_kl"]),
                    rel=r.get("mean_rel_err"),
                    wrel=r.get("mean_weighted_rel_err"),
                    wrel_source=("results"
                                 if r.get("mean_weighted_rel_err") is not None
                                 else "-"),
                    fit_weighting=r.get("fit_weighting") or "none",
                    coeff_rounding=r.get("coeff_rounding") or "nearest",
                    incoherence=r.get("incoherence") or "none",
                    row_name=r["name"],
                    eff_bytes=r.get("whole_model_effective_bytes"),
                    sources=[rel_path],
                )
    rows = sorted(merged.values(), key=lambda r: (r.model, r.config, r.bpw))
    flag_partial_fits(rows)
    return rows, copies, files


def flag_partial_fits(rows: list[Row]) -> None:
    """Drop rows whose fit did not cover the whole target tensor set.

    A ``--layers-limit N`` run writes a results row that is indistinguishable
    from a full-model row in every field a reader normally looks at — same
    `bpw_total`, same-shaped KL — while most of the model was still bf16 when
    that KL was measured.  Two such rows exist in the 0.6B history and one of
    them would otherwise be the headline finding of this document.

    They are detectable from the row's own arithmetic, with no cache and no
    model access.  ``whole_model_effective_bytes`` is

        (target_params * bpw_total + (total_params - target_params) * 16) / 8

    and the ``bf16`` control row of the same model pins ``total_params``, so
    ``target_params`` can be solved for row by row.  A row that compressed two
    layers out of 28 shows an implied target an order of magnitude below its
    siblings'.
    """
    by_model: dict[str, list[Row]] = {}
    for r in rows:
        by_model.setdefault(r.model, []).append(r)
    for model, mrows in by_model.items():
        ctrl = next((r for r in mrows if r.variant == "bf16" and r.eff_bytes), None)
        if ctrl is None:
            continue
        total_params = ctrl.eff_bytes * 8.0 / 16.0
        for r in mrows:
            if r.eff_bytes is None or abs(r.bpw - 16.0) < 1e-9:
                continue
            r.fitted_tensors = int(round(
                (r.eff_bytes * 8.0 - total_params * 16.0) / (r.bpw - 16.0)))
        fitted = [r.fitted_tensors for r in mrows
                  if r.fitted_tensors and r.kind == "seed"]
        if not fitted:
            continue
        full = max(fitted)
        for r in mrows:
            expect = f"{r.config}_{r.variant}" if r.kind == "seed" else r.variant
            reasons = []
            if r.kind == "seed" and r.fitted_tensors and r.fitted_tensors < 0.95 * full:
                reasons.append(
                    f"partial fit — its own effective-bytes field implies only "
                    f"{r.fitted_tensors / full:.0%} of the model's compressible "
                    f"parameters were compressed (a `--layers-limit` run), so "
                    f"most of the model was still bf16 when this KL was "
                    f"measured and its rel_err is over a different tensor set")
            if r.row_name != expect:
                reasons.append(f"non-standard row label `{r.row_name}`, written "
                               f"by an ad-hoc run rather than the variant matrix")
            r.excluded = "; ".join(reasons)


# ----------------------------------------------------------- cache metadata
def _cp_of(config: str) -> tuple[int, int]:
    return (12, 4) if config.startswith("c12p4") else (8, 3)


def _group_key(variant: str, C: int, P: int, n_seeds: int,
               fw: str, cr: str, inc: str) -> tuple:
    return (variant, C, P, n_seeds, fw, cr, inc)


@dataclass
class CacheIndex:
    """Everything the local fit cache knows, grouped by fit identity."""

    groups: dict[tuple, dict[str, dict]] = field(default_factory=dict)
    files: dict[tuple, Path] = field(default_factory=dict)   # +tensor


def scan_cache(model: str) -> CacheIndex:
    """Index every cached fit's metadata by (variant, C, P, seeds, knobs).

    This is the only way to learn what a results row's fit actually *covered*.
    A run launched with ``--layers-limit 2`` writes 14 cache entries and a
    results row that looks like a full-model row apart from its effective-bytes
    field; two such rows exist in the 0.6B history.
    """
    idx = CacheIndex()
    cdir = ROOT / "cache" / model
    if not cdir.is_dir():
        return idx
    for f in sorted(cdir.glob("*.safetensors")):
        try:
            with safe_open(str(f), framework="pt") as fh:
                m = json.loads(fh.metadata()["meta"])
        except Exception:
            continue
        if not m.get("tensor") or not m.get("variant"):
            continue
        key = _group_key(m["variant"], m.get("C"), m.get("P"), m.get("n_seeds"),
                         m.get("fit_weighting") or "none",
                         m.get("coeff_rounding") or "nearest",
                         m.get("incoherence") or "none")
        idx.groups.setdefault(key, {})[m["tensor"]] = m
        idx.files[key + (m["tensor"],)] = f
    return idx


def n_target_tensors(model: str) -> int | None:
    """Target-tensor count for a model, read from its own config.json."""
    cfg = ROOT / "models" / "originals" / model / "config.json"
    if not cfg.exists():
        return None
    n_layers = int(json.loads(cfg.read_text(encoding="utf-8"))["num_hidden_layers"])
    return len(swap_eval.target_tensor_names(n_layers))


def group_keys_for(row: Row) -> list[tuple]:
    """Cache-group keys a row's fit could be stored under.

    Rows written before the W2/W3 follow-up carry ``None`` for the rounding/incoherence
    fields and their cache metadata predates those fields too, so the pre-3.6
    spelling is tried as well.
    """
    C, P = _cp_of(row.config)
    keys = [_group_key(row.variant, C, P, row.n_seeds, row.fit_weighting,
                       row.coeff_rounding, row.incoherence)]
    fallback = _group_key(row.variant, C, P, row.n_seeds, "none", "nearest", "none")
    if fallback not in keys:
        keys.append(fallback)
    return keys


def attach_coverage(rows: list[Row], caches: dict[str, CacheIndex]) -> None:
    """Tag every seed row with how much of its fit is in the *local* cache.

    Informational only, and deliberately not an exclusion criterion: a row
    measured on the pod is not partial just because this laptop happens to hold
    a fraction of its fit.  Coverage is what makes a back-filled proxy quotable
    (§0 of the report), nothing more.  Whether a *fit* was partial is decided by
    :func:`flag_partial_fits`, from the row's own bytes field.
    """
    for r in rows:
        if r.kind != "seed":
            continue
        total = n_target_tensors(r.model)
        grp = None
        for key in group_keys_for(r):
            grp = caches.get(r.model, CacheIndex()).groups.get(key)
            if grp:
                break
        r.coverage = (f"{len(grp)}/{total}" if grp and total
                      else "not cached locally")


# ------------------------------------------------------- weighted back-fill
def backfill_weighted(rows: list[Row], caches: dict[str, CacheIndex],
                      log=print) -> list[str]:
    """Recompute the weighted proxy for rows fitted before the field existed.

    ``weighted_rel_err`` entered the cache metadata with W1.  Every *unweighted*
    Qwen3-1.7B fit predates it — which is exactly the set of rows the W1 step
    has to be measured against.  The reconstructions are still on disk, so

        wrel = || (W - What) diag(s) ||_F / || W diag(s) ||_F

    is recoverable with the harness's own ``runner._weighted_rel_err`` against
    the same cached ``act_scales``.  Fits are deterministic from
    ``generator_seed``, so a locally cached reconstruction is the same object
    the pod fitted under the same cache key.

    Where only part of a fit is cached locally the mean is taken over that
    subset, labelled ``cache n/m``, and the subset's own plain ``rel_err`` is
    reported next to the full row's so representativeness is checkable.
    """
    notes: list[str] = []
    todo = [r for r in rows if r.kind == "seed" and not r.excluded and r.wrel is None]
    by_model: dict[str, list[Row]] = {}
    for r in todo:
        by_model.setdefault(r.model, []).append(r)

    for model, mrows in sorted(by_model.items()):
        scales_path = ROOT / "cache" / model / "act_scales.safetensors"
        mdir = ROOT / "models" / "originals" / model
        idx = caches.get(model, CacheIndex())
        # group key -> (rows sharing that fit, {tensor: meta})
        groups: dict[tuple, tuple[list[Row], dict[str, dict]]] = {}
        for r in mrows:
            for key in group_keys_for(r):
                tensors = idx.groups.get(key)
                if tensors:
                    groups.setdefault(key, ([], tensors))[0].append(r)
                    break
        if not groups:
            continue
        if not scales_path.exists() or not mdir.is_dir():
            notes.append(f"`{model}`: no local `act_scales` or originals — "
                         f"{len(mrows)} row(s) keep an empty weighted axis")
            continue
        wanted = sorted({t for _rs, tensors in groups.values() for t in tensors})
        log(f"BACKFILL {model}: {len(groups)} fit(s), {len(wanted)} tensors")
        with safe_open(str(scales_path), framework="pt") as fh:
            scales = {k: fh.get_tensor(k) for k in fh.keys()}
        acc: dict[tuple, list[tuple[float, float]]] = {k: [] for k in groups}
        t0 = time.time()
        with runner.ShardReader(mdir) as shard:
            for i, tname in enumerate(wanted):
                s = scales.get(tname)
                if s is None:
                    continue
                w = shard.get_tensor(tname).float()
                for key, (_rs, tensors) in groups.items():
                    meta = tensors.get(tname)
                    if meta is None:
                        continue
                    path = idx.files.get(key + (tname,))
                    if path is None:
                        continue
                    with safe_open(str(path), framework="pt") as fh:
                        deq = fh.get_tensor("w")
                    acc[key].append((runner._weighted_rel_err(w, deq.float(), s),
                                     float(meta.get("rel_err") or 0.0)))
                    del deq
                del w
                if (i + 1) % 50 == 0:
                    log(f"BACKFILL   {i + 1}/{len(wanted)} "
                        f"({time.time() - t0:.0f}s)")
        total = n_target_tensors(model)
        for key, (rs, _t) in groups.items():
            vals = acc[key]
            if not vals:
                continue
            wrel = _mean(v for v, _ in vals)
            sub_rel = _mean(e for _, e in vals)
            for r in rs:
                r.wrel = wrel
                r.wrel_source = f"cache {len(vals)}/{total}"
            r0 = rs[0]
            notes.append(
                f"`{model} {r0.label}` (n_seeds={r0.n_seeds}): wrel = "
                f"**{wrel:.5f}** from {len(vals)}/{total} cached tensors; that "
                f"subset's plain rel_err is {sub_rel:.4f} against the full "
                f"row's {fmt(r0.rel)}")
        log(f"BACKFILL {model} done in {time.time() - t0:.0f}s")
    return notes


# ----------------------------------------------------------------- reports
def corr_rows(pairs: list[tuple[str, list[Row], str]]) -> list[str]:
    """Correlation table body over (label, rows, proxy-attribute) triples."""
    out = []
    for label, rs, proxy in pairs:
        keep = [(getattr(r, proxy), r.kl) for r in rs
                if getattr(r, proxy) is not None]
        if len(keep) < 3:
            out.append(f"| {label} | `{PROXY_LABEL[proxy]}` | {len(keep)} | — "
                       f"| — | n < 3 |")
            continue
        x = [a for a, _ in keep]
        y = [b for _, b in keep]
        out.append(f"| {label} | `{PROXY_LABEL[proxy]}` | {len(keep)} | "
                   f"{fmt(spearman(x, y), 3)} | {fmt(pearson(x, y), 3)} | "
                   f"{fmt(perm_p(x, y), 3)} |")
    return out


CORR_HEAD = ["| split | proxy | n | Spearman rho | Pearson r | exact p |",
             "|---|---|---|---|---|---|"]


def family_elasticity(rows: list[Row], family: str, proxy: str) -> float | None:
    """``d log KL / d log proxy`` across one family's own outlier-budget sweep.

    The reference exchange rate: what a unit of proxy movement is worth in the
    regime where the proxy demonstrably works (§1b).  Taken between the family's
    cheapest and most expensive row.
    """
    rs = sorted([r for r in rows if r.family == family
                 and getattr(r, proxy) is not None], key=lambda r: r.bpw)
    if len(rs) < 2:
        return None
    return elasticity(getattr(rs[0], proxy), getattr(rs[-1], proxy),
                      rs[0].kl, rs[-1].kl)


def _interp(xs: list[float], ys: list[float], x: float) -> float | None:
    """Piecewise-linear interpolation on an ascending-x sample, None outside."""
    if not xs or x < xs[0] or x > xs[-1]:
        return None
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            if xs[i + 1] == xs[i]:
                return ys[i]
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return None


def render(rows: list[Row], copies: int, files: int, notes: list[str]) -> str:
    live = [r for r in rows if not r.excluded]
    seed = [r for r in live if r.kind == "seed"]
    dropped = [r for r in rows if r.excluded]
    pod17 = [r for r in seed if r.model == "Qwen3-1.7B" and not r.host.startswith("test-laptop")]
    lap17 = [r for r in seed if r.model == "Qwen3-1.7B" and r.host.startswith("test-laptop")]
    lap06 = [r for r in seed if r.model == "Qwen3-0.6B"]

    L: list[str] = []
    A = L.append
    A("# Proxy alignment — which fit-time number actually predicts KL")
    A("")
    A("_Generated by `proxy_alignment.py`: CPU-only mining of results and "
      "cache metadata that already existed. No fit, no model load, no GPU._")
    A("")
    A("**Why.** The W2 step lowered the activation-weighted L2 objective "
      "by 1.7% and KL got 11% worse. That is a Goodhart signature, and it "
      "makes every future \"better objective\" proposal a gamble until we know "
      "what the proxies we already have do and do not predict. This is that "
      "measurement, over every row the project has.")
    A("")
    A("**Answer in one paragraph.** Inside a fixed fit mechanism both proxies "
      "rank behavioural damage perfectly (Spearman +1.000 in every family "
      "measured). Across fit mechanisms at identical bits they part company: "
      "plain `rel_err` has no resolution left — the whole 4.17x W1 effect fits "
      "inside a 0.52% move of it, an elasticity of 273 against ~24 in its own "
      "family, and against the matched-seed-budget control (KL 1.121121) it "
      "moves the *wrong way*, 15.6% worse for a 2.57x KL improvement — "
      "while the activation-weighted proxy predicted W1 *on trend* "
      "(elasticity 3.2, inside the 1.6-6.7 range of its own budget sweeps). "
      "One mechanism change later it stopped: W2 lowered the same weighted "
      "objective on all four rows and raised KL on three, and the whole "
      "`c12p4wr` family sits 7-13% above the `c12p4w` family's proxy→KL curve "
      "at equal weighted error. So W2 is **not** an outlier — the curve is "
      "mechanism-dependent, which means the weighted objective is no longer a "
      "function of damage near its own optimum. The cheapest next measurement "
      "is therefore not a better objective: it is finding out whether the "
      "objective's weights are simply under-estimated from 12 calibration "
      "prompts (the recalibration probe).")
    A("")

    # ---------------------------------------------------------- 0. the data
    A("## 0. The row set, and what was thrown out")
    A("")
    A(f"- {len(rows)} distinct measurements over {files} results files, "
      f"{len({r.model for r in rows})} models and "
      f"{len({s for r in rows for s in r.stacks})} stack labels.")
    A(f"- {copies} further row records were **copies**, not replications: the "
      f"pod's container id is part of its stack string and changes on every "
      f"restart, so one gate run is re-listed under each id with KL agreeing "
      f"to all 16 digits. Counting those again would have inflated every "
      f"correlation here. They are merged; all stack labels are kept.")
    A(f"- {len(seed)} rows are seed-method rows — the only kind with a "
      f"fit-time proxy. `q3km`/`q4km` record `rel_err = 0` (nothing in this "
      f"harness fits them) and `rtn4` is closed-form with no search; they are "
      f"behavioural reference points, never proxy samples.")
    A(f"- {len(dropped)} row(s) excluded:")
    for r in dropped:
        A(f"  - `{r.model}` `{r.row_name}` — {r.excluded}")
    A("")
    if any("partial fit" in r.excluded for r in dropped):
        A("**The partial-fit trap.** Those dropped rows carry a full-model "
          "`bpw_total` and a full-model-shaped KL, and nothing in the results "
          "table marks them: they are `--layers-limit 2` smoke rows whose other "
          "26 layers were still bf16 when their KL was measured. Taken at face "
          "value they say that *a worse fit* — rel_err 0.327 against 0.227 at "
          "the same 3.0 bpw — *did 1.9x less damage*, which is a spectacular "
          "fake Goodhart and would have been the headline of this document. "
          "They are detectable from their own arithmetic: "
          "`whole_model_effective_bytes` plus the `bf16` control pins how many "
          "parameters were actually compressed (see `flag_partial_fits`), and "
          "these rows imply 7% of the model. **Any future analysis of this "
          "results file needs the same guard.**")
        A("")
    A("| model | host | config | variant | bpw | seeds | mean KL | plain "
      "rel_err | act-wtd rel_err | wtd source | in local cache |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(live, key=lambda r: (r.model, r.config, r.bpw, r.variant)):
        A(f"| {r.model} | `{r.host}` | `{r.config}` | {r.variant} | "
          f"{r.bpw:.4f} | {r.n_seeds or '—'} | {r.kl:.6f} | {fmt(r.rel)} | "
          f"{fmt(r.wrel)} | {r.wrel_source} | {r.coverage} |")
    A("")
    if notes:
        A("**Back-filled weighted proxies.** `weighted_rel_err` only began "
          "being recorded when W1 landed, so every unweighted 1.7B row — "
          "exactly the rows the W1 step must be compared against — had a hole "
          "on the weighted axis. The reconstructions are still cached and fits "
          "are deterministic from `generator_seed`, so the number was "
          "recomputed here with the harness's own `runner._weighted_rel_err` "
          "against the same cached `act_scales`:")
        A("")
        for n in notes:
            A(f"- {n}")
        A("")
    A("**Stacks are never pooled.** The same-stack determinism rule forbids "
      "comparing numbers from different stacks, so every correlation below is "
      "computed inside one stack and cross-stack splits are reported "
      "separately.")
    A("")

    # -------------------------------------------- 1. proxy vs KL, by split
    A("## 1. Does the proxy rank the damage?")
    A("")
    A("Three nested questions. The answer changes at every level.")
    A("")
    A("### 1a. Pooled over everything (the misleading view)")
    A("")
    L += CORR_HEAD
    L += corr_rows([("Qwen3-1.7B, pod", pod17, "rel"),
                    ("Qwen3-1.7B, pod", pod17, "wrel"),
                    ("Qwen3-1.7B, laptop", lap17, "rel"),
                    ("Qwen3-1.7B, laptop", lap17, "wrel"),
                    ("Qwen3-0.6B, laptop", lap06, "rel"),
                    ("Qwen3-0.6B, laptop", lap06, "wrel")])
    A("")
    A("Pooled correlations look healthy and mean almost nothing: pooling mixes "
      "bit budgets, so it mostly measures that `c12p4` (3.0 bpw) both fits "
      "worse and damages more than `c8p3` (4.0 bpw). *Any* quantity monotone "
      "in the bit budget scores well here. The informative splits follow.")
    A("")

    A("### 1b. Within one fit mechanism — vary only the outlier budget")
    A("")
    fams = sorted({r.family for r in seed},
                  key=lambda f: ([r.model for r in seed if r.family == f][0], f))
    L += CORR_HEAD
    L += corr_rows([(f"`{f}` ({[r.model for r in seed if r.family == f][0]})",
                     [r for r in seed if r.family == f], proxy)
                    for f in fams for proxy in ("rel", "wrel")])
    A("")
    perfect = sum(1 for f in fams
                  for proxy in ("rel", "wrel")
                  if (lambda rs: len(rs) >= 3 and spearman(
                      [getattr(r, proxy) for r in rs], [r.kl for r in rs]) == 1.0)(
                      [r for r in seed if r.family == f
                       and getattr(r, proxy) is not None]))
    A(f"This is the regime the proxy was built for, and it works: inside a "
      f"fixed (config, fit objective, seed budget) family, spending bits on "
      f"the outlier side channel lowers both proxies and lowers KL, "
      f"monotonically, in every family measured ({perfect} of "
      f"{2 * len(fams)} (family, proxy) combinations reach a perfect +1.000; "
      f"the rest are families where that proxy was never recorded). Note the "
      f"p-value floor: a family has 4 rows, so perfect rank agreement is "
      f"p = 0.083 two-sided. Six families all achieving it is the evidence, "
      f"not any single one of them.")
    A("")

    # 1c matched bits
    A("### 1c. Across fit mechanisms at identical bits — the diagnostic split")
    A("")
    A("Every `c12p4*` row at a given outlier budget stores exactly the same "
      "number of bits: W1 and W2 change *which* coefficients are chosen, never "
      "how many bits they cost, and the harness asserts that in its accounting "
      "selftests. So each bpw group below is a controlled experiment with one "
      "variable — how the fit was obtained.")
    A("")
    A("| bpw | rows, best KL first | plain rel_err spread | act-wtd spread | "
      "KL spread | plain ranks KL? | act-wtd ranks KL? |")
    A("|---|---|---|---|---|---|---|")
    groups_rel: list[tuple[list[float], list[float]]] = []
    groups_wrel: list[tuple[list[float], list[float]]] = []
    for bpw in sorted({round(r.bpw, 4) for r in pod17 if r.config.startswith("c12p4")}):
        grp = [r for r in pod17
               if round(r.bpw, 4) == bpw and r.config.startswith("c12p4")]
        if len(grp) < 3:
            continue
        rels = [r.rel for r in grp if r.rel is not None]
        wrels = [r.wrel for r in grp if r.wrel is not None]
        kls = [r.kl for r in grp]
        if len(rels) == len(grp):
            groups_rel.append(([r.rel for r in grp], kls))
        if len(wrels) == len(grp):
            groups_wrel.append(([r.wrel for r in grp], kls))
        rho_r = spearman([r.rel for r in grp], kls) if len(rels) == len(grp) else None
        rho_w = spearman([r.wrel for r in grp], kls) if len(wrels) == len(grp) else None
        desc = " · ".join(f"`{r.config}` {r.kl:.3f}"
                          for r in sorted(grp, key=lambda r: r.kl))
        A(f"| {bpw:.4f} | {desc} | {max(rels) / min(rels) - 1:+.2%} | "
          f"{(max(wrels) / min(wrels) - 1):+.2%} | {max(kls) / min(kls):.1f}x | "
          f"rho {fmt(rho_r, 2)} | rho {fmt(rho_w, 2)} |")
    A("")
    for proxy, groups in (("rel", groups_rel), ("wrel", groups_wrel)):
        obs, p = stratified_sign_p(groups)
        A(f"- **{PROXY_LABEL[proxy]}**: mean within-group Spearman "
          f"{fmt(obs, 3)} over {len(groups)} matched-bits groups of 3, exact "
          f"stratified permutation p = {fmt(p, 3)}.")
    A("")
    A("**The two proxies fail this split in completely different ways, and the "
      "rank statistic cannot tell them apart.** Both land on the same mean "
      "within-group rho, both for the same reason: each gets the big call "
      "right (the unweighted fit is worst) and each inverts the `c12p4w` vs "
      "`c12p4wr` pair in three groups of four. What separates them is dynamic "
      "range. Plain `rel_err` spans about 1% across fits whose KL spans up to "
      "14.6x — there is no resolution there to be right or wrong with, and its "
      "one correct call is worth little because the row it calls out is also "
      "the row confounded by seed budget (§5). The weighted proxy spans "
      "55-84% across the same groups: it genuinely resolves them, and then "
      "orders the top two backwards. A blind proxy and a Goodharted proxy "
      "score the same on rank agreement; only the spread column distinguishes "
      "them.")
    A("")

    # ------------------------------------------- 2. the W1 discontinuity
    A("## 2. The W1 discontinuity, quantified")
    A("")
    A("W1 (the activation-weighted fit objective) is the largest behavioural "
      "move this project has measured at fixed bits. Both proxies were present "
      "for it. Only one of them saw it.")
    A("")
    A("| Qwen3-1.7B, 3.48 bpw, byte-identical stored format | plain rel_err | "
      "act-wtd rel_err | mean KL |")
    A("|---|---|---|---|")
    a = next((r for r in pod17 if r.config == "c12p4" and r.variant == "awq_p1.0"), None)
    b = next((r for r in pod17 if r.config == "c12p4w" and r.variant == "awq_p1.0"), None)
    e_rel = e_wrel = None
    if a and b:
        A(f"| `{a.label}` — plain L2, {a.n_seeds:,} seeds | {fmt(a.rel)} | "
          f"{fmt(a.wrel)} | {a.kl:.6f} |")
        A(f"| `{b.label}` — W1, {b.n_seeds:,} seeds | {fmt(b.rel)} | "
          f"{fmt(b.wrel)} | {b.kl:.6f} |")
        A(f"| **change** | {pct(b.rel / a.rel - 1, 2)} | "
          f"{pct(b.wrel / a.wrel - 1, 2) if a.wrel and b.wrel else '—'} | "
          f"**{pct(b.kl / a.kl - 1, 1)}** ({a.kl / b.kl:.2f}x better) |")
        e_rel = elasticity(a.rel, b.rel, a.kl, b.kl)
        e_wrel = elasticity(a.wrel, b.wrel, a.kl, b.kl)
    A("")
    A("Elasticity `d log KL / d log proxy` — how much KL movement each unit of "
      "proxy movement came with — against the same elasticity inside families "
      "where the proxy is known to work:")
    A("")
    A("| step | proxy | elasticity | reading |")
    A("|---|---|---|---|")
    if e_rel is not None:
        A(f"| W1 (`c12p4`→`c12p4w`) | plain rel_err | **{e_rel:.1f}** | the "
          f"proxy moved {abs(100 / e_rel):.2f}% for every 1% of KL |")
    if e_wrel is not None:
        A(f"| W1 (`c12p4`→`c12p4w`) | act-wtd rel_err | **{e_wrel:.1f}** | the "
          f"proxy carried the move |")
    for fam in sorted({r.family for r in pod17}):
        rs = sorted([r for r in pod17 if r.family == fam], key=lambda r: r.bpw)
        if len(rs) < 2:
            continue
        lo, hi = rs[0], rs[-1]
        for proxy in ("rel", "wrel"):
            e = elasticity(getattr(lo, proxy), getattr(hi, proxy), lo.kl, hi.kl)
            if e is not None:
                A(f"| budget sweep in `{fam}` | {PROXY_LABEL[proxy]} | {e:.1f} "
                  f"| reference: the regime where the proxy works |")
    A("")
    if e_rel is not None:
        fam_e = family_elasticity(pod17, a.family, "rel")
        pred = a.kl * math.exp(fam_e * math.log(b.rel / a.rel)) if fam_e else None
        A(f"**Reading.** Plain `rel_err` at the W1 step has an elasticity of "
          f"{e_rel:.0f}, against {fam_e:.1f} inside its own family. Take that "
          f"family elasticity as the proxy's honest exchange rate and it "
          f"predicts that a {pct(b.rel / a.rel - 1, 2)} move in plain rel_err "
          f"buys KL {a.kl:.3f} → {pred:.3f}. The measured value is "
          f"{b.kl:.3f}. The proxy is not merely imprecise here — it is off by "
          f"a factor of {pred / b.kl:.1f} on a quantity it is supposed to "
          f"rank, because it was not measuring the thing that moved.")
        A("")
        A(f"The weighted proxy, by contrast, moved by a large and *ordinary* "
          f"amount: {pct(b.wrel / a.wrel - 1, 1)}, at an elasticity of "
          f"{fmt(e_wrel, 1)} — squarely inside the 1.6-6.7 range its own "
          f"within-family budget sweeps show. **W1 did not surprise the "
          f"weighted proxy at all. It moved along the proxy's existing "
          f"curve.** That is the entire empirical case for the weighted "
          f"objective, it is a good one, and §3 is about the point where it "
          f"stops.")
        A("")

    # ---------------------------------------------- 3. the W2 Goodhart row
    A("## 3. The W2 row: an outlier, or does the trend itself bend?")
    A("")
    A("W2 (exact coefficient search) is a strictly tighter minimisation of the "
      "*same* objective W1 introduced: its candidate set contains the "
      "nearest-rounding point, so it cannot increase the objective on any "
      "block, and the harness selftests assert exactly that. It lowered the "
      "objective on every row.")
    A("")
    A("| outlier budget | wrel `c12p4w`→`c12p4wr` | Δ objective | KL "
      "`c12p4w`→`c12p4wr` | Δ KL | elasticity |")
    A("|---|---|---|---|---|---|")
    pairs = []
    for var in ("seed_only", "awq_p0.5", "awq_p1.0", "awq_p2.0"):
        x = next((r for r in pod17 if r.config == "c12p4w" and r.variant == var), None)
        y = next((r for r in pod17 if r.config == "c12p4wr" and r.variant == var), None)
        if not (x and y and x.wrel and y.wrel):
            continue
        pairs.append((x, y))
        A(f"| {var} | {x.wrel:.5f} → {y.wrel:.5f} | {pct(y.wrel / x.wrel - 1, 2)} "
          f"| {x.kl:.6f} → {y.kl:.6f} | **{pct(y.kl / x.kl - 1, 2)}** | "
          f"{fmt(elasticity(x.wrel, y.wrel, x.kl, y.kl), 1)} |")
    A("")
    if pairs:
        worse = sum(1 for x, y in pairs if y.kl > x.kl)
        A(f"Objective lower in {len(pairs)}/{len(pairs)} pairs; KL higher in "
          f"{worse}/{len(pairs)}. As a sign test that is "
          f"p = {binom_two_sided(worse, len(pairs)):.3f} two-sided — "
          f"suggestive, not decisive, on four paired rows. The interesting "
          f"structure is not the sign count, it is the geometry:")
        A("")
    A("**Is `c12p4wr` an outlier against the weighted-proxy → KL trend, or is "
      "the trend itself mechanism-dependent?** Take the `c12p4w` family as the "
      "reference curve (four rows, monotone in both axes), interpolate it "
      "piecewise-linearly, and ask where each `c12p4wr` row lands on it at "
      "*its own* weighted error:")
    A("")
    A("| `c12p4wr` row | its wrel | KL the `c12p4w` curve predicts | KL "
      "measured | residual |")
    A("|---|---|---|---|---|")
    curve = sorted([r for r in pod17 if r.config == "c12p4w" and r.wrel],
                   key=lambda r: r.wrel)
    resids = []
    for y in sorted([r for r in pod17 if r.config == "c12p4wr" and r.wrel],
                    key=lambda r: r.wrel):
        pred = _interp([r.wrel for r in curve], [r.kl for r in curve], y.wrel)
        if pred is None:
            A(f"| `{y.variant}` | {y.wrel:.5f} | outside the curve's range | "
              f"{y.kl:.6f} | — |")
            continue
        resids.append(y.kl - pred)
        A(f"| `{y.variant}` | {y.wrel:.5f} | {pred:.6f} | {y.kl:.6f} | "
          f"**{y.kl - pred:+.6f}** ({pct(y.kl / pred - 1, 1)}) |")
    A("")
    wr = [r for r in pod17 if r.config == "c12p4wr" and r.wrel]
    if resids:
        A(f"All {len(resids)} in-range rows sit **above** the reference curve, "
          f"by {min(resids):+.4f} to {max(resids):+.4f} KL (mean "
          f"{_mean(resids):+.4f}, i.e. {_mean(resids) / _mean(r.kl for r in wr) * 100:+.1f}% "
          f"of their own KL). A single outlier would be one row off a curve; "
          f"this is an entire family displaced in one direction. Inside "
          f"itself the `c12p4wr` family is perfectly well behaved — "
          f"Spearman(wrel, KL) = "
          f"{fmt(spearman([r.wrel for r in wr], [r.kl for r in wr]), 3)} — it "
          f"simply lives on a different curve.")
        A("")
    # The row below the reference curve's range needs no interpolation at all.
    if curve and wr:
        best_obj = min(wr, key=lambda r: r.wrel)
        if best_obj.wrel < min(r.wrel for r in curve):
            beaten = sum(1 for r in curve if r.kl < best_obj.kl)
            A(f"The fourth row needs no interpolation to make the same point. "
              f"`c12p4wr_{best_obj.variant}` has a **lower weighted error than "
              f"every `c12p4w` row in existence** ({best_obj.wrel:.5f} against "
              f"a best of {min(r.wrel for r in curve):.5f}) and lands at a KL "
              f"that {beaten} of the {len(curve)} `c12p4w` rows beat "
              f"({best_obj.kl:.6f}). If the objective ranked damage, that row "
              f"would be the best row in the table.")
            A("")
    both = [r for r in pod17 if r.config in ("c12p4w", "c12p4wr") and r.wrel]
    A(f"Pooling the two weighted families gives Spearman(wrel, KL) = "
      f"{fmt(spearman([r.wrel for r in both], [r.kl for r in both]), 3)} "
      f"(exact p = {fmt(perm_p([r.wrel for r in both], [r.kl for r in both]), 3)}, "
      f"n = {len(both)}) against +1.000 inside each family separately. The "
      f"degradation is the family offset, not noise.")
    A("")
    w2_e = None
    for x, y in pairs:
        if x.variant == "awq_p1.0":
            w2_e = elasticity(x.wrel, y.wrel, x.kl, y.kl)
    A(f"**The contrast with W1 is the whole finding.** W1 and W2 are both "
      f"changes to the fit mechanism at byte-identical bits, and both lowered "
      f"the weighted objective. W1 moved *along* the weighted proxy's own "
      f"curve (elasticity {fmt(e_wrel, 1)}, inside the 1.6-6.7 range of the "
      f"within-family budget sweeps). W2 moved *off* it (elasticity "
      f"{fmt(w2_e, 1)} on the same row — the wrong sign entirely). Same "
      f"objective, same units, same model, same bits: one mechanism change was "
      f"predicted by the proxy and the very next one was not. That is what an "
      f"exhausted proxy looks like, and it is why \"minimise the weighted "
      f"objective harder\" is not a plan.")
    A("")
    A("**Conclusion.** `c12p4wr` is *not* an outlier. The "
      "weighted-rel_err → KL relation is not a function of weighted rel_err "
      "alone; it also depends on *how* the error was obtained. Two fits with "
      "the same weighted error — one reached by per-coefficient nearest "
      "rounding, one by exact search over the same grid — do measurably "
      "different amounts of behavioural damage, and the tighter one does more. "
      "The proxy has stopped being a proxy and become a coordinate.")
    A("")

    # ------------------------------------------------------ 4. the verdict
    A("## 4. Verdict — which proxy where, and what a better objective must do")
    A("")
    A("| regime | trust | evidence |")
    A("|---|---|---|")
    A("| Choosing an outlier budget inside one fit mechanism | **either "
      "proxy** | §1b: Spearman +1.000 in every family measured |")
    A("| Comparing bit budgets (`c8p3` vs `c12p4`) | **plain rel_err** | §1a |")
    A("| Comparing fit *mechanisms* at identical bits | **neither — weighted "
      "is merely the only one with resolution** | §1c, §2 |")
    A("| Tightening the weighted objective further | **nothing we have** | "
      "§3: a strictly better objective value, worse KL, whole family "
      "displaced |")
    A("")
    A("Stated so the next measurement can falsify them:")
    A("")
    A("- **C1.** *Within a fixed fit mechanism, both proxies rank KL "
      "correctly.* Falsified by any family whose budget sweep has "
      "Spearman < 1. Six families, two models, no counterexample yet.")
    A("- **C2.** *Plain per-tensor `rel_err` cannot compare fit mechanisms.* "
      "The whole W1 effect fits inside a 0.52% move of it (elasticity 273 "
      "against 23.5 in its own family), and across the matched-bits groups it "
      "spans about 1% while KL spans 14.6x — there is no resolution there to "
      "rank anything with, whatever its rank statistic happens to come out at "
      "(§1c). Any objective proposal justified by plain rel_err alone should "
      "be rejected on this evidence.")
    A("- **C3.** *Lowering the current activation-weighted L2 further does not "
      "lower KL.* Measured once, at the frontier, by a mechanism (W2) that "
      "provably lowers the objective and changes nothing else. This is the "
      "claim the recalibration probe can still overturn cheaply: if `s` is "
      "badly estimated from 12 prompts, W2 was minimising a noisy objective "
      "more exactly, and a better-estimated `s` should restore alignment.")
    A("- **C4.** *A better objective must be mechanism-invariant.* The "
      "property the current one lacks is stated precisely in §3: fits reached "
      "by different mechanisms at equal objective value must land at equal KL. "
      "This is directly testable and needs no new objective — nearest vs exact "
      "rounding is already such a pair, and the current objective fails the "
      "test by 7-13% of KL. **Any candidate objective should be run through "
      "this test before it is run through a gate.**")
    A("- **C5.** *The objective's weights are a measurement whose error has "
      "never been quantified.* Both `s` (the fit weighting) and the awq "
      "outlier ranking are mean |activation| over **12 prompts**. No "
      "experiment in this project has ever varied that number. Until it is "
      "varied, every statement about \"the weighted objective\" is a statement "
      "about the weighted objective *as estimated from 12 prompts*.")
    A("")
    A("**What this says about Area 2.** Do not fund a new objective on this "
      "evidence. Two much cheaper measurements dominate it:")
    A("")
    A("1. **The recalibration probe**: re-estimate `s` from 256 "
      "prompts and refit W1. Tests C5 "
      "directly, and its outcome decides whether C3 is a statement about the "
      "objective or about our estimate of it.")
    A("2. **One missing row**: `c12p4r` — W2 *without* W1 — would say whether "
      "the §3 displacement is a property of exact rounding as such or only of "
      "exact rounding under the weighted metric. One pod rung, no new code "
      "(the flag combination already exists and has its own cache namespace).")
    A("")

    # ------------------------------------------------------- 5. the limits
    A("## 5. Honest limits")
    A("")
    A(f"- **n is small and structured.** {len(seed)} seed rows in "
      f"{len({r.family for r in seed})} families of four. Within a family the "
      f"four rows are not independent draws: they are one fit plus three "
      f"refits that *reuse* it, differing only in an outlier budget that is "
      f"monotone in bits. A Spearman of +1 on such a family is worth far less "
      f"than +1 on four independent samples, and its exact two-sided p cannot "
      f"go below 0.083 however clean the ordering is.")
    if a and b:
        A(f"- **The headline W1 comparison is confounded by seed budget.** "
          f"`{a.label}` was fitted at {a.n_seeds:,} candidate seeds and "
          f"`{b.label}` at {b.n_seeds:,}. More seeds means a better fit under "
          f"whichever objective is active, so those two rows differ in two "
          f"variables, not one. Direction is knowable, size is not: on "
          f"`c8p3`/1.7B the same 16,384 → 65,535 step moved KL 1.552 → 0.845 "
          f"(1.8x) under the plain objective, against 4.17x here. **The "
          f"one-variable control has since been run** — `c12p4_awq_p1.0` at "
          f"65,535 seeds, KL 1.121121 — and it decomposes the headline into "
          f"a 1.62x seed-budget factor and a **2.57x objective factor**. W1 "
          f"is real and its size is 2.6x, not 4.2x; the rows paired above "
          f"still cross two variables and are retained as the historical "
          f"comparison.")
    A("- **The weighted proxy for the unweighted 1.7B rows was computed here, "
      "not on the pod.** For `c8p3` the local cache holds the complete fit "
      "(196/196) and the number is exact; for `c12p4` it holds 80-81 of 196 "
      "tensors and the number is a subsample mean. The subsample's plain "
      "rel_err tracks the full row's to 3-4 decimal places (§0), which is the "
      "only representativeness evidence offered. A second, smaller caveat: "
      "those cached reconstructions were produced on the laptop GPU and the "
      "row's KL on a 4090. Fits are deterministic from `generator_seed` and "
      "the project relies on that to hand winners between machines, but "
      "cross-device fp agreement has only ever been checked to ~1e-7 (the W2 "
      "selftest note), not bit-exactly.")
    A("- **The three pod \"stacks\" are one machine image with three container "
      "ids.** Treated as distinct per the same-stack rule, which costs power "
      "for no physical reason; the rows they share are bit-identical, which is "
      "itself evidence the rule is being conservative here.")
    A("- **One model carries the entire objective comparison.** Every weighted "
      "row in existence is Qwen3-1.7B. The 0.6B W1 twin died with the laptop "
      "(crash #6) and was never re-run, so nothing here is known to transfer "
      "across scale — and Phase 3's central lesson was precisely that "
      "per-tensor error is scale-invariant while damage is not.")
    A("- **KL is itself a proxy**: teacher-forced, 12 prompts, 692 positions, "
      "against a bf16 reference. The calibration-set question this document "
      "raises about `s` applies to it too, and no downstream-benchmark row "
      "(`--stage lmeval`) exists for any weighted variant.")
    A("- **No causal claim is made about *why* the §3 displacement exists.** "
      "Two mechanisms fit the data equally well — (i) `s` is noisy, so exact "
      "minimisation overfits the noise; (ii) weighted L2 genuinely diverges "
      "from KL near the optimum — and nothing measured here separates them. "
      "The recalibration probe does.")
    A("")
    A(f"_Generated {time.strftime('%Y-%m-%d')} from {files} results files and "
      f"the local fit cache. Regenerate with `python proxy_alignment.py`._")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--no-backfill", action="store_true",
                    help="skip recomputing the weighted proxy from cached "
                         "reconstructions (fast, but leaves the W1 step "
                         "unmeasurable on the weighted axis)")
    args = ap.parse_args()

    rows, copies, files = load_rows()
    caches = {m: scan_cache(m) for m in sorted({r.model for r in rows})}
    attach_coverage(rows, caches)
    notes = [] if args.no_backfill else backfill_weighted(rows, caches)
    text = render(rows, copies, files, notes)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"proxy_alignment: wrote {out} ({len(text)} chars, "
          f"{len([r for r in rows if not r.excluded])} live rows, "
          f"{copies} copies merged)")


if __name__ == "__main__":
    main()
