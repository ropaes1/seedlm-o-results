"""Phase 2 salience rules and honest bits/weight accounting.

Three rules decide which part of a weight tensor is stored *exactly* in a side
channel while the rest is seed-fitted:

1. ``mag``   — top-p% entries by |w|                       (48 bits each)
2. ``awq``   — top-p% entries by |w| * act_scale[col]      (48 bits each)
3. ``spike`` — top-r SVD components as bf16 factors        ((m+n+1)*r*16 bits)

Rules 1 and 2 are *scattered* (a sparse index/value list, 16 value bits + 32
index bits per entry).  Rule 3 is *structured*: its rank r is budget-matched to
the scattered rules at the same p, so every variant at a given p pays (very
nearly) the same side-channel bpw and the comparison is apples-to-apples.

Accounting:
    bpw_total = bpw_base + side_bits / numel
with bpw_base = 4.0 for the C=8/P=3 seed config and 4.5 for the RTN comparator.
The *achieved* side bits are always reported, never the target.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

__all__ = [
    "SCATTER_BITS_PER_ENTRY",
    "SPIKE_BITS_PER_UNIT",
    "SideChannel",
    "scatter_count",
    "spike_rank_for_budget",
    "scatter_indices",
    "build_side_channel",
    "bpw_total",
    "singular_spectrum",
    "mp_bulk_edge",
    "save_spectra",
    "render_mp_atlas",
]

SCATTER_BITS_PER_ENTRY: int = 48   # 16 value (fp16) + 32 index (uint32)
SPIKE_BITS_PER_UNIT: int = 16      # bf16 factor entries


# --------------------------------------------------------------- container
@dataclass
class SideChannel:
    """Everything a variant holds outside the seed stream for one tensor.

    Attributes:
        kind: ``"none"``, ``"scatter"`` or ``"spike"``.
        side_bits: achieved side-channel cost in bits for this tensor.
        idx: flat indices of held-out entries (scatter only).
        values: exact float32 values at ``idx`` (scatter only).
        spike: dense [m, n] rank-r reconstruction from bf16-rounded factors.
        r: achieved rank (spike only).
        mass_fraction: share of the tensor's squared Frobenius norm carried by
            the side channel — the "outlier-heaviness" score used by the WEAK
            gate.
    """

    kind: str
    side_bits: int = 0
    idx: torch.Tensor | None = None
    values: torch.Tensor | None = None
    spike: torch.Tensor | None = None
    r: int = 0
    mass_fraction: float = 0.0
    extra: dict = field(default_factory=dict)


# ------------------------------------------------------------- accounting
def scatter_count(numel: int, p: float) -> int:
    """Number of exactly-stored entries for a top-p% scattered budget.

    Matches `phase1_fit.with_outliers`: ``max(1, int(numel * p / 100))``.
    """
    return max(1, int(numel * p / 100))


def spike_rank_for_budget(m: int, n: int, p: float) -> int:
    """Rank r whose bf16 factor cost matches a top-p% scattered budget.

    ``r = max(1, round(p/100 * m*n * 48 / ((m+n+1)*16)))``.
    The (m + n + 1) term counts the left factor column, the
    right factor row and the singular value itself.
    """
    target_bits = (p / 100.0) * m * n * SCATTER_BITS_PER_ENTRY
    unit_bits = (m + n + 1) * SPIKE_BITS_PER_UNIT
    return max(1, int(round(target_bits / unit_bits)))


def bpw_total(base_bpw: float, side_bits: int, numel: int) -> float:
    """``base + side_bits / numel``, the only bpw formula in this project."""
    return base_bpw + side_bits / numel


# ------------------------------------------------------------------ rules
def scatter_indices(w: torch.Tensor, rule: str, p: float,
                    act_scale: torch.Tensor | None = None) -> torch.Tensor:
    """Flat indices of the top-p% most salient entries under ``rule``.

    Args:
        w: [m, n] weight tensor (out_features, in_features).
        rule: ``"mag"`` (|w|) or ``"awq"`` (|w| * act_scale per input channel).
        p: budget in percent of ``w.numel()``.
        act_scale: [n] mean |input| per input channel; required for ``"awq"``.

    Returns:
        int64 flat indices, length ``scatter_count(w.numel(), p)``.
    """
    k = scatter_count(w.numel(), p)
    if rule == "mag":
        score = w.flatten().float().abs()
    elif rule == "awq":
        if act_scale is None:
            raise ValueError("awq rule requires act_scale")
        if act_scale.numel() != w.shape[1]:
            raise ValueError(
                f"act_scale has {act_scale.numel()} entries, expected {w.shape[1]}")
        s = act_scale.to(w.device).float().clamp(min=1e-12)
        score = (w.float().abs() * s.unsqueeze(0)).flatten()
    else:
        raise ValueError(f"unknown scatter rule {rule!r}")
    return score.topk(k).indices


def singular_spectrum(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full thin SVD of ``w`` in float32 on ``w``'s device.

    Returns:
        (U [m, k], S [k], Vh [k, n]) with k = min(m, n).
    """
    return torch.linalg.svd(w.float(), full_matrices=False)


def build_side_channel(w: torch.Tensor, rule: str, p: float,
                       act_scale: torch.Tensor | None = None,
                       svd: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
                       ) -> SideChannel:
    """Build the side channel for one tensor under one (rule, p).

    Args:
        w: [m, n] weight tensor.
        rule: ``"none"``, ``"mag"``, ``"awq"`` or ``"spike"``.
        p: budget in percent (ignored for ``"none"``).
        act_scale: [n] activation scale, required for ``"awq"``.
        svd: precomputed :func:`singular_spectrum` result, reused across the
            four spike budgets of the same tensor.

    Returns:
        A populated :class:`SideChannel`.
    """
    if rule == "none":
        return SideChannel(kind="none")

    wf = w.float()
    total_sq = wf.pow(2).sum().item()

    if rule in ("mag", "awq"):
        idx = scatter_indices(w, rule, p, act_scale)
        vals = wf.flatten()[idx].clone()
        return SideChannel(
            kind="scatter",
            side_bits=idx.numel() * SCATTER_BITS_PER_ENTRY,
            idx=idx,
            values=vals,
            mass_fraction=float(vals.pow(2).sum().item() / max(total_sq, 1e-30)),
        )

    if rule == "spike":
        m, n = w.shape
        r = spike_rank_for_budget(m, n, p)
        U, S, Vh = svd if svd is not None else singular_spectrum(w)
        r = min(r, S.numel())
        # Factors are *stored* in bf16 -> round-trip them before reconstructing
        # so the reported quality reflects the storage cost we charge for.
        Ur = U[:, :r].bfloat16().float()
        Sr = S[:r].bfloat16().float()
        Vr = Vh[:r, :].bfloat16().float()
        spike = (Ur * Sr.unsqueeze(0)) @ Vr
        return SideChannel(
            kind="spike",
            side_bits=(m + n + 1) * r * SPIKE_BITS_PER_UNIT,
            spike=spike,
            r=r,
            mass_fraction=float(S[:r].float().pow(2).sum().item() /
                                max(S.float().pow(2).sum().item(), 1e-30)),
            extra={"sv_top": float(S[0].item()), "k": int(S.numel())},
        )

    raise ValueError(f"unknown salience rule {rule!r}")


# --------------------------------------------------- MP atlas
def mp_bulk_edge(w: torch.Tensor) -> float:
    """Marchenko-Pastur upper bulk edge for an iid matrix of ``w``'s shape.

    For an m x n matrix with iid entries of standard deviation sigma the
    singular values concentrate below ``sigma * (sqrt(m) + sqrt(n))``.  Anything
    above that edge is a *spike* — structure the bulk cannot explain.
    """
    m, n = w.shape
    sigma = w.float().std().item()
    return sigma * (math.sqrt(m) + math.sqrt(n))


def save_spectra(spectra: dict[str, dict], path: str | Path) -> None:
    """Write the accumulated singular spectra to JSON (6 significant digits)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spectra, indent=1), encoding="utf-8")


def render_mp_atlas(spectra: dict[str, dict], path: str | Path,
                    families: list[str] | None = None) -> str:
    """Render one panel per tensor family: spectra vs the MP bulk edge.

    Each curve is one layer's singular spectrum divided by that tensor's MP
    bulk edge, so the dashed line at y = 1 is the edge for every panel at once.
    Values above the line are the RMT spikes the ``spike_*`` variants store.

    Args:
        spectra: ``{tensor_name: {"family":…, "layer":…, "s":[…], "mp_edge":…}}``.
        path: output PNG path.
        families: panel order; defaults to the 7 compressed families.

    Returns:
        The written path as a string.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if families is None:
        families = ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()
    for ax, fam in zip(axes, families):
        items = sorted((v for v in spectra.values() if v.get("family") == fam),
                       key=lambda v: v.get("layer", 0))
        if not items:
            ax.set_title(f"{fam} (no data)")
            ax.axis("off")
            continue
        n_items = len(items)
        cmap = plt.get_cmap("viridis")
        for i, item in enumerate(items):
            s = torch.tensor(item["s"], dtype=torch.float32)
            edge = max(float(item["mp_edge"]), 1e-30)
            x = torch.arange(1, s.numel() + 1, dtype=torch.float32) / s.numel()
            ax.plot(x.numpy(), (s / edge).numpy(),
                    color=cmap(i / max(n_items - 1, 1)), lw=0.9, alpha=0.85)
        ax.axhline(1.0, color="crimson", ls="--", lw=1.4)
        ax.set_yscale("log")
        ax.set_title(f"{fam}  ({n_items} layers)")
        ax.set_xlabel("singular index / min(m, n)")
        ax.set_ylabel("s / MP bulk edge")
        ax.grid(alpha=0.25)
    # 8th panel: legend / explanation
    axes[7].axis("off")
    axes[7].text(0.02, 0.95,
                 "MP atlas — Qwen3-0.6B\n\n"
                 "each curve = one layer's singular spectrum,\n"
                 "normalised by that tensor's Marchenko-Pastur\n"
                 "bulk edge  sigma * (sqrt(m) + sqrt(n)).\n\n"
                 "dashed red line = bulk edge (y = 1).\n"
                 "mass above the line = RMT spikes, i.e. the\n"
                 "structure the spike_* side channel stores.\n\n"
                 "colour = depth (dark = early layer).",
                 va="top", ha="left", fontsize=11, family="monospace")
    fig.suptitle("Singular spectra vs Marchenko-Pastur bulk edge", fontsize=15)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)
