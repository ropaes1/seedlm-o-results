"""Phase 2 core primitives — LFSR/seed-fit re-exports plus block-level refit.

Everything numeric here is either imported verbatim from `phase1_fit.py` (the
Gate 1a/1b-validated implementation) or built directly on top of its
primitives.  Nothing is re-derived: `fit_blocks` uses the same candidate seed
set, the same `make_U`, the same `quantize_coeffs` and the same argmin-of-SSE
selection, so a block fitted here is bit-identical to the same block fitted by
`phase1_fit.seedlm_fit`.

That identity is what makes *refit-reuse* legal:
the seed fit is independent per C-block, so for an outlier variant we only need
to re-fit the blocks that intersect the held-out mask and can splice them over
the cached `seed_only` reconstruction.

Storage accounting per block of C weights (the project's storage-accounting ground rule):
    16 (seed) + 4 (shared exponent) + P*4 (coefficients) bits
    C=8, P=3 -> 4.0 bpw

*Activation-weighted fit* (the W1 probe) is an
optional objective, off by default and bit-for-bit inert when off.  Instead of
the plain per-block least squares ``min ||w_b - U c||``, the weighted path
solves ``min ||diag(s) (w_b - U c)||`` where ``s`` holds the AWQ activation
scale of the input channel each of the block's C weights belongs to.  It costs
nothing in the stored format: the coefficients ``c`` live in exactly the same
parameterisation (multipliers on the columns of the same ``U``), so the seed,
the shared exponent and the P 4-bit coefficients are unchanged and decode is
untouched.  Only *which* c is chosen changes.

The W2/W3 follow-up adds two more optional, off-by-default knobs:

* **W2, ``rounding="weighted"``** — the coefficient quantizer stops rounding
  each coefficient to its own nearest grid point and instead *searches* the
  small exact candidate set (3 exponents x 2^P round-down/round-up patterns)
  for the combination minimising the fit's own metric.  Zero bit cost, decode
  and format unchanged; see :func:`refine_coeffs`.
* **W3, ``incoherence="had"``** — the fitted object becomes
  ``T = W diag(s) H`` for a seeded orthogonal ``H`` (randomised block-Hadamard,
  :func:`incoherence_forward`), fitted with the PLAIN L2 solver.  Because ``H``
  is orthogonal, plain L2 on ``T`` *is* the ``diag(s)``-weighted metric on
  ``W`` — design A of the brief; the algebra is derived in
  :func:`incoherence_forward` and asserted numerically by the selftest.  This
  one is NOT free: decode needs ``s``, priced at
  :data:`SCALE_BITS` bits per input channel by :func:`transform_side_bits`.
"""

from __future__ import annotations

import math

import torch

# Re-exported, validated primitives.  Do not reimplement these.
from phase1_fit import (  # noqa: F401
    TAPS,
    lfsr_states,
    make_U,
    quantize_coeffs,
    rel_err,
    rtn_int4,
    seedlm_fit,
    with_outliers,
)

__all__ = [
    "TAPS",
    "lfsr_states",
    "make_U",
    "quantize_coeffs",
    "rel_err",
    "rtn_int4",
    "seedlm_fit",
    "with_outliers",
    "C_DEFAULT",
    "P_DEFAULT",
    "GENERATOR_SEED",
    "RTN_BPW",
    "BF16_BPW",
    "seed_candidates",
    "bpw_seed_base",
    "to_blocks",
    "from_blocks",
    "block_ids_for_indices",
    "fit_blocks",
    "seed_fit_tensor",
    "refit_blocks_over",
    "FIT_WEIGHTINGS",
    "WEIGHT_RIDGE",
    "normalized_col_scale",
    "block_weight_layout",
    "COEFF_ROUNDINGS",
    "INCOHERENCE_MODES",
    "EXP_OFFSETS",
    "SCALE_BITS",
    "INCOHERENCE_SCALE_FLOOR",
    "refine_coeffs",
    "decode_col_scale",
    "transform_seed",
    "incoherence_forward",
    "incoherence_inverse",
    "transform_side_bits",
    "refit_block_ids",
]

C_DEFAULT: int = 8
P_DEFAULT: int = 3
GENERATOR_SEED: int = 3407
RTN_BPW: float = 4.5      # int4 values + fp16 scale per block of 32
BF16_BPW: float = 16.0

# Fit objectives.  "none" is the historical plain-L2 fit; "awq" weights each
# weight-space column by its activation scale.
FIT_WEIGHTINGS: tuple[str, ...] = ("none", "awq")

# Relative Tikhonov term on the weighted normal equations.  The unweighted path
# uses `torch.linalg.pinv` (SVD, rank-safe); the weighted path solves the normal
# equations instead, which squares the condition number, so a 1e-6 relative
# ridge buys unconditional invertibility for a perturbation ~1e-6 of the
# coefficients — three orders of magnitude below the 4-bit coefficient grid.
WEIGHT_RIDGE: float = 1e-6

# Coefficient quantizers (W2).  "nearest" is the historical
# per-coefficient nearest-grid-point rounding; "weighted" searches the exact
# candidate set for the combination that minimises the *fit's own* metric.
COEFF_ROUNDINGS: tuple[str, ...] = ("nearest", "weighted")

# Incoherence preprocessing (W3).  "none" fits W as it stands; "had"
# fits the seeded randomised block-Hadamard rotation of the (scaled) tensor.
INCOHERENCE_MODES: tuple[str, ...] = ("none", "had")

# Shared-exponent offsets the W2 search tries around the absmax exponent the
# nearest quantizer would pick.  0 reproduces today's grid (and therefore
# guarantees the search can never do worse); -1 halves the step at the cost of
# clipping the largest coefficient; +1 doubles the range.  All three cost
# nothing because the search runs on the *one* winning seed per block.
EXP_OFFSETS: tuple[int, ...] = (-1, 0, 1)

# fp16 per stored per-input-channel scale (design A's decode-side side info).
SCALE_BITS: int = 16

# Floor on the (mean-1) column scale used by design A's fold-in.  Decode divides
# by s, so a channel with s -> 0 would have its reconstruction error amplified
# by 1/s without bound: measured on Qwen3-1.7B's cached act_scales the smallest
# normalised scale is 4.5e-6, i.e. a 220000x amplification, betting the whole
# column on 12 calibration prompts.  Flooring at 0.05 caps the amplification at
# 20x while touching almost nothing: across the 196 cached 1.7B tensors the
# median fraction of channels below 0.05 is 0.000 and the worst tensor's is
# 0.062 (quantiles of the normalised scale: 1% = 0.47, 0.1% = 0.19).  The
# floored vector is a different diagonal metric from W1's unfloored one;
# `normalized_col_scale` itself is deliberately left untouched
# so W1's measured rows stay reproducible.
INCOHERENCE_SCALE_FLOOR: float = 0.05


# --------------------------------------------------------------- seed sets
def seed_candidates(n_seeds: int, generator_seed: int = GENERATOR_SEED) -> torch.Tensor:
    """Candidate seed set, identical to the one `phase1_fit.seedlm_fit` builds.

    Full 2**16-1 sweep when ``n_seeds >= 65535`` (deterministic `arange`),
    otherwise a CPU-generator draw seeded by ``generator_seed``.

    Args:
        n_seeds: number of candidate seeds to search per block.
        generator_seed: RNG seed for the sub-sampled search.

    Returns:
        int64 tensor of shape [S] with values in [1, 65535].
    """
    if n_seeds >= 65535:
        return torch.arange(1, 1 << 16, dtype=torch.int64)
    g = torch.Generator().manual_seed(generator_seed)
    return torch.randint(1, 1 << 16, (n_seeds,), generator=g, dtype=torch.int64)


def bpw_seed_base(C: int = C_DEFAULT, P: int = P_DEFAULT) -> float:
    """Bits/weight of the seed representation alone (no side channel)."""
    return (16 + 4 + P * 4) / C


# ------------------------------------------------------------ block layout
def to_blocks(flat: torch.Tensor, C: int) -> torch.Tensor:
    """Reshape a flat weight vector into zero-padded [B, C] blocks."""
    pad = (-flat.numel()) % C
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)])
    return flat.view(-1, C)


def from_blocks(blocks: torch.Tensor, numel: int, shape: torch.Size) -> torch.Tensor:
    """Inverse of :func:`to_blocks`: drop padding and restore the tensor shape."""
    return blocks.flatten()[:numel].view(shape)


def block_ids_for_indices(idx: torch.Tensor, C: int) -> torch.Tensor:
    """Map flat weight indices to the ids of the C-blocks that contain them.

    Args:
        idx: flat indices into the (unpadded) weight vector.
        C: block size.

    Returns:
        Sorted unique int64 block ids.
    """
    return torch.unique(idx // C)


def refit_block_ids(idx: torch.Tensor, C: int, shape: torch.Size,
                    incoherence: str = "none") -> torch.Tensor:
    """Ids of the blocks a held-out mask forces a refit of, in the *fitted* space.

    Without incoherence the fitted space is W itself and one held-out weight
    dirties exactly its own block (:func:`block_ids_for_indices`).

    Under design A the fitted object is ``T = W diag(s) H`` with ``H`` dense, so
    zeroing one weight of row i perturbs *every* entry of row i of T, and
    therefore every block that overlaps that row.  Refitting a superset of the
    dirty blocks is always safe (a block whose target is unchanged refits to the
    same thing, deterministically); refitting a subset is not, which is why the
    W-space mapping must not be reused here.  At the outlier budgets we actually
    run (p >= 0.5%) every row owns at least one outlier, so this degenerates to
    a full refit — that is a real cost.

    Args:
        idx: flat indices into the (unpadded) weight vector.
        C: block size.
        shape: [m, n] shape of the tensor the indices point into.
        incoherence: ``none`` or ``had``.

    Returns:
        Sorted unique int64 block ids on ``idx``'s device.
    """
    if incoherence == "none":
        return block_ids_for_indices(idx, C)
    if len(shape) != 2:
        raise ValueError(f"incoherence needs a 2-D tensor, got {tuple(shape)}")
    m, n = int(shape[0]), int(shape[1])
    numel = m * n
    n_blocks = (numel + C - 1) // C
    dirty = torch.zeros(n_blocks * C, dtype=torch.bool, device=idx.device)
    dirty[:numel].view(m, n)[torch.unique(idx // n)] = True
    return dirty.view(n_blocks, C).any(dim=1).nonzero(as_tuple=False).flatten()


# --------------------------------------------------- W2: coefficient search
def _rounding_masks(P: int, device) -> torch.Tensor:
    """[2**P, P] float32 matrix of every round-down/round-up pattern (0/1)."""
    bits = torch.arange(1 << P, device=device, dtype=torch.int64).unsqueeze(1)
    shift = torch.arange(P, device=device, dtype=torch.int64).unsqueeze(0)
    return ((bits >> shift) & 1).to(torch.float32)


def refine_coeffs(target: torch.Tensor, U_sel: torch.Tensor,
                  t_sel: torch.Tensor, s2: torch.Tensor | None = None
                  ) -> torch.Tensor:
    """W2: pick the coefficient grid point that minimises *reconstruction* error.

    ``quantize_coeffs`` rounds each of the P coefficients to its own nearest
    4-bit grid point.  That is nearest-in-coefficient-space, and coefficient
    space is not the space damage is measured in: the rounding error
    ``dc`` reaches the weights as ``U dc`` through a basis whose columns are
    neither orthogonal nor equal-norm (and, under W1, through a warped metric on
    top).  So the independently-nearest coefficient vector is routinely not the
    grid point with the smallest error.

    This enumerates the small exact candidate set instead —
    ``len(EXP_OFFSETS) * 2**P`` points: for each shared exponent offset, every
    combination of rounding each coefficient down or up — and takes the argmin
    of the true (optionally weighted) block SSE.

    Two properties that make it safe to switch on:

    * The candidate set *contains* the nearest-rounding point at offset 0
      (``round(x)`` is either ``floor(x)`` or ``floor(x)+1`` for every x, and the
      clamp to [-8, 7] is monotone), so the searched error can never exceed the
      nearest-rounding error for the same seed and block.  Never a regression.
    * Nothing about the stored format moves: the winner is still one 4-bit
      exponent plus P 4-bit coefficients, decoded by exactly the same
      ``U (q * 2^e)``.  Zero bits, unchanged decoder.

    Cost note: this runs on the *one* seed already selected per block, not on
    all S candidates, so it is O(b * C * P * 3 * 2^P) against the selection's
    O(S * b * C * P) — under a thousandth of the fit at S=65535.  The selection
    score itself is left on nearest rounding (see :func:`fit_blocks`).

    Args:
        target: [b, C] blocks being fitted.
        U_sel: [b, C, P] basis of each block's selected seed.
        t_sel: [b, P] unquantized least-squares coefficients for that seed.
        s2: [C] squared column weights, or None for the plain metric.

    Returns:
        [b, C] reconstruction, one grid point per block.
    """
    b, C, P = U_sel.shape
    masks = _rounding_masks(P, U_sel.device)                    # [K2, P]
    absmax = t_sel.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    e0 = torch.ceil(torch.log2(absmax / 7.0))                   # [b, 1]
    coeffs = []
    for off in EXP_OFFSETS:
        scale = torch.pow(2.0, e0 + off)                        # [b, 1]
        base = torch.floor(t_sel / scale)                       # [b, P]
        q = (base.unsqueeze(1) + masks.unsqueeze(0)).clamp(-8.0, 7.0)
        coeffs.append(q * scale.unsqueeze(1))                   # [b, K2, P]
    coef = torch.cat(coeffs, dim=1)                             # [b, K, P]
    recon = torch.einsum("bcp,bkp->bkc", U_sel, coef)           # [b, K, C]
    diff = (recon - target.unsqueeze(1)).pow(2)
    err = (diff if s2 is None else diff * s2).sum(-1)           # [b, K]
    best = err.argmin(dim=1)                                    # [b]
    return recon[torch.arange(b, device=recon.device), best]


# ------------------------------------------------------- fit-time weighting
def normalized_col_scale(act_scale: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-input-channel fit weights: positive, mean 1 over the tensor.

    Mean-1 normalisation keeps the weighted objective's magnitude comparable to
    the unweighted one across tensors — the fit is
    invariant to a global rescale of ``s`` anyway, but the reported per-block
    errors are not, and neither is the ridge term.

    Idempotent: normalising an already-normalised vector returns it unchanged,
    so it does not matter how many times the value passes through here.

    Args:
        act_scale: [n] mean |input| per input channel, as captured by
            ``swap_eval.Evaluator.capture_act_scales``.
        eps: floor applied before normalising (a dead channel would otherwise
            make its whole block unconstrained).

    Returns:
        float32 [n] tensor on ``act_scale``'s device with mean 1.
    """
    s = act_scale.detach().float().flatten().clamp(min=eps)
    return s / s.mean().clamp(min=eps)


def block_weight_layout(numel: int, C: int, scales: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Column weights per C-block, factored into a small set of patterns.

    Blocks are carved out of the *flattened* [m, n] tensor, so the weight vector
    of block k is ``scales[(k*C + t) mod n]`` for t in [0, C).  That depends on k
    only through ``(k*C) mod n``, which takes exactly ``n / gcd(C, n)`` distinct
    values — 128 for a 1024-wide tensor at C=8, 256 at C=12.  Factoring the
    [B, C] weight matrix into ``patterns[group_of_block]`` is what makes the
    weighted fit affordable: the per-seed solve operator depends on the pattern
    only, so it is computed once per group instead of once per block.

    The trailing zero-padded block (when ``numel % C != 0``) simply continues
    the modular pattern; its padding entries carry zero target weight anyway.

    Args:
        numel: element count of the unpadded tensor (``m * n``).
        C: block size.
        scales: [n] column weights, normally from :func:`normalized_col_scale`.

    Returns:
        ``(patterns [G, C], group_of_block [B])`` — both on ``scales``' device,
        ``group_of_block`` int64.
    """
    n = int(scales.numel())
    if n <= 0:
        raise ValueError("scales must be non-empty")
    dev = scales.device
    n_blocks = (numel + C - 1) // C
    step = math.gcd(C, n)
    n_groups = n // step
    offsets = torch.arange(n_groups, device=dev, dtype=torch.int64) * step
    cols = (offsets.unsqueeze(1)
            + torch.arange(C, device=dev, dtype=torch.int64).unsqueeze(0)) % n
    patterns = scales[cols]                                        # [G, C]
    blocks = torch.arange(n_blocks, device=dev, dtype=torch.int64)
    group_of_block = ((blocks * C) % n) // step                    # [B]
    return patterns, group_of_block


def _weighted_solve_op(U: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
    """[S, P, C] operator ``M`` with ``c = M w`` solving the weighted fit.

    ``min_c ||diag(s) (w - U c)||^2``  =>  ``c = (U^T S^2 U)^-1 U^T S^2 w``,
    the exact analogue of the unweighted path's ``pinv(U) w``.  Note what is
    *not* scaled: the reconstruction is ``U c`` with the plain ``U``, not
    ``diag(s) U c`` — the weighting reshapes the objective, never the stored
    basis.

    Args:
        U: [S, C, P] LFSR bases.
        s2: [C] squared column weights of this block group.

    Returns:
        [S, P, C] float32 solve operator.
    """
    P = U.shape[-1]
    rhs = U.transpose(1, 2) * s2                       # [S, P, C] = U^T S^2
    gram = rhs @ U                                     # [S, P, P] = U^T S^2 U
    ridge = WEIGHT_RIDGE * gram.diagonal(dim1=-2, dim2=-1).mean(-1)      # [S]
    gram = gram + torch.eye(P, device=U.device, dtype=U.dtype) \
        * ridge[:, None, None]
    return torch.linalg.solve(gram, rhs)


# --------------------------------------------------- W3: incoherence (design A)
def decode_col_scale(col_scale: torch.Tensor | None, n: int, device,
                     floor: float = INCOHERENCE_SCALE_FLOOR) -> torch.Tensor:
    """The [n] column scale design A folds into the fitted tensor.

    ``None`` gives ones — incoherence without activation weighting, a legal
    (and side-info-free) cell of the matrix.  Otherwise the mean-1 normalised
    activation scale, floored at ``floor``; see
    :data:`INCOHERENCE_SCALE_FLOOR` for why the floor exists and what it costs.
    """
    if col_scale is None:
        return torch.ones(n, device=device, dtype=torch.float32)
    s = normalized_col_scale(col_scale).to(device=device, dtype=torch.float32)
    if s.numel() != n:
        raise ValueError(f"col_scale has {s.numel()} entries, tensor is {n} wide")
    return s.clamp(min=floor)


def transform_seed(generator_seed: int, n: int) -> int:
    """Deterministic per-width seed for H.  Storage cost: zero.

    ``H`` is a function of ``(generator_seed, n)`` only, both of which the
    decoder already knows (the generator seed is part of the config and pinned
    at 3407; n is the tensor's own width), so nothing about H is stored.  Two
    tensors of the same width share an H, which is fine: H only has to be
    data-independent, not tensor-specific.
    """
    return (int(generator_seed) * 1_000_003 + int(n) * 7919) % (1 << 31)


_TRANSFORM_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _transform_params(n: int, seed: int, device
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """(signs [n] +-1 float32, perm [n] int64) drawn on CPU, cached per device.

    Drawn from a CPU generator so the draw is identical on the laptop and the
    pod, on CPU and CUDA — the same reason :func:`seed_candidates` uses one.
    """
    key = (int(n), int(seed), str(device))
    hit = _TRANSFORM_CACHE.get(key)
    if hit is None:
        g = torch.Generator().manual_seed(int(seed))
        signs = (torch.randint(0, 2, (n,), generator=g,
                               dtype=torch.int64) * 2 - 1).to(torch.float32)
        perm = torch.randperm(n, generator=g)
        hit = (signs.to(device), perm.to(device))
        _TRANSFORM_CACHE[key] = hit
    return hit


def _hadamard_split(n: int) -> tuple[int, int]:
    """(q, L): n = q * L with L the largest power of two dividing n."""
    L = n & (-n)
    return n // L, L


def _fwht(x: torch.Tensor) -> torch.Tensor:
    """Unnormalised fast Walsh-Hadamard transform along the last axis.

    ``x @ H_L`` for the Sylvester Hadamard matrix, computed with adds and
    subtracts only — no matmul, hence no TF32 (a plain fp32 ``@`` on an Ampere+
    GPU silently truncates to 10 mantissa bits by default, which would put the
    round-trip error at ~1e-3 instead of ~1e-7).
    """
    L = x.shape[-1]
    if L == 1:
        return x.clone()
    shape = x.shape
    y = x.reshape(-1, L)
    h = 1
    while h < L:
        y = y.view(-1, L // (2 * h), 2, h)
        a, b = y[:, :, 0, :], y[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2).reshape(-1, L)
        h *= 2
    return y.reshape(shape)


def incoherence_forward(x: torch.Tensor, generator_seed: int) -> torch.Tensor:
    """``x @ H`` for the seeded orthogonal incoherence transform H.

    **The transform.**  ``H = diag(d) . Pi . (I_q kron H_L / sqrt(L))`` where d is
    a seeded +-1 vector, Pi a seeded permutation of the n input channels, and
    ``H_L`` the Sylvester Hadamard of the largest power of two L dividing n
    (``n = q * L``).  Every factor is exactly orthogonal, so H is; and every
    factor is a pure function of ``(generator_seed, n)``, so H is exactly
    reproducible at decode with zero stored bits.  The permutation is what makes
    the q Hadamard blocks mix the whole row rather than q disjoint sixths: for
    Qwen3 widths, n=2048 gives q=1 (a full randomised Hadamard) and n=6144 gives
    q=3, L=2048.  Degenerate widths (odd n -> L=1) reduce to a signed
    permutation: still exactly orthogonal, just not mixing.

    **Why plain L2 here is the weighted metric there (design A).**  With
    ``T = W diag(s) H`` and any reconstruction ``T_hat``, decoded as
    ``W_hat = T_hat H^T diag(s)^-1``,

        || (W - W_hat) diag(s) ||_F  =  || (W - W_hat) diag(s) H ||_F
                                     =  || T - T_hat ||_F

    because right-multiplying by an orthogonal matrix preserves the Frobenius
    norm.  The identity is at *tensor* level and the Frobenius norm is
    entrywise, so it survives any partition of T into blocks — including the
    blocks that straddle two rows when C does not divide n (C=12, n=2048 does
    straddle).  Blocks never need to align with anything.  What the block
    partition does decide is the feasible set, and the feasible set is now
    "block-structured in T", i.e. W_hat is dense per row: design A is a
    *different codebook* from W1's, not a refinement of it.

    Two consequences worth stating plainly: decode needs diag(s)^-1 (priced by
    :func:`transform_side_bits`), and the W-space error of a channel is
    amplified by 1/s_j, which is what :data:`INCOHERENCE_SCALE_FLOOR` bounds.

    Args:
        x: [m, n] tensor (already scaled by diag(s) if design A is scaling).
        generator_seed: run's generator seed; H is derived from it and n.

    Returns:
        [m, n] float32 transformed tensor on x's device.
    """
    if x.dim() != 2:
        raise ValueError(f"incoherence needs a 2-D tensor, got {tuple(x.shape)}")
    m, n = x.shape
    q, L = _hadamard_split(int(n))
    signs, perm = _transform_params(int(n), transform_seed(generator_seed, n),
                                    x.device)
    y = (x.float() * signs)[:, perm]
    return _fwht(y.view(m, q, L)).view(m, n) * (1.0 / math.sqrt(L))


def incoherence_inverse(y: torch.Tensor, generator_seed: int) -> torch.Tensor:
    """``y @ H^T`` — the exact inverse of :func:`incoherence_forward`.

    Exact in exact arithmetic; in fp32 the round trip is a few 1e-7 relative
    (log2(L) butterfly stages of adds plus two 1/sqrt(L) multiplies), which the
    selftest pins at 1e-5.
    """
    if y.dim() != 2:
        raise ValueError(f"incoherence needs a 2-D tensor, got {tuple(y.shape)}")
    m, n = y.shape
    q, L = _hadamard_split(int(n))
    signs, perm = _transform_params(int(n), transform_seed(generator_seed, n),
                                    y.device)
    # H_L / sqrt(L) is symmetric and orthogonal, so it is its own inverse.
    x = _fwht(y.float().view(m, q, L)).view(m, n) * (1.0 / math.sqrt(L))
    out = torch.empty_like(x)
    out[:, perm] = x                       # undo the gather x1[:, perm]
    return out * signs                     # d = +-1, so d^-1 = d


def transform_side_bits(incoherence: str, weighted: bool, shape) -> int:
    """Decode-time side information of the incoherence transform, in bits.

    H costs nothing (seed-derived, :func:`transform_seed`).  ``diag(s)^-1``
    costs :data:`SCALE_BITS` per input channel and is needed *only* when the
    scale was folded in — W1 (weighting without incoherence) uses s at fit time
    only and stays free, which is why its measured bpw is unchanged.

    ``16 * n`` bits over ``m * n`` weights is ``16 / m`` bpw: 0.0026 for a
    [6144, 2048] MLP tensor, 0.0156 for a [1024, 2048] k_proj.  Small, not zero,
    and the accounting selftest checks the exact figure.
    """
    if incoherence == "none" or not weighted:
        return 0
    return SCALE_BITS * int(shape[1])


# ----------------------------------------------------------------- fitting
def fit_blocks(blocks: torch.Tensor, C: int, P: int, n_seeds: int,
               chunk: int = 8192, generator_seed: int = GENERATOR_SEED,
               weights: torch.Tensor | None = None,
               groups: torch.Tensor | None = None,
               rounding: str = "nearest") -> torch.Tensor:
    """Best-of-N seed fit for an arbitrary set of C-blocks.

    Mirrors the inner loop of `phase1_fit.seedlm_fit` exactly (same candidate
    seeds, same auto-chunking rule to keep the [S, b, C] buffer under ~1.5 GB),
    but takes the blocks directly so a *subset* of a tensor's blocks can be
    refitted in isolation.

    With ``weights`` given the objective becomes the column-weighted one.
    Three things change and nothing else:

    1. the coefficient solve uses :func:`_weighted_solve_op` instead of
       ``pinv(U)``;
    2. the argmin over candidate seeds scores the *weighted* SSE, so the search
       and the solve optimise the same objective;
    3. blocks are visited grouped by weight pattern, so the solve operator is
       built once per pattern.

    The reconstruction, the coefficient quantizer and therefore the stored
    format are identical to the unweighted path.  ``weights=None`` runs the
    original code verbatim.

    With ``rounding="weighted"`` (W2) one further thing changes: after the seed
    argmin, the winning seed's coefficients are re-chosen by the exact search of
    :func:`refine_coeffs` under the same metric the fit uses.  The *selection*
    score stays on nearest rounding deliberately — searching all S seeds would
    multiply the dominant term of the fit by 3 * 2^P, while refining only the
    winner is free and, because the search set contains the nearest point, can
    only ever lower the error of the block that was going to be emitted anyway.
    ``rounding="nearest"`` runs the original code verbatim.

    Args:
        blocks: [B, C] float32 blocks on the compute device.
        C: block size (columns of ``blocks``).
        P: number of coefficients per block.
        n_seeds: candidate seeds per block.
        chunk: max blocks per chunk before the VRAM-driven cap is applied.
        generator_seed: RNG seed for the candidate set.
        weights: [G, C] distinct column-weight patterns, or None for the plain
            L2 objective.  See :func:`block_weight_layout`.
        groups: [B] int64 pattern id per row of ``blocks``; required with
            ``weights``.  For a refit of a *subset* of a tensor's blocks these
            must be the ids of the absolute block positions, not of the subset.
        rounding: ``nearest`` (per-coefficient nearest grid point, the
            historical quantizer) or ``weighted`` (W2's exact search).

    Returns:
        [B, C] float32 reconstruction.
    """
    if rounding not in COEFF_ROUNDINGS:
        raise ValueError(f"unknown coeff rounding {rounding!r}, want "
                         f"{list(COEFF_ROUNDINGS)}")
    device = blocks.device
    cand = seed_candidates(n_seeds, generator_seed)
    chunk = min(chunk, max(256, (1 << 31) // (len(cand) * (C + P) * 4)))

    U = make_U(cand, C, P, device)                       # [S, C, P]
    out = torch.empty_like(blocks)

    if weights is None:
        pinv = torch.linalg.pinv(U)                      # [S, P, C]
        for lo in range(0, blocks.shape[0], chunk):
            b = blocks[lo:lo + chunk]                    # [b, C]
            t = torch.einsum("spc,bc->sbp", pinv, b)     # [S, b, P]
            q, scale = quantize_coeffs(t)
            recon = torch.einsum("scp,sbp->sbc", U, q * scale)
            err = (recon - b.unsqueeze(0)).pow(2).sum(-1)        # [S, b]
            best = err.argmin(dim=0)                             # [b]
            ar = torch.arange(len(b), device=device)
            out[lo:lo + chunk] = (
                refine_coeffs(b, U[best], t[best, ar], None)
                if rounding == "weighted" else recon[best, ar])
        return out

    if groups is None:
        raise ValueError("fit_blocks: `weights` requires `groups`")
    if groups.numel() != blocks.shape[0]:
        raise ValueError(f"fit_blocks: {groups.numel()} group ids for "
                         f"{blocks.shape[0]} blocks")
    if weights.shape[-1] != C:
        raise ValueError(f"fit_blocks: weight patterns are {weights.shape[-1]} "
                         f"wide, expected C={C}")
    w2 = weights.to(device=device, dtype=torch.float32).pow(2)   # [G, C]
    groups = groups.to(device=device, dtype=torch.int64)
    order = torch.argsort(groups, stable=True)           # deterministic
    ordered = groups[order]
    uniq, counts = torch.unique_consecutive(ordered, return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    for gid, start, count in zip(uniq.tolist(), starts.tolist(), counts.tolist()):
        s2 = w2[gid]                                     # [C]
        op = _weighted_solve_op(U, s2)                   # [S, P, C]
        rows = order[start:start + count]
        for lo in range(0, count, chunk):
            sel = rows[lo:lo + chunk]
            b = blocks[sel]                              # [b, C]
            t = torch.einsum("spc,bc->sbp", op, b)       # [S, b, P]
            q, scale = quantize_coeffs(t)
            recon = torch.einsum("scp,sbp->sbc", U, q * scale)
            err = ((recon - b.unsqueeze(0)).pow(2) * s2).sum(-1)  # [S, b]
            best = err.argmin(dim=0)                             # [b]
            ar = torch.arange(len(sel), device=device)
            out[sel] = (refine_coeffs(b, U[best], t[best, ar], s2)
                        if rounding == "weighted" else recon[best, ar])
    return out


def seed_fit_tensor(w: torch.Tensor, C: int = C_DEFAULT, P: int = P_DEFAULT,
                    n_seeds: int = 16384,
                    generator_seed: int = GENERATOR_SEED,
                    col_scale: torch.Tensor | None = None,
                    rounding: str = "nearest",
                    incoherence: str = "none") -> tuple[torch.Tensor, float]:
    """Seed-fit a whole tensor through the block path.

    With ``col_scale=None``, ``rounding="nearest"`` and ``incoherence="none"``
    this is numerically identical to ``phase1_fit.seedlm_fit(w, C, P, n_seeds)``;
    the selftest asserts that equality.

    ``col_scale`` is consumed in one of two *mutually exclusive* ways:

    * ``incoherence="none"`` (W1): it becomes the diagonal metric of the
      per-block weighted solve, and is fit-time-only — zero stored bits.
    * ``incoherence="had"`` (W3, design A): it is folded into the fitted object,
      ``T = W diag(s) H``, which is then fitted with the PLAIN solver because
      plain L2 on T *is* the s-weighted metric on W (:func:`incoherence_forward`).
      Decode divides it back out, so here it is real side information.

    Args:
        w: [m, n] weight tensor.
        C, P, n_seeds, generator_seed: fit configuration.
        col_scale: raw [n] activation scale, or None.  Normalisation happens
            here (once), so callers pass the scales exactly as captured.
        rounding: coefficient quantizer, see :func:`fit_blocks`.
        incoherence: ``none`` or ``had``.

    Returns:
        (dequantized tensor with ``w``'s shape/device, seed-base bpw).
    """
    if incoherence not in INCOHERENCE_MODES:
        raise ValueError(f"unknown incoherence {incoherence!r}, want "
                         f"{list(INCOHERENCE_MODES)}")
    if incoherence == "none":
        flat = w.flatten().float()
        blocks = to_blocks(flat, C)
        patterns = groups = None
        if col_scale is not None:
            s = normalized_col_scale(col_scale).to(flat.device)
            patterns, groups = block_weight_layout(w.numel(), C, s)
        recon = fit_blocks(blocks, C, P, n_seeds, generator_seed=generator_seed,
                           weights=patterns, groups=groups, rounding=rounding)
        return from_blocks(recon, w.numel(), w.shape), bpw_seed_base(C, P)

    if w.dim() != 2:
        raise ValueError(f"incoherence needs a 2-D tensor, got {tuple(w.shape)}")
    s = decode_col_scale(col_scale, int(w.shape[1]), w.device)
    t = incoherence_forward(w.float() * s, generator_seed)
    recon = fit_blocks(to_blocks(t.flatten(), C), C, P, n_seeds,
                       generator_seed=generator_seed, weights=None, groups=None,
                       rounding=rounding)
    t_hat = from_blocks(recon, w.numel(), w.shape)
    return incoherence_inverse(t_hat, generator_seed) / s, bpw_seed_base(C, P)


def refit_blocks_over(base_recon: torch.Tensor, target_flat: torch.Tensor,
                      block_ids: torch.Tensor, C: int = C_DEFAULT,
                      P: int = P_DEFAULT, n_seeds: int = 16384,
                      generator_seed: int = GENERATOR_SEED,
                      col_scale: torch.Tensor | None = None,
                      rounding: str = "nearest",
                      incoherence: str = "none",
                      shape: torch.Size | None = None) -> torch.Tensor:
    """Splice freshly refitted blocks over a cached reconstruction.

    Refit-reuse stays legal under activation weighting: the weighted objective
    is still separable per block (a block's weights depend only on its own
    absolute position), so refitting the mask-intersecting blocks and splicing
    them over a *weighted* ``base_recon`` reproduces a full weighted refit
    exactly.  The one thing that must not slip is the indexing — the weight
    pattern of a block follows its absolute id, so ``block_ids`` indexes the
    group table before the subset is gathered.

    Under design A it stays legal but stops being a shortcut: the fit is still
    separable per T-block, yet a single held-out weight dirties every block of
    its row (:func:`refit_block_ids`), so at p >= 0.5% essentially every block is
    refitted.  Exactness also weakens from bit-identical to fp32-round-trip
    (~1e-6 relative), because the kept blocks come back through
    forward(inverse(.)) rather than being carried untouched.

    Args:
        base_recon: flat float32 reconstruction to start from (typically the
            cached ``seed_only`` fit of the same tensor), length == numel.
        target_flat: flat float32 weights the refitted blocks should fit
            (typically the original weights with the held-out entries zeroed).
        block_ids: ids of the blocks to refit (see :func:`block_ids_for_indices`).
        C, P, n_seeds, generator_seed: fit configuration; must match the run
            that produced ``base_recon``.
        col_scale: raw [n] activation scale; must match the run that produced
            ``base_recon`` (None there -> None here).
        rounding: coefficient quantizer, see :func:`fit_blocks`.
        incoherence: ``none`` or ``had``; must match the run that produced
            ``base_recon``.  Under ``had`` the splice happens in T space, so
            ``block_ids`` must be T-space ids from :func:`refit_block_ids` and
            ``shape`` is required.  In and out stay W-space flat either way.
        shape: [m, n] shape of the tensor, required when ``incoherence != none``.

    Returns:
        Flat float32 reconstruction with the selected blocks replaced.
    """
    numel = target_flat.numel()
    if incoherence != "none":
        if shape is None:
            raise ValueError("refit_blocks_over: incoherence needs `shape`")
        s = decode_col_scale(col_scale, int(shape[1]), target_flat.device)
        base_t = incoherence_forward(base_recon.view(shape) * s, generator_seed)
        tgt_t = incoherence_forward(target_flat.view(shape) * s, generator_seed)
        out_blocks = to_blocks(base_t.flatten(), C).clone()
        if block_ids.numel():
            refit = fit_blocks(to_blocks(tgt_t.flatten(), C)[block_ids], C, P,
                               n_seeds, generator_seed=generator_seed,
                               weights=None, groups=None, rounding=rounding)
            out_blocks[block_ids] = refit
        t_hat = from_blocks(out_blocks, numel, torch.Size(shape))
        return (incoherence_inverse(t_hat, generator_seed) / s).flatten()

    out_blocks = to_blocks(base_recon.clone(), C)
    tgt_blocks = to_blocks(target_flat, C)
    if block_ids.numel():
        patterns = groups = None
        if col_scale is not None:
            s = normalized_col_scale(col_scale).to(target_flat.device)
            patterns, all_groups = block_weight_layout(numel, C, s)
            groups = all_groups[block_ids.to(all_groups.device)]
        refit = fit_blocks(tgt_blocks[block_ids], C, P, n_seeds,
                           generator_seed=generator_seed,
                           weights=patterns, groups=groups, rounding=rounding)
        out_blocks[block_ids] = refit
    return out_blocks.flatten()[:numel]
