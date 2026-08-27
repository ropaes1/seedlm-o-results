"""Build a synthetic Q4_K GGUF whose stream entropies are known analytically.

The point of this file is that the smoke test checks a *number*, not just that
the audit runs.  A float quantizer's output index distribution is whatever it
is; instead we choose the distribution first and pack genuine ``block_q4_K``
bytes around it with :func:`q4k_layout.pack_blocks`.

Three exactness tricks make the expected entropies analytic to floating-point
precision rather than merely "close":

1. **Indices, order-0.**  Pick integer counts ``c[0..15]`` summing to exactly
   256, lay down that multiset once per super-block, then shuffle the whole
   tensor globally.  Every tensor -- and any pooling of tensors -- then has an
   empirical distribution of exactly ``c/256``, so the measured ``H0`` must
   equal ``H(c/256)`` to ~1e-12.  Shuffling is what makes ``H1`` collapse back
   onto ``H0`` for this tensor (i.i.d. stream, no usable context).
2. **Indices, order-1.**  A second tensor is a mod-16 random walk:
   ``x[t+1] = (x[t] + step[t]) mod 16`` with ``step`` drawn from a chosen
   distribution ``q``.  The stationary marginal is uniform, so the true
   ``H0 = 4.0`` exactly, while the true ``H1 = H(q)`` exactly.  That pins the
   order-1 code path specifically -- a parser that mangled the nibble order
   would destroy the adjacency and push ``H1`` back up towards 4.0.
3. **6-bit scales/mins.**  Each super-block gets a permutation of 8 distinct
   ``sc`` values (exactly uniform over 8 -> ``H0 = 3.0`` exactly) and two
   copies each of 4 distinct ``m`` values (exactly uniform over 4 ->
   ``H0 = 2.0`` exactly).

The file also carries a Q6_K tensor and an F32 tensor so the smoke test
exercises the mixed-quant skip path and its byte accounting.

Usage::

    python make_test_gguf.py                 # write test-q4k.gguf
    python make_test_gguf.py --smoke-test    # write it, audit it, assert
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from q4k_layout import QK_K, pack_blocks

# ---- chosen distributions -------------------------------------------------

# Bell-shaped index distribution, the qualitative shape a real K-quant shows.
_GAUSS_CENTER, _GAUSS_SIGMA = 7.5, 3.0
# Step distribution for the mod-16 walk: strongly peaked on "stay/±1", which is
# what makes H1 << H0 and therefore a meaningful order-1 test.
_STEP_WEIGHTS = np.array([40, 20, 8, 4, 2, 1, 1, 1, 1, 1, 1, 1, 2, 4, 8, 20],
                         dtype=np.float64)

SC_VALUES = np.array([9, 17, 23, 31, 38, 44, 51, 60], dtype=np.uint8)   # 8 -> 3.0 bits
M_VALUES = np.array([5, 21, 40, 58], dtype=np.uint8)                    # 4 -> 2.0 bits


def largest_remainder(p: np.ndarray, total: int) -> np.ndarray:
    """Integer counts summing to exactly ``total``, proportional to ``p``."""
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum()
    raw = p * total
    base = np.floor(raw).astype(np.int64)
    short = total - int(base.sum())
    if short:
        order = np.argsort(-(raw - base))
        base[order[:short]] += 1
    assert base.sum() == total and (base >= 0).all()
    return base


def entropy_of_counts(counts: np.ndarray) -> float:
    """Exact Shannon entropy (bits) of the distribution ``counts/sum``."""
    c = np.asarray(counts, dtype=np.float64)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum())


def index_counts_per_block() -> np.ndarray:
    """The 16 integer counts, summing to 256, used in every super-block."""
    k = np.arange(16, dtype=np.float64)
    pmf = np.exp(-0.5 * ((k - _GAUSS_CENTER) / _GAUSS_SIGMA) ** 2)
    return largest_remainder(pmf, QK_K)


def step_counts(total: int) -> np.ndarray:
    """Integer step counts for the mod-16 walk, summing to ``total``."""
    return largest_remainder(_STEP_WEIGHTS, total)


# ---- stream builders ------------------------------------------------------


def _scales_for(n_blocks: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Per-block sc (perm of 8 values) and m (2 copies of each of 4 values)."""
    sc = np.tile(SC_VALUES, (n_blocks, 1))
    m = np.tile(np.repeat(M_VALUES, 2), (n_blocks, 1))
    # Permute within each block; the multiset per block is untouched, so the
    # entropies stay exactly 3.0 and 2.0.
    sc = rng.permuted(sc, axis=1)
    m = rng.permuted(m, axis=1)
    return sc.astype(np.uint8), m.astype(np.uint8)


def build_iid_indices(n_blocks: int, rng: np.random.Generator) -> np.ndarray:
    """``[n_blocks, 256]`` indices whose empirical pmf is exactly ``c/256``."""
    c = index_counts_per_block()
    one_block = np.repeat(np.arange(16, dtype=np.uint8), c)
    assert one_block.size == QK_K
    flat = np.tile(one_block, n_blocks)
    rng.shuffle(flat)                      # global shuffle -> i.i.d., pmf exact
    return flat.reshape(n_blocks, QK_K)


def build_markov_indices(n_blocks: int, rng: np.random.Generator
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Mod-16 random walk. Returns ``(indices[n_blocks,256], step_counts)``.

    Steps are generated *per super-block* (255 transitions each), so every
    generated step is one of the pairs the auditor actually measures -- none
    are discarded at a block boundary.
    """
    n_steps = n_blocks * (QK_K - 1)
    sc_counts = step_counts(n_steps)
    steps = np.repeat(np.arange(16, dtype=np.uint8), sc_counts)
    rng.shuffle(steps)
    steps = steps.reshape(n_blocks, QK_K - 1)

    start = rng.integers(0, 16, size=(n_blocks, 1)).astype(np.uint8)
    seq = np.concatenate([start, steps], axis=1).astype(np.int64)
    idx = np.cumsum(seq, axis=1) % 16
    return idx.astype(np.uint8), sc_counts


def _q4k_rows(idx: np.ndarray, n_rows: int, rng: np.random.Generator) -> np.ndarray:
    """Pack indices + scales into ``[n_rows, row_bytes]`` Q4_K bytes."""
    n_blocks = idx.shape[0]
    sc, m = _scales_for(n_blocks, rng)
    d = rng.uniform(0.002, 0.02, n_blocks).astype(np.float16)
    dmin = rng.uniform(0.002, 0.02, n_blocks).astype(np.float16)
    blocks = pack_blocks(idx, sc, m, d, dmin)
    return blocks.reshape(n_rows, -1)


# ---- file writer ----------------------------------------------------------


def make_test_gguf(path: str | Path, seed: int = 20260819) -> dict:
    """Write the synthetic GGUF and return the analytic expectations.

    Args:
        path: output ``.gguf`` path.
        seed: RNG seed (only affects shuffling, never the exact multisets).

    Returns:
        Dict of expected values keyed for the smoke test: ``h0_iid``,
        ``h1_iid``, ``h0_markov``, ``h1_markov``, ``h0_sc``, ``h0_min``,
        ``q4k_weights``, ``skipped_bytes`` and per-family expectations.
    """
    import gguf

    path = Path(path)
    rng = np.random.default_rng(seed)

    # --- family attn_q: two tensors, same exact index multiset -------------
    a_rows, a_cols = 512, 2048          # 4096 super-blocks
    b_rows, b_cols = 64, 256            #   64 super-blocks
    a_blocks = a_rows * a_cols // QK_K
    b_blocks = b_rows * b_cols // QK_K
    idx_a = build_iid_indices(a_blocks, rng)
    idx_b = build_iid_indices(b_blocks, rng)

    # --- family ffn_down: the mod-16 walk ---------------------------------
    m_rows, m_cols = 512, 2048          # 4096 super-blocks
    m_blocks = m_rows * m_cols // QK_K
    idx_m, step_c = build_markov_indices(m_blocks, rng)

    # --- non-Q4_K tensors, to exercise the skip path -----------------------
    q6k_rows, q6k_cols = 128, 256       # Q6_K: 210 bytes per 256 weights
    q6k_blocks = q6k_rows * q6k_cols // QK_K
    q6k_bytes = q6k_blocks * 210
    q6k = rng.integers(0, 256, size=(q6k_rows, q6k_bytes // q6k_rows), dtype=np.uint8)
    norm = rng.standard_normal(2048).astype(np.float32)

    w = gguf.GGUFWriter(str(path), "qwen3")
    w.add_block_count(2)
    w.add_tensor("blk.0.attn_q.weight", _q4k_rows(idx_a, a_rows, rng),
                 raw_dtype=gguf.GGMLQuantizationType.Q4_K)
    w.add_tensor("blk.1.attn_q.weight", _q4k_rows(idx_b, b_rows, rng),
                 raw_dtype=gguf.GGMLQuantizationType.Q4_K)
    w.add_tensor("blk.0.ffn_down.weight", _q4k_rows(idx_m, m_rows, rng),
                 raw_dtype=gguf.GGMLQuantizationType.Q4_K)
    w.add_tensor("blk.0.attn_v.weight", q6k,
                 raw_dtype=gguf.GGMLQuantizationType.Q6_K)
    w.add_tensor("blk.0.attn_norm.weight", norm)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    c = index_counts_per_block()
    h0_iid = entropy_of_counts(c)
    h_steps = entropy_of_counts(step_c)
    q4k_weights = a_rows * a_cols + b_rows * b_cols + m_rows * m_cols

    return {
        "path": str(path),
        "h0_iid": h0_iid,
        "h1_iid": h0_iid,          # i.i.d. -> context buys nothing
        "h0_markov": 4.0,          # uniform stationary marginal
        "h1_markov": h_steps,      # exactly the step-distribution entropy
        "h0_sc": 3.0,              # uniform over 8 values
        "h0_min": 2.0,             # uniform over 4 values
        "index_counts": c.tolist(),
        "step_counts": step_c.tolist(),
        "q4k_weights": q4k_weights,
        "q4k_blocks": a_blocks + b_blocks + m_blocks,
        "skipped_bytes": q6k_bytes + norm.nbytes,
        "skipped_qtypes": {"Q6_K": q6k_bytes, "F32": int(norm.nbytes)},
    }


# ---- smoke test -----------------------------------------------------------


def smoke_test(tmp: str | Path | None = None, tol: float = 0.01) -> int:
    """Build the synthetic file, audit it, assert every analytic expectation.

    Returns:
        0 on success.

    Raises:
        AssertionError: on any mismatch outside tolerance.
    """
    import entropy_audit as ea
    from q4k_layout import verify_against_gguf

    out = Path(tmp) if tmp else Path(__file__).parent / "test-q4k.gguf"
    print("[0] parser cross-check vs gguf.quants.Q4_K ...")
    print("   ", verify_against_gguf(n_blocks=256, seed=7))

    print(f"[1] building synthetic GGUF -> {out}")
    exp = make_test_gguf(out)
    print(f"    expected H0(iid)={exp['h0_iid']:.9f}  "
          f"H1(markov)={exp['h1_markov']:.9f}")

    print("[2] auditing ...")
    res = ea.audit(out, verbose=False)
    fam = {b.name: b for b in res["families"]}

    checks: list[tuple[str, float, float, float]] = []

    aq = fam["attn_q"].stats
    fd = fam["ffn_down"].stats
    tot = res["total"]

    # Exact (multiset) expectations -- tolerance 1e-9, not 0.01.
    checks.append(("attn_q  H0 index (exact multiset, 2 tensors pooled)",
                   aq.h0_idx, exp["h0_iid"], 1e-9))
    checks.append(("attn_q  H0 6-bit scales (uniform over 8)",
                   aq.h0_sc, exp["h0_sc"], 1e-9))
    checks.append(("attn_q  H0 6-bit mins   (uniform over 4)",
                   aq.h0_mn, exp["h0_min"], 1e-9))
    checks.append(("ffn_down H0 6-bit scales", fd.h0_sc, exp["h0_sc"], 1e-9))
    checks.append(("ffn_down H0 6-bit mins", fd.h0_mn, exp["h0_min"], 1e-9))

    # Sampling-limited expectations -- tolerance `tol`.
    checks.append(("attn_q  H1 index (i.i.d. -> no context gain)",
                   aq.h1_idx, exp["h1_iid"], tol))
    checks.append(("ffn_down H0 index (uniform stationary marginal)",
                   fd.h0_idx, exp["h0_markov"], tol))
    checks.append(("ffn_down H1 index (= entropy of step distribution)",
                   fd.h1_idx, exp["h1_markov"], tol))

    ok = True
    for label, got, want, t in checks:
        good = abs(got - want) <= t
        ok &= good
        print(f"    {'PASS' if good else 'FAIL'}  {label:<52s} "
              f"got={got:.9f} want={want:.9f} |d|={abs(got - want):.2e} tol={t:g}")

    # Structural / accounting checks.
    struct = [
        ("Q4_K weights counted", tot.q4k_weights, exp["q4k_weights"]),
        ("Q4_K super-blocks parsed", tot.stats.n_blocks, exp["q4k_blocks"]),
        ("unmodelled bytes", tot.other_bytes, exp["skipped_bytes"]),
        ("skipped qtypes", {k: v["bytes"] for k, v in res["skipped"].items()},
         exp["skipped_qtypes"]),
        ("stored bpw of a pure-Q4_K family", round(fam["ffn_down"].bpw(
            fam["ffn_down"].stored_bits()), 6), 4.5),
    ]
    for label, got, want in struct:
        good = got == want
        ok &= good
        print(f"    {'PASS' if good else 'FAIL'}  {label:<52s} got={got} want={want}")

    # The audit must never claim to have compressed the unmodelled bytes.
    bits1 = tot.modelled_bits(1)
    good = bits1 >= tot.other_bytes * 8
    ok &= good
    print(f"    {'PASS' if good else 'FAIL'}  "
          f"{'totals carry unmodelled bytes at full price':<52s} "
          f"{bits1:.0f} >= {tot.other_bytes * 8}")

    print()
    print(ea.render(res))
    assert ok, "smoke test FAILED -- see above"
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":  # pragma: no cover
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default=str(Path(__file__).parent / "test-q4k.gguf"))
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--tol", type=float, default=0.01)
    a = p.parse_args()
    if a.smoke_test:
        raise SystemExit(smoke_test(a.out, a.tol))
    info = make_test_gguf(a.out)
    for k, v in info.items():
        print(f"{k}: {v}")
