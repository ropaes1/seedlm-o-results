"""Entropy audit of a Q4_K_M GGUF's index / scale streams.

Measures how many bits a *lossless* entropy coder could strip out of a
Q4_K_M file without touching a single dequantized value.  Nothing is
re-quantized, nothing is decoded, no model is run: we parse the super-blocks,
histogram the streams, and price them at their empirical entropy.

What is measured, per super-block (144 bytes / 256 weights / 4.5 bpw stored):

======================  =========  ===========================================
stream                  stored     priced at
======================  =========  ===========================================
256 x 4-bit indices     1024 bits  ``256 * H0`` or ``H0 + 255 * H1``
8 x 6-bit sub scales      48 bits  ``8 * H0(sc)``
8 x 6-bit sub mins        48 bits  ``8 * H0(m)``
fp16 ``d`` + ``dmin``     32 bits  32 bits (left alone -- conservative)
======================  =========  ===========================================

``H1`` is the previous-weight-conditioned entropy, taken over the 255 adjacent
pairs *inside* each super-block.  Pairs are not carried across super-block
boundaries, so the number stays valid for a decoder that keeps super-blocks
independently seekable (which is what a GPU kernel wants).  Since the index
stream is recovered in weight order, "previous index" really is the previous
weight of the row, not a packing neighbour.

Non-Q4_K tensors (Q6_K/Q5_K/F32 -- a Q4_K_M file is a mix) are **not**
modelled.  Their bytes are counted at full stored size in every total and
reported separately, so the whole-model figure is a floor, never a flattering
one.

Usage::

    python entropy_audit.py <path.gguf> [--out results-<model>.md] [--top N]
    python entropy_audit.py --self-test

Memory: the GGUF is memory-mapped and walked in fixed-size block chunks, so
resident set stays flat regardless of file size.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from q4k_layout import (
    BITS_DDMIN,
    BITS_INDICES,
    BITS_MINS,
    BITS_SCALES,
    Q4K_BLOCK_BYTES,
    Q4K_STORED_BPW,
    QK_K,
    unpack_indices,
    unpack_scale_min,
    verify_against_gguf,
)

# Blocks per chunk: 16384 blocks = 2.36 MB of file, ~4.2M nibbles per pass.
# Keeps every numpy temporary (incl. bincount's intp cast) in the tens of MB.
CHUNK_BLOCKS = 16384


# ------------------------------------------------------------------ entropy


def h0(counts: np.ndarray) -> float:
    """Order-0 (plug-in) entropy of a count vector, in bits per symbol."""
    c = np.asarray(counts, dtype=np.float64)
    n = c.sum()
    if n <= 0:
        return 0.0
    p = c[c > 0] / n
    return float(-(p * np.log2(p)).sum())


def h1(joint: np.ndarray) -> float:
    """Conditional entropy ``H(next | prev)`` from a joint count matrix.

    Args:
        joint: ``[K, K]`` counts where ``joint[a, b]`` is #(prev=a, next=b).

    Returns:
        Bits per symbol.  Equals ``H(prev, next) - H(prev)``.
    """
    j = np.asarray(joint, dtype=np.float64)
    n = j.sum()
    if n <= 0:
        return 0.0
    return h0(j.ravel()) - h0(j.sum(axis=1))


def pct(num: float, den: float) -> float:
    """``num/den`` as a percentage, 0.0 when ``den`` is 0."""
    return 100.0 * num / den if den else 0.0


# ------------------------------------------------------------- accumulators


@dataclass
class StreamStats:
    """Histograms for every Q4_K stream of one tensor or one family."""

    n_blocks: int = 0
    idx: np.ndarray = field(default_factory=lambda: np.zeros(16, np.int64))
    pair: np.ndarray = field(default_factory=lambda: np.zeros(256, np.int64))
    sc: np.ndarray = field(default_factory=lambda: np.zeros(64, np.int64))
    mn: np.ndarray = field(default_factory=lambda: np.zeros(64, np.int64))

    def merge(self, other: "StreamStats") -> None:
        self.n_blocks += other.n_blocks
        self.idx += other.idx
        self.pair += other.pair
        self.sc += other.sc
        self.mn += other.mn

    # --- derived entropies -------------------------------------------------

    @property
    def h0_idx(self) -> float:
        return h0(self.idx)

    @property
    def h1_idx(self) -> float:
        return h1(self.pair.reshape(16, 16))

    @property
    def h0_sc(self) -> float:
        return h0(self.sc)

    @property
    def h0_mn(self) -> float:
        return h0(self.mn)

    # --- bit accounting ----------------------------------------------------

    def stored_bits(self) -> float:
        """Bits actually on disk for these super-blocks."""
        return float(self.n_blocks) * Q4K_BLOCK_BYTES * 8

    def modelled_bits(self, order: int) -> float:
        """Bits at the empirical entropy of these histograms.

        ``d``/``dmin`` are always charged in full; only the index and 6-bit
        scale/min streams are priced at their entropy.
        """
        nb = float(self.n_blocks)
        if order == 0:
            idx_bits = QK_K * self.h0_idx
        else:
            # First index of a super-block has no in-block predecessor.
            idx_bits = self.h0_idx + (QK_K - 1) * self.h1_idx
        return nb * (idx_bits + 8 * self.h0_sc + 8 * self.h0_mn + BITS_DDMIN)


@dataclass
class Bucket:
    """One row of the report: a tensor family, or the whole model."""

    name: str
    stats: StreamStats = field(default_factory=StreamStats)
    n_tensors: int = 0
    n_q4k_tensors: int = 0
    n_weights: int = 0          # every weight, all quant types
    stored_bytes: int = 0       # every byte, all quant types
    q4k_weights: int = 0
    other_bytes: int = 0        # bytes of non-Q4_K tensors (unmodelled)
    qtypes: set[str] = field(default_factory=set)
    # Sum over tensors of that tensor's own entropy model (adaptive / per-tensor
    # codebook) -- a tighter bound than one shared model per family.
    per_tensor_bits_h0: float = 0.0
    per_tensor_bits_h1: float = 0.0

    def stored_bits(self) -> float:
        return float(self.stored_bytes) * 8

    def modelled_bits(self, order: int, per_tensor: bool = False) -> float:
        """Total bits for this bucket, with unmodelled tensors at full price."""
        if per_tensor:
            q4k = self.per_tensor_bits_h1 if order else self.per_tensor_bits_h0
        else:
            q4k = self.stats.modelled_bits(order)
        return q4k + float(self.other_bytes) * 8

    def bpw(self, bits: float) -> float:
        return bits / self.n_weights if self.n_weights else 0.0


# ------------------------------------------------------------------ parsing


def family_of(name: str) -> str:
    """Map a GGUF tensor name to a readable family label.

    ``blk.17.ffn_down.weight`` -> ``ffn_down``; ``token_embd.weight`` ->
    ``token_embd``; ``output.weight`` -> ``output``.
    """
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] == "blk" and parts[1].isdigit():
        parts = parts[2:]
    if parts and parts[-1] in ("weight", "bias"):
        parts = parts[:-1] or [name]
    return ".".join(parts) if parts else name


def scan_q4k_tensor(data: np.ndarray, n_bytes: int) -> StreamStats:
    """Histogram every stream of one Q4_K tensor, streaming in block chunks.

    Args:
        data: memory-mapped ``uint8`` view of the tensor's bytes (may be
            multi-dimensional; it is flattened lazily).
        n_bytes: expected byte count, must be a multiple of 144.

    Returns:
        Populated :class:`StreamStats`.

    Raises:
        ValueError: if the byte count is not a whole number of super-blocks.
    """
    if n_bytes % Q4K_BLOCK_BYTES:
        raise ValueError(
            f"Q4_K tensor of {n_bytes} bytes is not a multiple of "
            f"{Q4K_BLOCK_BYTES}; layout mismatch")
    total_blocks = n_bytes // Q4K_BLOCK_BYTES
    flat = data.reshape(-1)
    st = StreamStats(n_blocks=total_blocks)

    for start in range(0, total_blocks, CHUNK_BLOCKS):
        nb = min(CHUNK_BLOCKS, total_blocks - start)
        raw = np.asarray(flat[start * Q4K_BLOCK_BYTES:(start + nb) * Q4K_BLOCK_BYTES])
        blocks = raw.reshape(nb, Q4K_BLOCK_BYTES)

        idx = unpack_indices(blocks)                       # [nb, 256] uint8
        st.idx += np.bincount(idx.ravel(), minlength=16)
        # Adjacent-pair code a*16+b, within super-block only (255 pairs/block).
        pair = (idx[:, :-1] << np.uint8(4)) | idx[:, 1:]
        st.pair += np.bincount(pair.ravel(), minlength=256)

        sc, mn = unpack_scale_min(blocks)
        st.sc += np.bincount(sc.ravel(), minlength=64)
        st.mn += np.bincount(mn.ravel(), minlength=64)

    return st


def audit(path: str | Path, verbose: bool = True) -> dict:
    """Walk a GGUF and price every Q4_K stream at its empirical entropy.

    Args:
        path: path to the ``.gguf`` file.
        verbose: print per-tensor progress to stderr.

    Returns:
        Dict with ``families`` (list of :class:`Bucket`), ``total`` (Bucket),
        ``skipped`` (per-quant-type byte counts), ``file_bytes``,
        ``n_tensors``, ``elapsed_s`` and ``model``.

    Raises:
        ValueError: on a header/layout inconsistency (loud, never silent).
    """
    from gguf.gguf_reader import GGUFReader

    path = Path(path)
    t0 = time.time()
    reader = GGUFReader(path, "r")

    families: dict[str, Bucket] = {}
    total = Bucket("TOTAL")
    skipped: dict[str, dict[str, int]] = {}

    for t in reader.tensors:
        qname = t.tensor_type.name
        fam = family_of(t.name)
        b = families.setdefault(fam, Bucket(fam))

        # --- sanity guard: header shape vs advertised element count ---------
        shape_elems = int(np.prod([int(x) for x in t.shape]))
        if shape_elems != int(t.n_elements):
            raise ValueError(
                f"{t.name}: header shape {list(map(int, t.shape))} implies "
                f"{shape_elems} elements but n_elements={t.n_elements}")
        n_bytes = int(t.n_bytes)

        b.n_tensors += 1
        b.n_weights += shape_elems
        b.stored_bytes += n_bytes
        b.qtypes.add(qname)
        total.n_tensors += 1
        total.n_weights += shape_elems
        total.stored_bytes += n_bytes
        total.qtypes.add(qname)

        if qname != "Q4_K":
            b.other_bytes += n_bytes
            total.other_bytes += n_bytes
            s = skipped.setdefault(qname, {"tensors": 0, "bytes": 0, "weights": 0})
            s["tensors"] += 1
            s["bytes"] += n_bytes
            s["weights"] += shape_elems
            continue

        if n_bytes * QK_K != shape_elems * Q4K_BLOCK_BYTES:
            raise ValueError(
                f"{t.name}: Q4_K tensor has {shape_elems} weights but "
                f"{n_bytes} bytes (expected "
                f"{shape_elems * Q4K_BLOCK_BYTES // QK_K}); layout mismatch")

        st = scan_q4k_tensor(t.data, n_bytes)
        if st.n_blocks * QK_K != shape_elems:
            raise ValueError(
                f"{t.name}: parsed {st.n_blocks} super-blocks "
                f"({st.n_blocks * QK_K} weights) vs header {shape_elems}")

        b.stats.merge(st)
        total.stats.merge(st)
        b.n_q4k_tensors += 1
        total.n_q4k_tensors += 1
        b.q4k_weights += shape_elems
        total.q4k_weights += shape_elems
        for bucket in (b, total):
            bucket.per_tensor_bits_h0 += st.modelled_bits(0)
            bucket.per_tensor_bits_h1 += st.modelled_bits(1)

        if verbose:
            print(f"  {t.name:<40s} {qname:<6s} {st.n_blocks:>9d} blk  "
                  f"H0={st.h0_idx:.4f} H1={st.h1_idx:.4f}", file=sys.stderr)

    del reader  # drop the mmap

    order = sorted(families.values(), key=lambda x: -x.stored_bytes)
    return {
        "model": path.stem,
        "path": str(path),
        "families": order,
        "total": total,
        "skipped": skipped,
        "file_bytes": path.stat().st_size,
        "n_tensors": total.n_tensors,
        "elapsed_s": time.time() - t0,
    }


# ------------------------------------------------------------------ report


def _fam_row(b: Bucket, total_bytes: int) -> str:
    s = b.stats
    stored = b.bpw(b.stored_bits())
    b0 = b.bpw(b.modelled_bits(0))
    b1 = b.bpw(b.modelled_bits(1))
    qt = ",".join(sorted(b.qtypes))
    return (f"| {b.name:<16s} | {qt:<11s} | {b.n_tensors:>4d} | "
            f"{b.n_weights / 1e6:>8.2f} | {stored:>6.3f} | {b0:>6.3f} | "
            f"{b1:>6.3f} | {stored - b1:>6.3f} | "
            f"{pct(b.stored_bytes, total_bytes):>6.2f} | "
            f"{pct(b.stored_bytes - b.other_bytes, b.stored_bytes):>5.1f} | "
            f"{s.h0_idx:>6.3f} | {s.h1_idx:>6.3f} | "
            f"{s.h0_sc:>5.3f} | {s.h0_mn:>5.3f} |")


TABLE_HEADER = (
    "| family           | qtypes      | tens |  Mweight | stored |  H0bpw |"
    "  H1bpw | recov  | %bytes | q4k% |  H0idx |  H1idx | H0sc  | H0min |\n"
    "|------------------|-------------|-----:|---------:|-------:|-------:|"
    "-------:|-------:|-------:|-----:|-------:|-------:|------:|------:|"
)


def render(res: dict) -> str:
    """Build the markdown report."""
    tot: Bucket = res["total"]
    tb = tot.stored_bytes
    stored_bits = tot.stored_bits()
    bits0 = tot.modelled_bits(0)
    bits1 = tot.modelled_bits(1)
    bits1_pt = tot.modelled_bits(1, per_tensor=True)

    bpw_stored = tot.bpw(stored_bits)
    bpw0 = tot.bpw(bits0)
    bpw1 = tot.bpw(bits1)
    bpw1_pt = tot.bpw(bits1_pt)
    recov = bpw_stored - bpw1
    shrink = pct(stored_bits - bits1, stored_bits)
    speedup = (stored_bits / bits1 - 1.0) * 100.0 if bits1 else 0.0

    L: list[str] = []
    A = L.append
    A(f"# Q4_K_M entropy audit -- `{res['model']}`")
    A("")
    A(f"- file: `{res['path']}`")
    A(f"- file bytes: {res['file_bytes']:,} "
      f"({res['file_bytes'] / 2**20:.1f} MiB); tensor bytes: {tb:,}")
    A(f"- tensors: {res['n_tensors']:,} "
      f"({tot.n_q4k_tensors:,} Q4_K, {res['n_tensors'] - tot.n_q4k_tensors:,} other)")
    A(f"- weights: {tot.n_weights:,} "
      f"({pct(tot.q4k_weights, tot.n_weights):.1f}% in Q4_K super-blocks)")
    A(f"- audit wall time: {res['elapsed_s']:.1f} s (CPU only)")
    A("")
    A("## Headline")
    A("")
    A(f"**recoverable: {recov:.3f} bpw of {bpw_stored:.3f} stored "
      f"({shrink:.1f}% smaller, ~{speedup:.1f}% decode speedup if "
      f"bandwidth-bound)**")
    A("")
    A(f"- stored             : {bpw_stored:.4f} bpw  ({stored_bits / 8 / 2**20:,.1f} MiB)")
    A(f"- order-0 model      : {bpw0:.4f} bpw  "
      f"({bits0 / 8 / 2**20:,.1f} MiB, {pct(stored_bits - bits0, stored_bits):.1f}% smaller)")
    A(f"- order-1 model      : {bpw1:.4f} bpw  "
      f"({bits1 / 8 / 2**20:,.1f} MiB, {shrink:.1f}% smaller)")
    A(f"- order-1, per-tensor: {bpw1_pt:.4f} bpw  "
      f"({bits1_pt / 8 / 2**20:,.1f} MiB, "
      f"{pct(stored_bits - bits1_pt, stored_bits):.1f}% smaller)")
    A("")
    # The Q4_K subset alone -- the number that is actually *about* Q4_K, with
    # the unmodelled Q6_K/F32 mass taken out of both sides of the ratio.
    q_stored = tot.stats.stored_bits()
    q_bits1 = tot.stats.modelled_bits(1)
    q_bits0 = tot.stats.modelled_bits(0)
    if tot.q4k_weights:
        qw = tot.q4k_weights
        A(f"Q4_K tensors only ({pct(q_stored / 8, tb):.1f}% of tensor bytes, "
          f"the part this audit actually models):")
        A("")
        A(f"- stored        : {q_stored / qw:.4f} bpw (= {Q4K_STORED_BPW} exactly)")
        A(f"- order-0 model : {q_bits0 / qw:.4f} bpw "
          f"({pct(q_stored - q_bits0, q_stored):.2f}% smaller)")
        A(f"- order-1 model : {q_bits1 / qw:.4f} bpw "
          f"({pct(q_stored - q_bits1, q_stored):.2f}% smaller)")
        A(f"- **recoverable within Q4_K: {(q_stored - q_bits1) / qw:.3f} bpw "
          f"of {q_stored / qw:.3f}**")
        A("")

    A("Index stream alone (the 4.0-bit headline number):")
    A("")
    A(f"- H0 = **{tot.stats.h0_idx:.4f}** bits/index vs 4.0 stored "
      f"({pct(4.0 - tot.stats.h0_idx, 4.0):.2f}% slack)")
    A(f"- H1 = **{tot.stats.h1_idx:.4f}** bits/index vs 4.0 stored "
      f"({pct(4.0 - tot.stats.h1_idx, 4.0):.2f}% slack)")
    A(f"- order-1 gain over order-0: {tot.stats.h0_idx - tot.stats.h1_idx:.4f} "
      "bits/index -- i.e. how much adjacency context is worth.")
    A(f"- 6-bit sub-block scales: H0 = {tot.stats.h0_sc:.4f} vs 6.0 stored "
      f"({pct(6.0 - tot.stats.h0_sc, 6.0):.1f}% slack)")
    A(f"- 6-bit sub-block mins  : H0 = {tot.stats.h0_mn:.4f} vs 6.0 stored "
      f"({pct(6.0 - tot.stats.h0_mn, 6.0):.1f}% slack)")
    A(f"- fp16 d/dmin ({BITS_DDMIN} bits/super-block) charged at full price -- "
      "not modelled, so every number above is a floor.")
    A("")
    A("### Model-wide 4-bit index distribution")
    A("")
    A("| index | " + " | ".join(f"{i:>5d}" for i in range(16)) + " |")
    A("|-------|" + "|".join(["------:"] * 16) + "|")
    n_idx = float(tot.stats.idx.sum()) or 1.0
    A("| p (%) | " + " | ".join(f"{100 * c / n_idx:5.2f}"
                                for c in tot.stats.idx) + " |")
    A("")
    A("Uniform would be 6.25% everywhere and exactly 4.0 bits; the departure "
      "from flat is the entire source of the index slack.")
    A("")
    A("## Per-family")
    A("")
    A("bpw columns are over *all* weights of the family, mixed quant types "
      "included; `q4k%` is the share of the family's bytes that were actually "
      "modelled (the rest is charged at full stored size).")
    A("")
    A(TABLE_HEADER)
    for b in res["families"]:
        L.append(_fam_row(b, tb))
    L.append(_fam_row(tot, tb))
    A("")
    A("Columns: `stored`/`H0bpw`/`H1bpw`/`recov` are whole-tensor bits per "
      "weight (indices + 6-bit scales + fp16 d/dmin). `H0idx`/`H1idx` are the "
      "index stream alone, against 4.0 stored. `H0sc`/`H0min` are the 6-bit "
      "streams, against 6.0 stored.")
    A("")
    A("## Unmodelled bytes (honest-total guard)")
    A("")
    if res["skipped"]:
        A("| qtype | tensors |       bytes | % of tensor bytes |")
        A("|-------|--------:|------------:|------------------:|")
        for q, s in sorted(res["skipped"].items(), key=lambda kv: -kv[1]["bytes"]):
            A(f"| {q:<5s} | {s['tensors']:>7d} | {s['bytes']:>11,d} | "
              f"{pct(s['bytes'], tb):>16.2f}% |")
        A("")
        A(f"Total unmodelled: {tot.other_bytes:,} bytes "
          f"({pct(tot.other_bytes, tb):.2f}% of tensor bytes). These are "
          "carried at full stored size in every total above; measuring them "
          "(Q6_K/Q5_K have their own 4/8-bit index streams) is a follow-up and "
          "would only improve the headline.")
    else:
        A("None -- every tensor was Q4_K.")
    A("")
    A("## Method")
    A("")
    A("- Super-block layout parsed from the ggml `block_q4_K` struct "
      "(144 B / 256 weights); the parser is asserted to agree *exactly* with "
      "`gguf.quants.Q4_K.dequantize_blocks` at startup (`--self-test`).")
    A("- Order-0 and order-1 are plug-in (maximum-likelihood) estimates over "
      "the whole family; order-1 conditions on the previous weight and uses "
      "only the 255 pairs inside each super-block, so super-blocks stay "
      "independently decodable.")
    A("- Entropy is the coder-independent bound; a real rANS/arithmetic coder "
      "lands within ~0.1-1% of it, and a static table costs a few KB.")
    A("- Speedup figure assumes a purely bandwidth-bound decode "
      "(bytes moved / bytes moved).")
    A("")
    return "\n".join(L) + "\n"


def render_console(res: dict) -> str:
    """Same table, sans markdown pipes, for the terminal."""
    return render(res)


# -------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("gguf", nargs="?", help="path to the .gguf file")
    p.add_argument("--out", help="markdown output path "
                                 "(default: results-<model>.md next to this script)")
    p.add_argument("--quiet", action="store_true", help="no per-tensor progress")
    p.add_argument("--self-test", action="store_true",
                   help="verify the Q4_K parser against gguf.quants and exit")
    a = p.parse_args(argv)

    if a.self_test:
        print(verify_against_gguf(n_blocks=256, seed=7))
        print("self-test OK: parser matches gguf.quants.Q4_K exactly")
        return 0

    if not a.gguf:
        p.error("a .gguf path is required (or use --self-test)")

    # Always confirm the layout before trusting a single measured bit.
    verify_against_gguf(n_blocks=64, seed=0)

    res = audit(a.gguf, verbose=not a.quiet)
    md = render(res)
    out = Path(a.out) if a.out else Path(__file__).parent / f"results-{res['model']}.md"
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"[written] {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
