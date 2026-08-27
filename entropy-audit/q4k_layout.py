"""Q4_K super-block layout: parse + pack, written from the ggml struct.

Reference (llama.cpp ``ggml/src/ggml-common.h``, verified 2026-08-19)::

    #define QK_K 256
    #define K_SCALE_SIZE 12

    typedef struct {
        ggml_half d;                  // super-block scale for quantized scales
        ggml_half dmin;               // super-block scale for quantized mins
        uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
        uint8_t qs[QK_K/2];           // 4-bit quants
    } block_q4_K;
    static_assert(sizeof(block_q4_K) == 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2);

So one super-block is **144 bytes** covering **256 weights** (4.5 bpw stored):

===========  =====  ==========================================================
byte range   bits   contents
===========  =====  ==========================================================
``[0:2]``    16     ``d``    fp16, super-block scale for the 6-bit scales
``[2:4]``    16     ``dmin`` fp16, super-block scale for the 6-bit mins
``[4:16]``   96     8 x 6-bit sub-block scales + 8 x 6-bit sub-block mins
``[16:144]`` 1024   256 x 4-bit weight indices (8 sub-blocks of 32)
===========  =====  ==========================================================

Dequant rule per sub-block ``j`` (32 weights): ``x = d*sc[j]*q - dmin*m[j]``.

Two layout details this module gets right, and which the self-test in
:func:`verify_against_gguf` pins down by exact numeric agreement with
``gguf.quants.Q4_K.dequantize_blocks``:

1. **Nibble order.**  ``qs`` is four groups of 32 bytes; group ``g`` holds
   sub-block ``2g`` in the *low* nibble and sub-block ``2g+1`` in the *high*
   nibble.  Element ``i`` of sub-block ``j`` is weight ``j*32 + i`` of the
   super-block, so the recovered index stream is in **weight order** --
   which is what makes the order-1 (previous-weight-conditioned) entropy
   meaningful rather than an artefact of the packing.
2. **6-bit scale packing.**  The 12 ``scales`` bytes split into three groups
   of four, ``a = scales[0:4]``, ``b = scales[4:8]``, ``c = scales[8:12]``::

       sc[0:4] = a & 0x3F          m[0:4] = b & 0x3F
       sc[4:8] = (c & 0x0F) | ((a >> 2) & 0x30)
       m[4:8]  = (c >> 4)   | ((b >> 2) & 0x30)

This module is standalone on purpose: it does not import or modify anything
under ``seedlm-o/``.  It was written from the ggml struct above and is then
*checked* against the ``gguf`` package's own dequantizer.
"""

from __future__ import annotations

import numpy as np

QK_K = 256
K_SCALE_SIZE = 12
Q4K_BLOCK_BYTES = 2 + 2 + K_SCALE_SIZE + QK_K // 2  # 144
Q4K_STORED_BPW = Q4K_BLOCK_BYTES * 8 / QK_K         # 4.5

# Bit budget of one super-block, by stream (sums to 144*8 = 1152).
BITS_INDICES = QK_K * 4      # 1024
BITS_SCALES = 8 * 6          #   48
BITS_MINS = 8 * 6            #   48
BITS_DDMIN = 32              #   32


def unpack_indices(blocks: np.ndarray) -> np.ndarray:
    """Recover the 256 4-bit indices per super-block, in weight order.

    Args:
        blocks: ``uint8`` array ``[n_blocks, 144]``.

    Returns:
        ``uint8`` array ``[n_blocks, 256]`` with values in ``0..15``, ordered
        so that column ``j*32 + i`` is element ``i`` of sub-block ``j``.
    """
    n = blocks.shape[0]
    qs = blocks[:, 16:Q4K_BLOCK_BYTES].reshape(n, 4, 32)
    idx = np.empty((n, 8, 32), dtype=np.uint8)
    idx[:, 0::2, :] = qs & np.uint8(0x0F)   # sub-blocks 0,2,4,6
    idx[:, 1::2, :] = qs >> np.uint8(4)     # sub-blocks 1,3,5,7
    return idx.reshape(n, QK_K)


def unpack_scale_min(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover the 8 six-bit sub-block scales and 8 six-bit mins.

    Args:
        blocks: ``uint8`` array ``[n_blocks, 144]``.

    Returns:
        ``(sc, m)``, each ``uint8`` ``[n_blocks, 8]`` with values in ``0..63``.
    """
    n = blocks.shape[0]
    s = blocks[:, 4:16]
    a, b, c = s[:, 0:4], s[:, 4:8], s[:, 8:12]
    sc = np.empty((n, 8), dtype=np.uint8)
    m = np.empty((n, 8), dtype=np.uint8)
    sc[:, 0:4] = a & np.uint8(0x3F)
    sc[:, 4:8] = (c & np.uint8(0x0F)) | ((a >> np.uint8(2)) & np.uint8(0x30))
    m[:, 0:4] = b & np.uint8(0x3F)
    m[:, 4:8] = (c >> np.uint8(4)) | ((b >> np.uint8(2)) & np.uint8(0x30))
    return sc, m


def unpack_d_dmin(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover the two fp16 super-block scales as float32 ``[n_blocks]``."""
    n = blocks.shape[0]
    dd = np.ascontiguousarray(blocks[:, 0:4]).view(np.float16).reshape(n, 2)
    return dd[:, 0].astype(np.float32), dd[:, 1].astype(np.float32)


def pack_blocks(idx: np.ndarray, sc: np.ndarray, m: np.ndarray,
                d: np.ndarray, dmin: np.ndarray) -> np.ndarray:
    """Exact inverse of the three unpackers -- builds genuine ``block_q4_K`` bytes.

    Used only by the synthetic-GGUF generator, so that a smoke-test file can be
    built with an *analytically chosen* index distribution instead of whatever
    distribution a float quantizer happens to produce.

    Args:
        idx: ``uint8`` ``[n_blocks, 256]``, values ``0..15``, weight order.
        sc: ``uint8`` ``[n_blocks, 8]``, values ``0..63``.
        m: ``uint8`` ``[n_blocks, 8]``, values ``0..63``.
        d: float ``[n_blocks]`` super-block scale (stored as fp16).
        dmin: float ``[n_blocks]`` super-block min scale (stored as fp16).

    Returns:
        ``uint8`` array ``[n_blocks, 144]``.
    """
    n = idx.shape[0]
    if idx.shape != (n, QK_K):
        raise ValueError(f"idx must be [n, {QK_K}], got {idx.shape}")
    if sc.shape != (n, 8) or m.shape != (n, 8):
        raise ValueError("sc and m must be [n, 8]")
    if idx.max(initial=0) > 15:
        raise ValueError("indices must be 0..15")
    if sc.max(initial=0) > 63 or m.max(initial=0) > 63:
        raise ValueError("scales/mins must be 0..63")

    out = np.zeros((n, Q4K_BLOCK_BYTES), dtype=np.uint8)
    dd = np.stack([np.asarray(d, dtype=np.float16),
                   np.asarray(dmin, dtype=np.float16)], axis=1)
    out[:, 0:4] = np.ascontiguousarray(dd).view(np.uint8).reshape(n, 4)

    out[:, 4:8] = (sc[:, 0:4] & np.uint8(0x3F)) | ((sc[:, 4:8] & np.uint8(0x30)) << np.uint8(2))
    out[:, 8:12] = (m[:, 0:4] & np.uint8(0x3F)) | ((m[:, 4:8] & np.uint8(0x30)) << np.uint8(2))
    out[:, 12:16] = (sc[:, 4:8] & np.uint8(0x0F)) | ((m[:, 4:8] & np.uint8(0x0F)) << np.uint8(4))

    q = idx.reshape(n, 8, 32)
    out[:, 16:] = (q[:, 0::2, :] | (q[:, 1::2, :] << np.uint8(4))).reshape(n, 128)
    return out


def verify_against_gguf(n_blocks: int = 64, seed: int = 0) -> dict[str, float]:
    """Cross-check this parser against ``gguf.quants.Q4_K``.

    Builds random super-blocks, dequantizes them two ways -- once with the
    ``gguf`` package's own ``Q4_K.dequantize_blocks`` (the reference), once by
    applying ``x = d*sc[j]*q - dmin*m[j]`` to the streams this module parses --
    and requires *exact* float32 agreement.  A wrong nibble order, a wrong
    6-bit unpack, or a swapped ``d``/``dmin`` all break it loudly.

    Also asserts our :func:`unpack_scale_min` equals ``Q4_K.get_scale_min``
    bit-for-bit, and that :func:`pack_blocks` round-trips.

    Returns:
        Dict with ``max_abs_err``, ``n_blocks`` and ``roundtrip_exact``.

    Raises:
        AssertionError: on any mismatch.
    """
    from gguf.constants import GGMLQuantizationType
    from gguf.quants import Q4_K

    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, size=(n_blocks, Q4K_BLOCK_BYTES), dtype=np.uint8)
    # Keep d/dmin finite and ordinary: overwrite the fp16 pair with sane values.
    dd = np.stack([rng.uniform(1e-3, 1e-1, n_blocks).astype(np.float16),
                   rng.uniform(1e-3, 1e-1, n_blocks).astype(np.float16)], axis=1)
    blocks[:, 0:4] = np.ascontiguousarray(dd).view(np.uint8).reshape(n_blocks, 4)

    ref = np.asarray(Q4_K.dequantize_blocks(blocks)).astype(np.float32)

    sc_ref, m_ref = Q4_K.get_scale_min(blocks[:, 4:16])
    sc, m = unpack_scale_min(blocks)
    assert np.array_equal(sc, np.asarray(sc_ref)), "6-bit scale unpack disagrees with gguf"
    assert np.array_equal(m, np.asarray(m_ref)), "6-bit min unpack disagrees with gguf"

    idx = unpack_indices(blocks)
    d, dmin = unpack_d_dmin(blocks)
    mine = ((d[:, None, None] * sc[:, :, None].astype(np.float32))
            * idx.reshape(n_blocks, 8, 32).astype(np.float32)
            - (dmin[:, None, None] * m[:, :, None].astype(np.float32))
            ).reshape(n_blocks, QK_K).astype(np.float32)

    max_err = float(np.max(np.abs(ref - mine)))
    assert max_err == 0.0, f"dequant disagrees with gguf.quants.Q4_K (max err {max_err})"

    repacked = pack_blocks(idx, sc, m, d, dmin)
    rt = bool(np.array_equal(repacked, blocks))
    assert rt, "pack_blocks is not the inverse of the unpackers"

    assert GGMLQuantizationType.Q4_K.name == "Q4_K"
    return {"max_abs_err": max_err, "n_blocks": float(n_blocks), "roundtrip_exact": rt}


if __name__ == "__main__":  # pragma: no cover
    print(verify_against_gguf())
    print("q4k_layout: OK -- exact agreement with gguf.quants.Q4_K")
