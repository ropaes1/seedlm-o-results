"""Phase 3 real comparators — dequantize foreign quantizations back to torch.

The same-stack determinism rule forbids comparing a llama.cpp-runtime output
against a torch-runtime output.  So Phase 3 does not compare *runtimes*, it
compares *weights*: every comparator (Q4_K_M, Q3_K_M, AWQ W4A16) is dequantized
back to dense bf16 and swapped into the same torch model, through the same
:class:`swap_eval.Evaluator`, as our own seed variants.  The only thing that
differs between a comparator row and a seed row is the numbers in the tensor.

Two sources are supported:

* **GGUF K-quants** (`Q4_K_M`, `Q3_K_M`, imatrix-calibrated) produced on the pod
  by ``llama-quantize``.  Read with the ``gguf`` package's :mod:`gguf.quants`
  numpy dequantizers (no llama.cpp runtime involved) and re-mapped from GGUF
  tensor names back to HF parameter names.
* **AWQ W4A16 group-128** produced on the pod by ``llmcompressor`` (or any
  AutoAWQ-format checkpoint), unpacked from its int32 bit-packing.

Effective bits/weight for every comparator is computed from the **stored bytes**
of that comparator's own tensors (GGUF ``n_bytes``; AWQ qweight+scales+zeros),
never from the nominal label — the project's storage-accounting ground rule applies to their bits
exactly as it applies to ours.

Local testability: the GGUF *dequant* path is unit-tested offline by
:func:`q4k_roundtrip_check` (AC-4), which encodes a random tensor into genuine
ggml ``block_q4_K`` bytes with :func:`quantize_q4_k_reference` and decodes them
with the same ``gguf.quants`` call the production path uses.  The AWQ unpack and
the end-to-end GGUF file reading are **pod-tested only** (they need artifacts we
do not build on the laptop); their docstrings say so.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

__all__ = [
    "GGUF_BLOCK_FOR_FAMILY",
    "hf_to_gguf_name",
    "gguf_to_hf_name",
    "ComparatorTensor",
    "GGUFDequantizer",
    "AWQDequantizer",
    "quantize_q4_k_reference",
    "q4k_roundtrip_check",
    "awq_effective_bits",
]


# --------------------------------------------------------------- name mapping
# Inverted from llama.cpp's ``convert_hf_to_gguf.py`` (TensorNameMap, LLM_ARCH
# blocks used for Qwen2/Qwen3): the seven linear families we compress map
# one-to-one onto GGUF ``blk.{i}.*`` names.
GGUF_BLOCK_FOR_FAMILY: dict[str, str] = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}
_FAMILY_FOR_GGUF: dict[str, str] = {v: k for k, v in GGUF_BLOCK_FOR_FAMILY.items()}

_HF_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)\.weight$")
_GGUF_RE = re.compile(r"^blk\.(\d+)\.(.+)\.weight$")


def hf_to_gguf_name(hf_name: str) -> str:
    """``model.layers.3.mlp.up_proj.weight`` -> ``blk.3.ffn_up.weight``.

    Args:
        hf_name: HF parameter name of one of the seven compressed families.

    Returns:
        The GGUF tensor name ``llama.cpp`` writes for the same weight.

    Raises:
        KeyError: if the name is not one of the seven families (fail loud —
            a silently unmapped tensor would corrupt a comparator row).
    """
    m = _HF_RE.match(hf_name)
    if not m:
        raise KeyError(f"not a mappable HF layer tensor: {hf_name!r}")
    layer, family = m.group(1), m.group(2)
    if family not in GGUF_BLOCK_FOR_FAMILY:
        raise KeyError(f"no GGUF mapping for family {family!r} (from {hf_name!r})")
    return f"blk.{layer}.{GGUF_BLOCK_FOR_FAMILY[family]}.weight"


def gguf_to_hf_name(gguf_name: str) -> str:
    """Inverse of :func:`hf_to_gguf_name`.

    Raises:
        KeyError: if the GGUF name is not one of the seven mapped families.
    """
    m = _GGUF_RE.match(gguf_name)
    if not m:
        raise KeyError(f"not a mappable GGUF block tensor: {gguf_name!r}")
    layer, block = m.group(1), m.group(2)
    if block not in _FAMILY_FOR_GGUF:
        raise KeyError(f"no HF mapping for GGUF block {block!r} (from {gguf_name!r})")
    return f"model.layers.{layer}.{_FAMILY_FOR_GGUF[block]}.weight"


# ------------------------------------------------------------------ container
@dataclass
class ComparatorTensor:
    """One dequantized comparator tensor plus its honest storage cost.

    Attributes:
        hf_name: HF parameter name the tensor is swapped into.
        weight: dense float32 [out_features, in_features] reconstruction.
        stored_bits: bits this comparator actually spends on this tensor, taken
            from the container's own byte accounting.
        qtype: label of the source quantization (e.g. ``"Q4_K"``, ``"awq_w4"``).
    """

    hf_name: str
    weight: torch.Tensor
    stored_bits: int
    qtype: str

    @property
    def bpw(self) -> float:
        """Effective bits per weight for this tensor."""
        return self.stored_bits / max(self.weight.numel(), 1)


# ------------------------------------------------------------- GGUF K-quants
class GGUFDequantizer:
    """Read a GGUF file and hand back dense torch tensors under HF names.

    Uses :func:`gguf.quants.dequantize` — the ``gguf`` Python package's numpy
    K-quant decoders — so no llama.cpp runtime is involved and the result is a
    plain array we can swap into the torch model.

    Args:
        path: path to the quantized ``.gguf`` file.

    Note:
        Constructing this class needs a real GGUF file, which only exists on the
        pod (``llama-quantize`` output).  The *decode* it delegates to is the
        same call unit-tested offline by :func:`q4k_roundtrip_check`.
    """

    def __init__(self, path: str | Path) -> None:
        import gguf  # imported lazily: only the comparator stage needs it

        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"GGUF not found: {self.path}")
        self.reader = gguf.GGUFReader(str(self.path), "r")
        self.tensors = {t.name: t for t in self.reader.tensors}

    # ------------------------------------------------------------- coverage
    def assert_covers(self, hf_names: list[str]) -> None:
        """Fail loud unless every requested HF tensor exists in the GGUF.

        Args:
            hf_names: HF parameter names the run intends to swap.

        Raises:
            KeyError: listing every HF name whose GGUF counterpart is absent.
        """
        missing = []
        for hf in hf_names:
            if hf_to_gguf_name(hf) not in self.tensors:
                missing.append(hf)
        if missing:
            raise KeyError(
                f"{self.path.name}: {len(missing)}/{len(hf_names)} target tensors "
                f"have no GGUF counterpart, first few: {missing[:5]}")

    def unmapped_block_tensors(self) -> list[str]:
        """GGUF ``blk.*`` tensors that our mapping does not understand.

        Reported (not fatal) by the comparator stage: norms, biases and the
        token embedding legitimately fall outside the seven compressed families,
        but an unexpected *projection* name here means the architecture moved
        and the coverage assertion above is measuring the wrong thing.
        """
        out = []
        for name in self.tensors:
            if not name.startswith("blk."):
                continue
            try:
                gguf_to_hf_name(name)
            except KeyError:
                out.append(name)
        return out

    # -------------------------------------------------------------- decoding
    def dequantize(self, hf_name: str) -> ComparatorTensor:
        """Decode one tensor to dense float32 under its HF name.

        Args:
            hf_name: HF parameter name (one of the seven families).

        Returns:
            A :class:`ComparatorTensor` whose ``stored_bits`` comes from the
            GGUF tensor's own ``n_bytes``.
        """
        import gguf
        from gguf.quants import dequantize as gguf_dequantize

        t = self.tensors[hf_to_gguf_name(hf_name)]
        # GGUF stores shapes in ggml order (fastest-varying first); torch wants
        # [out_features, in_features], i.e. the reverse.
        shape = tuple(int(x) for x in reversed(t.shape))
        n_rows = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
        raw = np.asarray(t.data)
        if t.tensor_type in (gguf.GGMLQuantizationType.F32,
                             gguf.GGMLQuantizationType.F16,
                             gguf.GGMLQuantizationType.BF16):
            deq = gguf_dequantize(raw.reshape(-1).view(np.uint8), t.tensor_type)
        else:
            deq = gguf_dequantize(raw.reshape(n_rows, -1).view(np.uint8),
                                  t.tensor_type)
        deq = np.ascontiguousarray(deq).reshape(shape).astype(np.float32)
        return ComparatorTensor(
            hf_name=hf_name,
            weight=torch.from_numpy(deq),
            stored_bits=int(t.n_bytes) * 8,
            qtype=t.tensor_type.name,
        )

    def iter_targets(self, hf_names: list[str]) -> Iterator[ComparatorTensor]:
        """Yield every requested tensor, decoded, after asserting coverage."""
        self.assert_covers(hf_names)
        for hf in hf_names:
            yield self.dequantize(hf)

    def close(self) -> None:
        """Drop the memory map."""
        self.reader = None
        self.tensors = {}


# ---------------------------------------------------------------------- AWQ
def awq_effective_bits(out_features: int, in_features: int, group_size: int,
                       zero_point: bool = True, scale_bits: int = 16) -> int:
    """Stored bits for one AWQ W4A16 tensor, counted adversarially.

    4 bits per weight (int32-packed, 8 per word) + one ``scale_bits`` scale per
    (group, output channel) + one 4-bit zero-point per (group, output channel)
    when the scheme is asymmetric.

    Args:
        out_features: rows of the dense weight.
        in_features: columns of the dense weight.
        group_size: AWQ group size along the input dimension (128 by default).
        zero_point: whether zero-points are stored (asymmetric AWQ).
        scale_bits: bit width of the stored scales (fp16 -> 16).

    Returns:
        Total stored bits for the tensor.
    """
    n_groups = (in_features + group_size - 1) // group_size
    bits = 4 * out_features * in_features
    bits += scale_bits * n_groups * out_features
    if zero_point:
        bits += 4 * n_groups * out_features
    return int(bits)


class AWQDequantizer:
    """Unpack an AWQ W4A16 checkpoint back to dense float32 tensors.

    Handles the two layouts Phase 3 can encounter on the pod:

    * **AutoAWQ / AWQ-GEMM**: ``<prefix>.qweight`` int32 ``[in, out//8]``,
      ``<prefix>.qzeros`` int32 ``[in//g, out//8]``, ``<prefix>.scales`` fp16
      ``[in//g, out]``, with AWQ's interleaved nibble order ``[0,4,1,5,2,6,3,7]``.
    * **compressed-tensors** (``llmcompressor`` default): ``weight_packed``
      int32 ``[out, in//8]``, ``weight_scale`` ``[out, in//g]``,
      ``weight_zero_point`` optional, contiguous nibble order.

    Args:
        model_dir: directory holding the quantized checkpoint (safetensors +
            ``config.json`` with a ``quantization_config`` block).

    Note:
        **Pod-tested only.**  We do not produce an AWQ checkpoint on the laptop,
        so this path has no local unit test; the layouts above are the
        documented ones and the class fails loud rather than guessing when a
        tensor's shapes do not match the layout it picked.
    """

    _AWQ_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)

    def __init__(self, model_dir: str | Path) -> None:
        from safetensors import safe_open

        self.dir = Path(model_dir)
        cfg_path = self.dir / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        qcfg = cfg.get("quantization_config", {}) or {}
        self.group_size = int(qcfg.get("group_size", 128) or 128)
        self.zero_point = bool(qcfg.get("zero_point", True))
        self._safe_open = safe_open
        self._index: dict[str, Path] = {}
        idx = self.dir / "model.safetensors.index.json"
        if idx.exists():
            weight_map = json.loads(idx.read_text(encoding="utf-8"))["weight_map"]
            for k, fn in weight_map.items():
                self._index[k] = self.dir / fn
        else:
            for f in sorted(self.dir.glob("*.safetensors")):
                with safe_open(str(f), framework="pt") as fh:
                    for k in fh.keys():
                        self._index[k] = f

    def _get(self, key: str) -> torch.Tensor | None:
        path = self._index.get(key)
        if path is None:
            return None
        with self._safe_open(str(path), framework="pt") as fh:
            return fh.get_tensor(key)

    @staticmethod
    def _unpack_int32_nibbles(packed: torch.Tensor, order: tuple[int, ...] | None
                              ) -> torch.Tensor:
        """int32 word -> 8 unsigned 4-bit values along a new last axis.

        Args:
            packed: int32 tensor whose last axis packs 8 nibbles per element.
            order: nibble permutation to undo (AWQ interleaving), or None for
                contiguous low-to-high order.

        Returns:
            int32 tensor with the last axis expanded 8x.
        """
        shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=packed.device)
        vals = (packed.to(torch.int32).unsqueeze(-1) >> shifts) & 0x0F
        if order is not None:
            inv = [0] * 8
            for pos, src in enumerate(order):
                inv[src] = pos
            vals = vals[..., inv]
        return vals.reshape(*packed.shape[:-1], packed.shape[-1] * 8)

    def dequantize(self, hf_name: str) -> ComparatorTensor:
        """Dequantize one HF weight from its AWQ packing.

        Args:
            hf_name: ``model.layers.{i}.{block}.{family}.weight``.

        Returns:
            A :class:`ComparatorTensor` with float32 ``[out, in]`` weights and
            stored bits from :func:`awq_effective_bits`.

        Raises:
            KeyError: if neither supported layout is present for the tensor.
        """
        prefix = hf_name[: -len(".weight")]
        g = self.group_size

        qweight = self._get(f"{prefix}.qweight")
        if qweight is not None:                          # AutoAWQ / AWQ-GEMM
            scales = self._get(f"{prefix}.scales")
            qzeros = self._get(f"{prefix}.qzeros")
            in_features = int(qweight.shape[0])
            out_features = int(scales.shape[1])
            w = self._unpack_int32_nibbles(qweight, self._AWQ_ORDER).float()
            if w.shape != (in_features, out_features):
                raise KeyError(f"{hf_name}: AWQ qweight unpacks to {tuple(w.shape)}, "
                               f"expected {(in_features, out_features)}")
            s = scales.float()                           # [in//g, out]
            if qzeros is not None:
                z = self._unpack_int32_nibbles(qzeros, self._AWQ_ORDER).float()
            else:
                z = torch.zeros_like(s)
            rep = torch.repeat_interleave(torch.arange(s.shape[0]), g)[:in_features]
            dense = (w - z[rep]) * s[rep]                # [in, out]
            dense = dense.t().contiguous()               # -> [out, in]
        else:                                            # compressed-tensors
            packed = self._get(f"{prefix}.weight_packed")
            if packed is None:
                raise KeyError(f"{hf_name}: no AWQ tensors "
                               f"({prefix}.qweight / .weight_packed both absent)")
            scales = self._get(f"{prefix}.weight_scale")
            zeros = self._get(f"{prefix}.weight_zero_point")
            out_features = int(packed.shape[0])
            w = self._unpack_int32_nibbles(packed, None).float()   # [out, in]
            in_features = int(w.shape[1])
            s = scales.float()                                     # [out, in//g]
            z = zeros.float() if zeros is not None else torch.zeros_like(s)
            if s.shape[1] * g < in_features:
                raise KeyError(f"{hf_name}: weight_scale {tuple(s.shape)} does not "
                               f"cover in_features={in_features} at group {g}")
            rep = torch.repeat_interleave(torch.arange(s.shape[1]), g)[:in_features]
            dense = (w - z[:, rep]) * s[:, rep]
        return ComparatorTensor(
            hf_name=hf_name,
            weight=dense.float(),
            stored_bits=awq_effective_bits(out_features, in_features, g,
                                           self.zero_point),
            qtype="awq_w4",
        )

    def iter_targets(self, hf_names: list[str]) -> Iterator[ComparatorTensor]:
        """Yield every requested tensor, dequantized; fails loud on any miss."""
        for hf in hf_names:
            yield self.dequantize(hf)


# ---------------------------------------------- Q4_K reference encoder (AC-4)
_QK_K = 256
_K_SCALE_SIZE = 12
_Q4K_TYPE_SIZE = 2 + 2 + _K_SCALE_SIZE + _QK_K // 2      # 144 bytes


def quantize_q4_k_reference(w: np.ndarray) -> np.ndarray:
    """Encode float data into genuine ggml ``block_q4_K`` bytes.

    Written directly from the ggml struct definition::

        typedef struct {
            ggml_half d;                 // super-block scale for the scales
            ggml_half dmin;              // super-block scale for the mins
            uint8_t   scales[12];        // 8 x 6-bit scales + 8 x 6-bit mins
            uint8_t   qs[128];           // 256 x 4-bit quants
        } block_q4_K;

    with the affine sub-block rule ``x ~= d*sc[j]*q - dmin*m[j]`` over eight
    32-element sub-blocks.  This is the *simple* (non-imatrix) quantizer, which
    is all the round-trip unit needs: it exists so the decode path can be
    exercised offline against bytes produced by an implementation that does not
    share a line of code with :mod:`gguf.quants`.

    Args:
        w: float array whose last axis length is a multiple of 256.

    Returns:
        ``uint8`` array of shape ``[n_blocks, 144]``.
    """
    flat = np.ascontiguousarray(w, dtype=np.float32).reshape(-1)
    if flat.size % _QK_K:
        raise ValueError(f"Q4_K needs a multiple of {_QK_K} elements, got {flat.size}")
    x = flat.reshape(-1, 8, 32)                          # [B, 8 sub-blocks, 32]
    n_blocks = x.shape[0]

    sub_max = x.max(axis=-1)                             # [B, 8]
    sub_min = np.minimum(x.min(axis=-1), 0.0)            # mins are stored negated
    scale = np.maximum((sub_max - sub_min) / 15.0, 1e-30)
    negmin = -sub_min                                    # >= 0

    d = np.float16(np.maximum(scale.max(axis=-1) / 63.0, 1e-30))       # [B]
    dmin = np.float16(np.maximum(negmin.max(axis=-1) / 63.0, 1e-30))   # [B]
    df, dminf = d.astype(np.float32), dmin.astype(np.float32)

    sc = np.clip(np.rint(scale / df[:, None]), 0, 63).astype(np.uint8)     # [B, 8]
    mn = np.clip(np.rint(negmin / dminf[:, None]), 0, 63).astype(np.uint8)  # [B, 8]

    eff_scale = np.maximum(df[:, None] * sc.astype(np.float32), 1e-30)
    eff_min = -(dminf[:, None] * mn.astype(np.float32))
    q = np.clip(np.rint((x - eff_min[..., None]) / eff_scale[..., None]),
                0, 15).astype(np.uint8)                  # [B, 8, 32]

    out = np.zeros((n_blocks, _Q4K_TYPE_SIZE), dtype=np.uint8)
    out[:, 0:2] = d.view(np.uint8).reshape(n_blocks, 1).repeat(2, axis=1) \
        if False else np.ascontiguousarray(d).view(np.uint8).reshape(n_blocks, 2)
    out[:, 2:4] = np.ascontiguousarray(dmin).view(np.uint8).reshape(n_blocks, 2)

    # Pack 8 x 6-bit scales and 8 x 6-bit mins into 12 bytes (inverse of
    # gguf.quants.Q4_K.get_scale_min).
    j = np.arange(4)
    out[:, 4 + j] = (sc[:, j] & 0x3F) | ((sc[:, j + 4] & 0x30) << 2)
    out[:, 8 + j] = (mn[:, j] & 0x3F) | ((mn[:, j + 4] & 0x30) << 2)
    out[:, 12 + j] = (sc[:, j + 4] & 0x0F) | ((mn[:, j + 4] & 0x0F) << 4)

    # qs: byte group i holds sub-block 2i in the low nibble, 2i+1 in the high.
    lo = q[:, 0::2, :]                                   # [B, 4, 32]
    hi = q[:, 1::2, :]
    out[:, 16:] = (lo | (hi << 4)).reshape(n_blocks, 128)
    return out


def gguf_file_roundtrip_check(tmpdir: str | Path,
                              shape: tuple[int, int] = (32, 256),
                              seed: int = 1) -> dict:
    """End-to-end check of the *production* GGUF reader, with no llama.cpp.

    Writes a real ``.gguf`` file (``gguf.GGUFWriter``) whose single tensor is
    genuine ``block_q4_K`` bytes from :func:`quantize_q4_k_reference` under the
    GGUF tensor name ``blk.0.attn_q.weight``, then reads it back through
    :class:`GGUFDequantizer` — the same class the comparator stage uses on
    llama-quantize output.  That exercises the whole chain the pod depends on:
    header parsing, ggml-vs-torch shape reversal, per-tensor type dispatch, the
    GGUF->HF name mapping, coverage assertion, and the ``n_bytes``-derived bpw.

    Args:
        tmpdir: directory to write the throwaway GGUF into.
        shape: ``[out_features, in_features]``; in_features must be a multiple
            of 256.
        seed: numpy RNG seed.

    Returns:
        Dict with ``hf_name``, ``shape_ok``, ``dtype``, ``bpw``, ``qtype``,
        ``rel_err``, ``in_band`` and ``unmapped``.
    """
    import gguf

    tmpdir = Path(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    path = tmpdir / "roundtrip.gguf"
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(shape).astype(np.float32) * 0.02
    blocks = quantize_q4_k_reference(w)
    rows = blocks.reshape(shape[0], -1)

    writer = gguf.GGUFWriter(str(path), "qwen3")
    writer.add_block_count(1)
    writer.add_tensor("blk.0.attn_q.weight", rows,
                      raw_dtype=gguf.GGMLQuantizationType.Q4_K)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    hf_name = "model.layers.0.self_attn.q_proj.weight"
    reader = GGUFDequantizer(path)
    reader.assert_covers([hf_name])
    ct = reader.dequantize(hf_name)
    unmapped = reader.unmapped_block_tensors()
    reader.close()
    rel = float(np.linalg.norm(w - ct.weight.numpy()) / np.linalg.norm(w))
    return {
        "hf_name": ct.hf_name,
        "shape_ok": tuple(ct.weight.shape) == shape,
        "dtype": str(ct.weight.dtype),
        "bpw": ct.bpw,
        "qtype": ct.qtype,
        "rel_err": rel,
        "in_band": 0.05 <= rel <= 0.15,
        "unmapped": unmapped,
        "file_bytes": path.stat().st_size,
    }


def q4k_roundtrip_check(shape: tuple[int, int] = (64, 512), seed: int = 0
                        ) -> dict[str, float | str | bool]:
    """AC-4: encode -> ``gguf.quants.dequantize`` -> error band.

    The bytes are produced by :func:`quantize_q4_k_reference` (written from the
    ggml struct layout) and decoded by the very function the comparator stage
    calls, so a layout misreading on either side shows up as a wrecked relative
    error rather than passing silently.

    Args:
        shape: shape of the random test tensor; last axis must be a multiple of
            256.
        seed: numpy RNG seed.

    Returns:
        Dict with ``rel_err``, ``dtype``, ``shape_ok``, ``in_band`` and the
        stored ``bpw``.
    """
    import gguf
    from gguf.quants import dequantize as gguf_dequantize

    rng = np.random.default_rng(seed)
    w = rng.standard_normal(shape).astype(np.float32) * 0.02
    blocks = quantize_q4_k_reference(w)
    packed = blocks.reshape(shape[0], -1)
    deq = gguf_dequantize(packed, gguf.GGMLQuantizationType.Q4_K)
    deq = np.asarray(deq).reshape(shape).astype(np.float32)
    rel = float(np.linalg.norm(w - deq) / np.linalg.norm(w))
    return {
        "rel_err": rel,
        "dtype": str(deq.dtype),
        "shape_ok": deq.shape == shape,
        "in_band": 0.05 <= rel <= 0.15,
        "bpw": blocks.size * 8 / float(w.size),
    }
