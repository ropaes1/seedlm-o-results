#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ladder_bench.py -- the bytes-touched ladder.

Orchestrates one rung of the laptop-inference ladder, parses llama-bench /
llama-server output, estimates bytes-touched-per-token, samples RAM+VRAM,
and CRASH-SAFELY appends one row per measurement to ladder/results.jsonl
(fsync'd) before re-rendering ladder/results.md.

Design constraint: the target machine has a known hard-crash fault under
sustained GPU+CPU load. Therefore *every* measurement is durably on disk the
instant it exists. A crash loses at most the rung currently in flight.

Requires: Python 3.12, stdlib only.
psutil is OPTIONAL (better per-process RSS); without it we use the Win32
GlobalMemoryStatusEx via ctypes, which is enough for system-wide RAM.
    pip install psutil        # ~0.5 MB, optional

Subcommands
-----------
  selftest        Run the built-in fixtures end-to-end (no binaries needed).
  run             Run a rung with llama-bench (R0/R1/R2, incl. -ncmoe sweep).
  spec            Run the speculative-decoding variant via llama-server
                  (llama-bench does NOT support speculative decoding).
  ingest          Parse an already-written llama-bench JSON file (crash recovery).
  render          Re-render results.md from results.jsonl.
  bytes           Print the bytes/token estimate for a model (and show the math).

Run `python ladder_bench.py <cmd> --help` for options.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_JSONL = HERE / "results.jsonl"
RESULTS_MD = HERE / "results.md"
LOG_DIR = HERE / "logs"

WIN_CONDITION_TOK_S = 12.0  # R2 decode >= 12 tok/s == "daily-drivable"

# =====================================================================
# 1. GGUF quant type table
# =====================================================================
# (block_size, bytes_per_block). Source: ggml.h / ggml.c type_traits.
# Verified bits-per-weight in the comment column.
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0:  ("F32",     1,   4),    # 32.00 bpw
    1:  ("F16",     1,   2),    # 16.00
    2:  ("Q4_0",    32,  18),   # 4.50
    3:  ("Q4_1",    32,  20),   # 5.00
    6:  ("Q5_0",    32,  22),   # 5.50
    7:  ("Q5_1",    32,  24),   # 6.00
    8:  ("Q8_0",    32,  34),   # 8.50
    9:  ("Q8_1",    32,  40),   # 10.00
    10: ("Q2_K",    256, 84),   # 2.625
    11: ("Q3_K",    256, 110),  # 3.4375
    12: ("Q4_K",    256, 144),  # 4.50
    13: ("Q5_K",    256, 176),  # 5.50
    14: ("Q6_K",    256, 210),  # 6.5625
    15: ("Q8_K",    256, 292),  # 9.125
    16: ("IQ2_XXS", 256, 66),   # 2.0625
    17: ("IQ2_XS",  256, 74),   # 2.3125
    18: ("IQ3_XXS", 256, 98),   # 3.0625
    19: ("IQ1_S",   256, 50),   # 1.5625
    20: ("IQ4_NL",  32,  18),   # 4.50
    21: ("IQ3_S",   256, 110),  # 3.4375
    22: ("IQ2_S",   256, 82),   # 2.5625
    23: ("IQ4_XS",  256, 136),  # 4.25
    24: ("I8",      1,   1),
    25: ("I16",     1,   2),
    26: ("I32",     1,   4),
    27: ("I64",     1,   8),
    28: ("F64",     1,   8),
    29: ("IQ1_M",   256, 56),   # 1.75
    30: ("BF16",    1,   2),    # 16.00
    34: ("TQ1_0",   256, 54),   # 1.6875
    35: ("TQ2_0",   256, 66),   # 2.0625
    39: ("MXFP4",   32,  17),   # 4.25
}

# GGUF metadata value types
(_GT_U8, _GT_I8, _GT_U16, _GT_I16, _GT_U32, _GT_I32,
 _GT_F32, _GT_BOOL, _GT_STR, _GT_ARR, _GT_U64, _GT_I64, _GT_F64) = range(13)

_GT_FMT = {
    _GT_U8: ("<B", 1), _GT_I8: ("<b", 1), _GT_U16: ("<H", 2), _GT_I16: ("<h", 2),
    _GT_U32: ("<I", 4), _GT_I32: ("<i", 4), _GT_F32: ("<f", 4), _GT_BOOL: ("<?", 1),
    _GT_U64: ("<Q", 8), _GT_I64: ("<q", 8), _GT_F64: ("<d", 8),
}


# =====================================================================
# 2. Model registry -- analytic parameter accounting
# =====================================================================
# Every number below is derived from the upstream Qwen config.json (verified
# 2026-08-19 against huggingface.co/Qwen/<model>/raw/main/config.json) and is
# used for the ANALYTIC bytes/token fallback (Method B). Method A (exact GGUF
# tensor accounting) supersedes it whenever the .gguf file is present.


def _dense_params(*, layers, hidden, n_head, n_kv_head, head_dim, inter, vocab, tied):
    q = hidden * n_head * head_dim
    k = hidden * n_kv_head * head_dim
    v = k
    o = n_head * head_dim * hidden
    attn = q + k + v + o
    ffn = 3 * hidden * inter
    per_layer = attn + ffn
    embd = vocab * hidden
    total = per_layer * layers + embd + (0 if tied else embd)
    # Per decoded token every transformer weight is read once, plus the output
    # projection (which *is* the embedding matrix when tie_word_embeddings).
    # The input embedding is a single-row gather -> negligible, excluded.
    active = per_layer * layers + embd
    return dict(total_params=total, active_params=active, embd_params=embd,
                attn_params=attn * layers, ffn_params=ffn * layers, tied=tied)


def _moe_params(*, layers, hidden, n_head, n_kv_head, head_dim, moe_inter,
                n_expert, n_expert_used, vocab, tied):
    q = hidden * n_head * head_dim
    k = hidden * n_kv_head * head_dim
    v = k
    o = n_head * head_dim * hidden
    attn = q + k + v + o
    router = hidden * n_expert
    per_expert = 3 * hidden * moe_inter
    experts_all = per_expert * n_expert
    experts_act = per_expert * n_expert_used
    embd = vocab * hidden
    total = (attn + router + experts_all) * layers + embd + (0 if tied else embd)
    active = (attn + router + experts_act) * layers + embd
    return dict(total_params=total, active_params=active, embd_params=embd,
                attn_params=attn * layers, router_params=router * layers,
                experts_all_params=experts_all * layers,
                experts_active_params=experts_act * layers,
                n_expert=n_expert, n_expert_used=n_expert_used, tied=tied)


MODELS: dict[str, dict] = {
    "qwen3-0.6b": dict(
        label="Qwen3-0.6B",
        rung="R0",
        repo="unsloth/Qwen3-0.6B-GGUF",
        gguf="Qwen3-0.6B-Q4_K_M.gguf",
        file_bytes=396_705_472,
        sha256="ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a",
        kind="dense",
        arch=_dense_params(layers=28, hidden=1024, n_head=16, n_kv_head=8,
                           head_dim=128, inter=3072, vocab=151936, tied=True),
    ),
    "qwen3-4b": dict(
        label="Qwen3-4B",
        rung="R1",
        repo="unsloth/Qwen3-4B-GGUF",
        gguf="Qwen3-4B-Q4_K_M.gguf",
        file_bytes=2_497_281_312,
        sha256="f6f851777709861056efcdad3af01da38b31223a3ba26e61a4f8bf3a2195813a",
        kind="dense",
        arch=_dense_params(layers=36, hidden=2560, n_head=32, n_kv_head=8,
                           head_dim=128, inter=9728, vocab=151936, tied=True),
    ),
    "qwen3-30b-a3b": dict(
        label="Qwen3-30B-A3B",
        rung="R2",
        repo="unsloth/Qwen3-30B-A3B-GGUF",
        gguf="Qwen3-30B-A3B-Q4_K_M.gguf",
        file_bytes=18_556_686_912,
        sha256="9f1a24700a339b09c06009b729b5c809e0b64c213b8af5b711b3dbdfd0c5ba48",
        kind="moe",
        arch=_moe_params(layers=48, hidden=2048, n_head=32, n_kv_head=4,
                         head_dim=128, moe_inter=768, n_expert=128,
                         n_expert_used=8, vocab=151936, tied=False),
    ),
}

RUNGS: dict[str, dict] = {
    "R0": dict(model="qwen3-0.6b", ngl=99, ncmoe=[None], draft=None,
               lever="machine effective bandwidth (baseline)"),
    "R1": dict(model="qwen3-4b", ngl=99, ncmoe=[None], draft="qwen3-0.6b",
               lever="bpw + dense scaling"),
    # -ncmoe sweep: 48 == every layer's experts in system RAM (always fits);
    # lower values pull expert layers back onto the 8 GB GPU until it OOMs.
    "R2": dict(model="qwen3-30b-a3b", ngl=99, ncmoe=[48, 44, 40, 36, 32],
               draft="qwen3-0.6b", lever="bytes-TOUCHED (MoE, ~3B of 30.5B active)"),
}


# =====================================================================
# 3. bytes-touched-per-token estimation
# =====================================================================
BYTES_METHODOLOGY = """\
BYTES-TOUCHED-PER-TOKEN -- methodology and assumptions
=======================================================
The roofline claim we are testing is  decode_tok_s ~= B_eff / bytes_per_token,
where B_eff is the machine's effective memory bandwidth for the tier of memory
the weights actually live in. So bytes_per_token has to be an honest count of
*weight bytes read from memory to produce one output token*.

METHOD A -- exact GGUF tensor accounting (used whenever the .gguf is present)
----------------------------------------------------------------------------
We parse the GGUF header's tensor table directly (name, shape, ggml type) and
compute each tensor's on-disk byte size from its quant block geometry. Then:

  bytes_per_token =  SUM(non-expert, non-embedding tensors)          x 1.0
                   + SUM(tensors whose name matches '*_exps.*')      x (n_used / n_expert)
                   + (token_embd row)                                ~ 0

  * Non-expert weights (attention Q/K/V/O, norms, the MoE router
    `ffn_gate_inp`, dense FFN, and `output.weight`) are read in full for every
    single token. Counted at 1.0.
  * Routed-expert stacks (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`) are
    3-D tensors holding all N experts. Only `expert_used_count` of them are
    touched per token, so they are scaled by n_used/n_expert. For
    Qwen3-30B-A3B that is 8/128 = 6.25%.
  * `token_embd.weight` is a single-row gather per token (~2 KB), so its
    contribution is rounded to zero -- BUT if the model has
    tie_word_embeddings (Qwen3-0.6B and Qwen3-4B do), llama.cpp reuses that
    same tensor as the output projection, which IS a full read. In that case
    GGUF contains no separate `output.weight` and we count token_embd at 1.0.
    We detect this by the absence of an `output.weight` tensor.

METHOD B -- analytic fallback (used when the .gguf is not on disk)
-----------------------------------------------------------------
Parameter counts are computed from the published config.json, and converted to
bytes with a UNIFORM average bits-per-weight taken from the real file size:

  bytes_per_param = file_size_bytes / total_params
  bytes_per_token = active_params x bytes_per_param

ASSUMPTIONS AND KNOWN BIASES (read these before quoting a number)
-----------------------------------------------------------------
 1. Method B's uniform-bpw assumption is WRONG in detail: llama.cpp's K-quant
    mixes store token_embd / output / attn_v / some ffn_down at Q6_K while the
    bulk sits at Q4_K. For an MoE that under-counts, because the ~90% of the
    file that is expert weight is at the LOW precision -- so the dense/active
    part is at ABOVE-average bpw. Expect Method B to under-estimate MoE
    bytes/token by roughly 10-20%. Method A has no such error; prefer it.
 2. We count WEIGHT traffic only. KV-cache reads are excluded. At batch 1 and
    short context that is small, but it grows linearly with context depth --
    this is exactly why the runbook has an optional `-d 4096` depth sweep: the
    gap between predicted and measured tok/s at depth IS the KV traffic.
 3. Activations, norms scratch and the sampling head's logits (151936 x 4 B =
    ~0.6 MB/token) are excluded. Logits are ~0.03% of R2's weight traffic.
 4. Perfect caching is assumed nowhere, and zero cache reuse is assumed
    nowhere: at batch 1 a weight read is a weight read. On R2 with
    --n-cpu-moe the expert bytes come from DDR5 and the attention bytes from
    GDDR7, so the single "implied bandwidth" number for R2 is a BLEND of two
    tiers and must not be compared to R0/R1's pure-VRAM number. results.md
    labels R2 rows accordingly.
 5. MoE routing is assumed uniform across experts. Real routing is skewed, so
    a hot expert may stay resident in cache/page-cache and cost less than the
    model predicts. That makes the measured tok/s a bit BETTER than predicted,
    not worse.
 6. Speculative decoding does not change bytes/token per *verified* token; it
    changes tokens per weight-read. For +S rows we report the measured
    end-to-end tok/s and the draft acceptance rate; the bytes/token column
    stays the target model's value, and the implied bandwidth for those rows
    is therefore an APPARENT bandwidth that can exceed the hardware's real
    bandwidth. That over-shoot IS the amortization win, quantified.
"""


def _read_gguf_string(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", errors="replace")


def _read_gguf_value(f, vtype: int):
    if vtype in _GT_FMT:
        fmt, size = _GT_FMT[vtype]
        return struct.unpack(fmt, f.read(size))[0]
    if vtype == _GT_STR:
        return _read_gguf_string(f)
    if vtype == _GT_ARR:
        (etype,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        if etype == _GT_STR:
            return [_read_gguf_string(f) for _ in range(n)]
        if etype in _GT_FMT:
            fmt, size = _GT_FMT[etype]
            raw = f.read(size * n)
            return list(struct.unpack("<" + fmt[1] * n, raw))
        raise ValueError(f"unsupported GGUF array element type {etype}")
    raise ValueError(f"unsupported GGUF value type {vtype}")


def read_gguf(path: Path) -> dict:
    """Parse a GGUF header -> {'kv': {...}, 'tensors': [ {name,dims,type,nbytes} ]}."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file (magic={magic!r})")
        version, = struct.unpack("<I", f.read(4))
        if version not in (2, 3):
            raise ValueError(f"{path}: unsupported GGUF version {version}")
        n_tensors, = struct.unpack("<Q", f.read(8))
        n_kv, = struct.unpack("<Q", f.read(8))

        kv = {}
        for _ in range(n_kv):
            key = _read_gguf_string(f)
            vtype, = struct.unpack("<I", f.read(4))
            try:
                kv[key] = _read_gguf_value(f, vtype)
            except ValueError:
                kv[key] = None  # tolerate exotic metadata; we only need a few keys

        tensors = []
        for _ in range(n_tensors):
            name = _read_gguf_string(f)
            n_dims, = struct.unpack("<I", f.read(4))
            dims = list(struct.unpack("<" + "Q" * n_dims, f.read(8 * n_dims)))
            ttype, = struct.unpack("<I", f.read(4))
            _offset, = struct.unpack("<Q", f.read(8))
            if ttype not in GGML_TYPES:
                raise ValueError(f"{path}: unknown ggml type {ttype} on tensor {name}")
            tname, blk, blk_bytes = GGML_TYPES[ttype]
            n_elem = 1
            for d in dims:
                n_elem *= d
            if n_elem % blk:
                raise ValueError(f"{path}: tensor {name} elems {n_elem} not divisible "
                                 f"by block {blk} of {tname}")
            tensors.append(dict(name=name, dims=dims, type=tname,
                                nbytes=(n_elem // blk) * blk_bytes))
    return dict(kv=kv, tensors=tensors, version=version)


def _kv_get(kv: dict, suffix: str, default=None):
    """Fetch an arch-prefixed GGUF key, e.g. 'expert_used_count'."""
    arch = kv.get("general.architecture")
    if arch and f"{arch}.{suffix}" in kv:
        return kv[f"{arch}.{suffix}"]
    for k, v in kv.items():
        if k.endswith("." + suffix):
            return v
    return default


def bytes_per_token_gguf(path: Path) -> dict:
    """METHOD A: exact per-tensor accounting from the GGUF tensor table."""
    g = read_gguf(path)
    kv, tensors = g["kv"], g["tensors"]
    n_expert = int(_kv_get(kv, "expert_count", 0) or 0)
    n_used = int(_kv_get(kv, "expert_used_count", 0) or 0)
    frac = (n_used / n_expert) if n_expert else 0.0

    has_output = any(t["name"] == "output.weight" for t in tensors)

    dense_b = expert_b = embd_b = 0
    for t in tensors:
        nm, nb = t["name"], t["nbytes"]
        if "_exps." in nm:
            expert_b += nb
        elif nm.startswith("token_embd."):
            embd_b += nb
        else:
            dense_b += nb

    # Tied embeddings: no separate output.weight, so token_embd IS the output
    # projection and is read in full every token.
    embd_counted = embd_b if not has_output else 0

    bpt = dense_b + expert_b * frac + embd_counted
    return dict(
        method="gguf-exact",
        bytes_per_token=int(round(bpt)),
        file_bytes=path.stat().st_size,
        detail=dict(
            arch=kv.get("general.architecture"),
            n_expert=n_expert, n_expert_used=n_used, expert_frac=frac,
            dense_bytes=dense_b, expert_bytes_total=expert_b,
            expert_bytes_active=int(round(expert_b * frac)),
            token_embd_bytes=embd_b, token_embd_counted=embd_counted,
            tied_embeddings=not has_output,
            n_tensors=len(tensors),
        ),
    )


def bytes_per_token_analytic(model_key: str) -> dict:
    """METHOD B: analytic param counts x uniform bytes/param from file size."""
    m = MODELS[model_key]
    a = m["arch"]
    bpp = m["file_bytes"] / a["total_params"]
    bpt = a["active_params"] * bpp
    return dict(
        method="analytic",
        bytes_per_token=int(round(bpt)),
        file_bytes=m["file_bytes"],
        detail=dict(
            total_params=a["total_params"],
            active_params=a["active_params"],
            active_frac=a["active_params"] / a["total_params"],
            bytes_per_param=bpp,
            bits_per_weight=bpp * 8,
            **{k: v for k, v in a.items()
               if k not in ("total_params", "active_params")},
        ),
    )


def estimate_bytes_per_token(model_key: str, gguf_path: Path | None) -> dict:
    if gguf_path and gguf_path.is_file():
        try:
            return bytes_per_token_gguf(gguf_path)
        except Exception as e:  # noqa: BLE001 -- never let this kill a rung
            warn(f"GGUF parse failed ({e}); falling back to analytic estimate")
    return bytes_per_token_analytic(model_key)


# =====================================================================
# 4. Machine occupancy sampling (RAM / VRAM)
# =====================================================================
try:
    import psutil  # type: ignore
    HAVE_PSUTIL = True
except Exception:  # noqa: BLE001
    psutil = None  # type: ignore
    HAVE_PSUTIL = False


def _ram_used_total() -> tuple[int, int]:
    """(used_bytes, total_bytes) system-wide."""
    if HAVE_PSUTIL:
        vm = psutil.virtual_memory()
        return vm.total - vm.available, vm.total
    if os.name == "nt":
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = _MS()
        st.dwLength = ctypes.sizeof(_MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return st.ullTotalPhys - st.ullAvailPhys, st.ullTotalPhys
    try:  # POSIX best-effort
        pg = os.sysconf("SC_PAGE_SIZE")
        total = os.sysconf("SC_PHYS_PAGES") * pg
        avail = os.sysconf("SC_AVPHYS_PAGES") * pg
        return total - avail, total
    except Exception:  # noqa: BLE001
        return 0, 0


def _nvidia_smi() -> dict | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.used,memory.total,utilization.gpu,"
                  "temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True).stdout
    except Exception:  # noqa: BLE001
        return None
    line = out.strip().splitlines()[0] if out.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        return None

    def _f(x):
        try:
            return float(x)
        except Exception:  # noqa: BLE001
            return None

    return dict(name=parts[0],
                vram_used_mib=_f(parts[1]), vram_total_mib=_f(parts[2]),
                util_pct=_f(parts[3]) if len(parts) > 3 else None,
                temp_c=_f(parts[4]) if len(parts) > 4 else None,
                power_w=_f(parts[5]) if len(parts) > 5 else None)


class OccupancySampler(threading.Thread):
    """Polls RAM + VRAM in the background and keeps peaks. Never raises."""

    def __init__(self, interval: float = 2.0):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop = threading.Event()
        self.ram_peak = 0
        self.ram_total = 0
        self.vram_peak_mib = 0.0
        self.vram_total_mib = 0.0
        self.gpu_name = None
        self.gpu_temp_max = None
        self.gpu_power_max = None
        self.samples = 0

    def run(self):
        while not self._stop.is_set():
            try:
                used, total = _ram_used_total()
                self.ram_peak = max(self.ram_peak, used)
                self.ram_total = total or self.ram_total
                nv = _nvidia_smi()
                if nv:
                    self.gpu_name = nv["name"]
                    if nv["vram_used_mib"]:
                        self.vram_peak_mib = max(self.vram_peak_mib, nv["vram_used_mib"])
                    if nv["vram_total_mib"]:
                        self.vram_total_mib = nv["vram_total_mib"]
                    if nv.get("temp_c") is not None:
                        self.gpu_temp_max = max(self.gpu_temp_max or 0, nv["temp_c"])
                    if nv.get("power_w") is not None:
                        self.gpu_power_max = max(self.gpu_power_max or 0, nv["power_w"])
                self.samples += 1
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.interval)

    def stop(self) -> dict:
        self._stop.set()
        with contextlib.suppress(Exception):
            self.join(timeout=5)
        return dict(
            ram_peak_bytes=self.ram_peak or None,
            ram_total_bytes=self.ram_total or None,
            ram_peak_gib=round(self.ram_peak / 2**30, 2) if self.ram_peak else None,
            ram_total_gib=round(self.ram_total / 2**30, 2) if self.ram_total else None,
            vram_peak_mib=round(self.vram_peak_mib, 1) or None,
            vram_total_mib=round(self.vram_total_mib, 1) or None,
            gpu_name=self.gpu_name,
            gpu_temp_max_c=self.gpu_temp_max,
            gpu_power_max_w=self.gpu_power_max,
            samples=self.samples,
            source="psutil" if HAVE_PSUTIL else "ctypes/GlobalMemoryStatusEx",
        )


# =====================================================================
# 5. Crash-safe result store
# =====================================================================
def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def info(msg: str) -> None:
    print(f"[ladder] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[ladder] WARNING: {msg}", file=sys.stderr, flush=True)


def append_result(row: dict, path: Path = RESULTS_JSONL) -> None:
    """Append ONE row and fsync it. This is the crash-safety guarantee: after
    this call returns, the measurement survives a hard power loss."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False, sort_keys=False)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
    info(f"appended -> {path.name}: {row.get('rung')} / {row.get('variant')} "
         f"/ {row.get('config_label')}  decode={row.get('decode_tok_s')}")


def load_results(path: Path = RESULTS_JSONL) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A torn final line is exactly what a hard crash mid-write looks
            # like. Skip it rather than losing the whole file.
            warn(f"{path.name}:{i} is not valid JSON (torn write?) -- skipping")
    return rows


# =====================================================================
# 6. Rendering results.md
# =====================================================================
def _fmt(x, nd=2, dash="-"):
    if x is None:
        return dash
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return dash
    return f"{x:,.{nd}f}" if isinstance(x, float) else f"{x:,}"


def _gib(b):
    return None if not b else b / 2**30


def render_results_md(rows: list[dict], path: Path = RESULTS_MD) -> str:
    lines: list[str] = []
    a = lines.append
    a("# The bytes-touched ladder -- results")
    a("")
    a(f"_Generated {_now()} from `results.jsonl` ({len(rows)} measurements)._")
    a("")
    host = rows[-1].get("host") if rows else None
    if host:
        a(f"Host: `{host.get('node')}` / {host.get('platform')} / "
          f"{host.get('cpu_count')} logical CPUs")
        a("")
    a(f"**Win condition: R2 decode >= {WIN_CONDITION_TOK_S:g} tok/s.**")
    a("")

    a("## Ladder")
    a("")
    a("| Rung | Model | Variant | Config | prefill tok/s | decode tok/s | "
      "bytes/tok | implied BW (GB/s) | RAM peak | VRAM peak | Win |")
    a("|---|---|---|---|---:|---:|---:|---:|---:|---:|:--:|")
    for r in rows:
        bpt = r.get("bytes_per_token")
        dec = r.get("decode_tok_s")
        bw = r.get("implied_bandwidth_gb_s")
        ram = _gib(r.get("occupancy", {}).get("ram_peak_bytes"))
        vram = r.get("occupancy", {}).get("vram_peak_mib")
        win = ""
        if r.get("rung") == "R2" and dec is not None:
            win = "YES" if dec >= WIN_CONDITION_TOK_S else "no"
        bw_s = _fmt(bw, 1)
        if bw_s != "-" and r.get("bandwidth_is_blended"):
            bw_s += " *"
        if bw_s != "-" and r.get("variant", "").endswith("spec"):
            bw_s += " +"
        a(f"| {r.get('rung','')} | {r.get('model_label','')} | "
          f"{r.get('variant','')} | {r.get('config_label','')} | "
          f"{_fmt(r.get('prefill_tok_s'), 1)} | {_fmt(dec, 2)} | "
          f"{_fmt(bpt/1e6, 1) if bpt else '-'} MB | {bw_s} | "
          f"{_fmt(ram, 1)} GiB | {_fmt(vram, 0)} MiB | {win} |")
    a("")
    a("`*` blended bandwidth: weights span GDDR7 (attention/KV on GPU) and "
      "DDR5 (routed experts on CPU) -- not comparable to a pure-VRAM row.  ")
    a("`+` apparent bandwidth under speculative decoding: exceeding real "
      "hardware bandwidth here is the amortization win, not an error.")
    a("")

    spec_rows = [r for r in rows if r.get("spec")]
    if spec_rows:
        a("## Speculative decoding")
        a("")
        a("| Rung | Variant | Draft | draft n | accepted | accept rate | "
          "decode tok/s | speedup vs matched base |")
        a("|---|---|---|---:|---:|---:|---:|---:|")
        base_by_rung = {r["rung"]: r.get("decode_tok_s")
                        for r in rows if r.get("variant") == "server-base"}
        for r in spec_rows:
            s = r["spec"]
            base = base_by_rung.get(r.get("rung"))
            sp = (r["decode_tok_s"] / base) if (base and r.get("decode_tok_s")) else None
            a(f"| {r.get('rung')} | {r.get('variant')} | "
              f"{r.get('draft_label') or '-'} | {_fmt(s.get('draft_n'), 0)} | "
              f"{_fmt(s.get('draft_n_accepted'), 0)} | "
              f"{_fmt((s.get('accept_rate') or 0)*100, 1)}% | "
              f"{_fmt(r.get('decode_tok_s'), 2)} | "
              f"{(_fmt(sp, 2) + 'x') if sp else '-'} |")
        a("")

    a("## Roofline read")
    a("")
    a("Decode is memory-bound at batch 1: `tok/s ~= B_eff / bytes_per_token`.")
    a("R0 measures B_eff for pure-VRAM residency. Every later rung's implied")
    a("bandwidth should be read as *how close that configuration gets to the")
    a("memory tier it actually lives in* -- not as a hardware spec.")
    a("")
    a("<details><summary>bytes/token methodology (click)</summary>")
    a("")
    a("```")
    a(BYTES_METHODOLOGY.rstrip())
    a("```")
    a("</details>")
    a("")

    a("## Raw rows")
    a("")
    a("Every row above is one line of `results.jsonl`, written and fsync'd")
    a("before the next measurement started. If the machine hard-crashes, the")
    a("file is complete up to the last finished measurement.")
    a("")

    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return text


# =====================================================================
# 7. llama-bench parsing
# =====================================================================
def parse_llama_bench_json(text: str) -> list[dict]:
    """llama-bench -o json emits a JSON array of test objects."""
    text = text.strip()
    if not text:
        raise ValueError("empty llama-bench output")
    start = text.find("[")
    if start < 0:
        raise ValueError("no JSON array found in llama-bench output")
    data = json.loads(text[start:])
    if not isinstance(data, list):
        raise ValueError("llama-bench JSON root is not an array")
    return data


def _config_key(e: dict) -> tuple:
    return (e.get("model_filename"), e.get("n_gpu_layers"), e.get("n_cpu_moe"),
            e.get("flash_attn"), e.get("n_threads"), e.get("n_depth"),
            e.get("type_k"), e.get("type_v"))


def group_bench_entries(entries: list[dict]) -> list[dict]:
    """Collapse llama-bench's per-test entries into one record per config,
    filling prefill (pp) from n_prompt>0 rows and decode (tg) from n_gen>0."""
    groups: dict[tuple, dict] = {}
    for e in entries:
        k = _config_key(e)
        g = groups.setdefault(k, dict(entries=[], prefill=None, decode=None,
                                      prefill_sd=None, decode_sd=None,
                                      n_prompt=0, n_gen=0, n_depth=e.get("n_depth", 0)))
        g["entries"].append(e)
        np_, ng = e.get("n_prompt", 0), e.get("n_gen", 0)
        ts, sd = e.get("avg_ts"), e.get("stddev_ts")
        if ng and not np_:
            g["decode"], g["decode_sd"], g["n_gen"] = ts, sd, ng
        elif np_ and not ng:
            g["prefill"], g["prefill_sd"], g["n_prompt"] = ts, sd, np_
        else:  # combined pp+tg run: llama-bench reports one blended t/s
            g["combined"], g["n_prompt"], g["n_gen"] = ts, np_, ng
    out = []
    for k, g in groups.items():
        g["key"] = k
        g["meta"] = g["entries"][0]
        out.append(g)
    return out


def bench_group_to_row(g: dict, *, rung: str, model_key: str, variant: str,
                       bpt: dict, occ: dict, cmd: list[str],
                       notes: str = "") -> dict:
    m = MODELS[model_key]
    meta = g["meta"]
    ncmoe = meta.get("n_cpu_moe")
    depth = meta.get("n_depth", 0) or 0
    cfg = []
    if meta.get("n_gpu_layers") is not None:
        cfg.append(f"ngl={meta['n_gpu_layers']}")
    if ncmoe:
        cfg.append(f"ncmoe={ncmoe}")
    if depth:
        cfg.append(f"d={depth}")
    if meta.get("flash_attn") is not None:
        cfg.append(f"fa={meta['flash_attn']}")

    decode = g.get("decode") or g.get("combined")
    b = bpt["bytes_per_token"]
    bw = (b * decode / 1e9) if (decode and b) else None
    blended = bool(ncmoe)

    return dict(
        ts=_now(),
        rung=rung,
        variant=variant,
        model_key=model_key,
        model_label=m["label"],
        gguf=m["gguf"],
        tool="llama-bench",
        config_label=" ".join(cfg) or "default",
        n_gpu_layers=meta.get("n_gpu_layers"),
        n_cpu_moe=ncmoe,
        n_depth=depth,
        flash_attn=meta.get("flash_attn"),
        n_threads=meta.get("n_threads"),
        n_prompt=g.get("n_prompt"),
        n_gen=g.get("n_gen"),
        prefill_tok_s=g.get("prefill"),
        prefill_stddev=g.get("prefill_sd"),
        decode_tok_s=decode,
        decode_stddev=g.get("decode_sd"),
        bytes_per_token=b,
        bytes_method=bpt["method"],
        bytes_detail=bpt.get("detail"),
        implied_bandwidth_gb_s=bw,
        bandwidth_is_blended=blended,
        occupancy=occ,
        build=dict(commit=meta.get("build_commit"), number=meta.get("build_number"),
                   gpu_info=meta.get("gpu_info"), cpu_info=meta.get("cpu_info"),
                   backends=meta.get("backends")),
        model_size_bytes=meta.get("model_size"),
        model_n_params=meta.get("model_n_params"),
        cmd=cmd,
        host=host_info(),
        notes=notes,
        spec=None,
        draft_label=None,
    )


def host_info() -> dict:
    return dict(node=platform.node(), platform=platform.platform(),
                python=platform.python_version(),
                cpu_count=os.cpu_count())


# =====================================================================
# 8. Subcommand: run (llama-bench)
# =====================================================================
def _resolve(bin_dir: Path, name: str, *, required: bool = True) -> Path:
    p = bin_dir / (name + (".exe" if os.name == "nt" else ""))
    if required and not p.is_file():
        raise SystemExit(f"ERROR: {p} not found.\n"
                         f"       Run RUNBOOK.md step 1 first, or pass --bin DIR.\n"
                         f"       (Use --dry-run to preview commands without the "
                         f"binaries present.)")
    return p


def cmd_run(args) -> int:
    rung = args.rung.upper()
    if rung not in RUNGS:
        raise SystemExit(f"unknown rung {rung}; choose from {sorted(RUNGS)}")
    spec = RUNGS[rung]
    model_key = spec["model"]
    m = MODELS[model_key]
    bin_dir, models_dir = Path(args.bin).resolve(), Path(args.models).resolve()
    bench = _resolve(bin_dir, "llama-bench", required=not args.dry_run)
    gguf = models_dir / m["gguf"]
    if not gguf.is_file() and not args.dry_run:
        raise SystemExit(f"ERROR: {gguf} not found. Run RUNBOOK.md step 1.")

    bpt = estimate_bytes_per_token(model_key, gguf)
    info(f"{rung} {m['label']}: bytes/token = {bpt['bytes_per_token']:,} "
         f"({bpt['bytes_per_token']/1e6:.1f} MB) via {bpt['method']}")

    ncmoe_values = ([int(x) for x in args.ncmoe.split(",")]
                    if args.ncmoe else spec["ncmoe"])
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rc = 0

    # One llama-bench invocation per -ncmoe value. Deliberate: a separate
    # process per config means a crash takes out one config, not the sweep,
    # and each config's result is fsync'd before the next process starts.
    for ncm in ncmoe_values:
        cmd = [str(bench), "-m", str(gguf),
               "-p", str(args.n_prompt), "-n", str(args.n_gen),
               "-ngl", str(args.ngl), "-r", str(args.reps),
               "-fa", args.flash_attn, "-o", "json"]
        if args.depth:
            cmd += ["-d", args.depth]
        if ncm is not None:
            cmd += ["-ncmoe", str(ncm)]
        if args.threads:
            cmd += ["-t", str(args.threads)]
        if args.extra:
            cmd += args.extra

        info("run: " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        if args.dry_run:
            continue

        tag = f"{rung}_ncmoe{ncm if ncm is not None else 'na'}_{int(time.time())}"
        sampler = OccupancySampler(interval=args.sample_interval)
        sampler.start()
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=args.timeout)
        except subprocess.TimeoutExpired:
            sampler.stop()
            warn(f"{tag}: TIMEOUT after {args.timeout}s -- skipping this config")
            rc = 1
            continue
        wall = time.time() - t0
        occ = sampler.stop()

        (LOG_DIR / f"{tag}.stdout.json").write_text(proc.stdout or "", encoding="utf-8")
        (LOG_DIR / f"{tag}.stderr.log").write_text(proc.stderr or "", encoding="utf-8")

        if proc.returncode != 0:
            warn(f"{tag}: llama-bench exited {proc.returncode}. "
                 f"Tail of stderr:\n" + "\n".join((proc.stderr or "").splitlines()[-15:]))
            rc = 1
            continue
        try:
            entries = parse_llama_bench_json(proc.stdout)
        except Exception as e:  # noqa: BLE001
            warn(f"{tag}: could not parse llama-bench JSON: {e}")
            rc = 1
            continue

        for g in group_bench_entries(entries):
            row = bench_group_to_row(g, rung=rung, model_key=model_key,
                                     variant="base", bpt=bpt, occ=occ, cmd=cmd,
                                     notes=f"wall={wall:.1f}s log={tag}")
            append_result(row)          # <-- durable BEFORE next config starts
        render_results_md(load_results())

    if not args.dry_run:
        render_results_md(load_results())
        info(f"rendered {RESULTS_MD}")
    return rc


# =====================================================================
# 9. Subcommand: spec (llama-server; llama-bench cannot do speculation)
# =====================================================================
SPEC_PROMPTS = [
    ("chat", "Explain, in about 200 words and for a working software engineer, "
             "why decoding a single token from a large language model on a "
             "laptop is bound by memory bandwidth rather than by FLOPs."),
    ("code", "Write a complete Python function `merge_intervals(intervals)` "
             "that merges overlapping closed intervals given as a list of "
             "(start, end) tuples. Include a docstring, type hints, and three "
             "doctest examples. Then explain the time complexity."),
    ("reason", "A train leaves city A at 09:00 travelling at 80 km/h toward "
               "city B, 400 km away. A second train leaves city B at 10:30 "
               "travelling at 100 km/h toward city A. Work through the problem "
               "step by step and state when and where they meet."),
]


def _wait_health(port: int, timeout: float) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2.0)
    return False


def _completion(port: int, prompt: str, n_predict: int, timeout: float) -> dict:
    body = json.dumps(dict(prompt=prompt, n_predict=n_predict, temperature=0.0,
                           top_k=1, stream=False, cache_prompt=False)).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _server_measure(server_exe: Path, base_args: list[str], *, port: int,
                    label: str, args) -> tuple[dict, dict, list]:
    """Launch llama-server, run the prompt set, tear it down. Returns
    (aggregate timings, occupancy, per-prompt records)."""
    cmd = [str(server_exe), *base_args, "--host", "127.0.0.1", "--port", str(port)]
    info(f"[{label}] " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    if args.dry_run:
        return {}, {}, []

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG_DIR / f"server_{label}_{int(time.time())}.log", "w",
               encoding="utf-8", errors="replace")
    sampler = OccupancySampler(interval=args.sample_interval)
    sampler.start()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    recs: list[dict] = []
    try:
        if not _wait_health(port, args.load_timeout):
            raise RuntimeError(f"[{label}] server did not become healthy in "
                               f"{args.load_timeout}s (see {log.name})")
        for name, prompt in SPEC_PROMPTS:
            r = _completion(port, prompt, args.n_predict, args.req_timeout)
            t = r.get("timings", {}) or {}
            recs.append(dict(prompt=name, timings=t,
                             content_chars=len(r.get("content", ""))))
            info(f"[{label}] {name}: {t.get('predicted_per_second', 0):.2f} tok/s "
                 f"({t.get('predicted_n')} tok)"
                 + (f"  draft {t.get('draft_n_accepted')}/{t.get('draft_n')}"
                    if t.get("draft_n") else ""))
    finally:
        with contextlib.suppress(Exception):
            proc.terminate()
            proc.wait(timeout=30)
        with contextlib.suppress(Exception):
            proc.kill()
        occ = sampler.stop()
        log.close()

    # Aggregate: total tokens / total ms is the honest pooled rate.
    pt = sum(x["timings"].get("predicted_n", 0) or 0 for x in recs)
    pms = sum(x["timings"].get("predicted_ms", 0) or 0 for x in recs)
    ppt = sum(x["timings"].get("prompt_n", 0) or 0 for x in recs)
    ppms = sum(x["timings"].get("prompt_ms", 0) or 0 for x in recs)
    dn = sum(x["timings"].get("draft_n", 0) or 0 for x in recs)
    da = sum(x["timings"].get("draft_n_accepted", 0) or 0 for x in recs)
    agg = dict(decode_tok_s=(pt / pms * 1000.0) if pms else None,
               prefill_tok_s=(ppt / ppms * 1000.0) if ppms else None,
               predicted_n=pt, predicted_ms=pms,
               draft_n=dn or None, draft_n_accepted=da or None,
               accept_rate=(da / dn) if dn else None)
    return agg, occ, recs


def cmd_spec(args) -> int:
    rung = args.rung.upper()
    if rung not in RUNGS:
        raise SystemExit(f"unknown rung {rung}; choose from {sorted(RUNGS)}")
    spec = RUNGS[rung]
    is_ngram = args.spec_type.startswith("ngram")
    if not spec["draft"] and not is_ngram:
        raise SystemExit(
            f"{rung} has no draft model: Qwen3-0.6B IS the R0 target, and there is "
            f"no smaller Qwen3 sharing its tokenizer. Use --spec-type ngram-mod "
            f"instead if you want a draft-free speculation number for R0.")
    model_key, draft_key = spec["model"], spec["draft"]
    m = MODELS[model_key]
    d = MODELS[draft_key] if draft_key else None
    bin_dir, models_dir = Path(args.bin).resolve(), Path(args.models).resolve()
    server = _resolve(bin_dir, "llama-server", required=not args.dry_run)
    gguf = models_dir / m["gguf"]
    dgguf = models_dir / d["gguf"] if d else None
    need = [gguf] if is_ngram else [gguf, dgguf]
    for p in need:
        if not p.is_file() and not args.dry_run:
            raise SystemExit(f"ERROR: {p} not found. Run RUNBOOK.md step 1.")

    bpt = estimate_bytes_per_token(model_key, gguf)
    common = ["-m", str(gguf), "-ngl", str(args.ngl), "-c", str(args.ctx),
              "-fa", args.flash_attn]
    if args.ncmoe is not None:
        common += ["-ncmoe", str(args.ncmoe)]
    if args.threads:
        common += ["-t", str(args.threads)]

    cfg = f"ngl={args.ngl}" + (f" ncmoe={args.ncmoe}" if args.ncmoe else "")
    rc = 0

    runs = []
    if not args.spec_only:
        runs.append(("server-base", list(common), None))
    if is_ngram:
        # ngram speculation needs no draft model; -md would load one anyway
        # and b10502 crashes loading a dense draft under -ncmoe overrides.
        runs.append(("server-spec",
                     common + ["--spec-type", args.spec_type,
                               "--spec-draft-n-max", str(args.draft_n_max),
                               "--spec-draft-n-min", str(args.draft_n_min)],
                     args.spec_type))
    else:
        runs.append(("server-spec",
                     common + ["-md", str(dgguf),
                               "--spec-type", args.spec_type,
                               "--spec-draft-n-max", str(args.draft_n_max),
                               "--spec-draft-n-min", str(args.draft_n_min),
                               "--spec-draft-ngl", str(args.draft_ngl)],
                     d["label"]))

    for variant, sargs, draft_label in runs:
        try:
            agg, occ, recs = _server_measure(server, sargs, port=args.port,
                                             label=f"{rung}_{variant}", args=args)
        except Exception as e:  # noqa: BLE001
            warn(f"{rung} {variant} failed: {e}")
            rc = 1
            continue
        if args.dry_run:
            continue
        b = bpt["bytes_per_token"]
        dec = agg["decode_tok_s"]
        row = dict(
            ts=_now(), rung=rung, variant=variant, model_key=model_key,
            model_label=m["label"], gguf=m["gguf"], tool="llama-server",
            config_label=cfg + (f" {args.spec_type}" if draft_label else ""),
            n_gpu_layers=args.ngl, n_cpu_moe=args.ncmoe, n_depth=0,
            flash_attn=args.flash_attn, n_threads=args.threads,
            n_prompt=None, n_gen=args.n_predict,
            prefill_tok_s=agg["prefill_tok_s"],
            prefill_stddev=None,
            decode_tok_s=dec, decode_stddev=None,
            bytes_per_token=b, bytes_method=bpt["method"],
            bytes_detail=bpt.get("detail"),
            implied_bandwidth_gb_s=(b * dec / 1e9) if (dec and b) else None,
            bandwidth_is_blended=bool(args.ncmoe),
            occupancy=occ, build=None,
            model_size_bytes=m["file_bytes"], model_n_params=None,
            cmd=sargs, host=host_info(),
            notes=f"llama-server pooled over {len(recs)} prompts, "
                  f"n_predict={args.n_predict}, greedy",
            spec=(dict(draft_n=agg["draft_n"],
                       draft_n_accepted=agg["draft_n_accepted"],
                       accept_rate=agg["accept_rate"],
                       spec_type=args.spec_type,
                       draft_gguf=d["gguf"],
                       per_prompt=recs) if draft_label else None),
            draft_label=draft_label,
        )
        append_result(row)              # <-- durable immediately
        render_results_md(load_results())
    return rc


# =====================================================================
# 10. Subcommands: ingest / render / bytes
# =====================================================================
def cmd_ingest(args) -> int:
    model_key = args.model
    if model_key not in MODELS:
        raise SystemExit(f"unknown model {model_key}; choose from {sorted(MODELS)}")
    gguf = Path(args.gguf) if args.gguf else None
    bpt = estimate_bytes_per_token(model_key, gguf)
    text = Path(args.json).read_text(encoding="utf-8")
    entries = parse_llama_bench_json(text)
    occ = args.occupancy and json.loads(Path(args.occupancy).read_text()) or {}
    n = 0
    for g in group_bench_entries(entries):
        row = bench_group_to_row(g, rung=args.rung.upper(), model_key=model_key,
                                 variant=args.variant, bpt=bpt, occ=occ,
                                 cmd=["<ingested>", args.json],
                                 notes=args.notes or "ingested from file")
        append_result(row)
        n += 1
    render_results_md(load_results())
    info(f"ingested {n} configs from {args.json}")
    return 0


def cmd_render(args) -> int:
    rows = load_results(Path(args.jsonl))
    text = render_results_md(rows, Path(args.md))
    info(f"rendered {len(rows)} rows -> {args.md}")
    if args.stdout:
        print(text)
    return 0


def cmd_bytes(args) -> int:
    key = args.model
    if key not in MODELS:
        raise SystemExit(f"unknown model {key}; choose from {sorted(MODELS)}")
    gguf = Path(args.gguf) if args.gguf else None
    est = estimate_bytes_per_token(key, gguf)
    print(json.dumps(est, indent=2))
    b = est["bytes_per_token"]
    print(f"\n{MODELS[key]['label']}: {b:,} bytes/token = {b/1e6:.1f} MB/token "
          f"[{est['method']}]")
    for bw in (60, 120, 240, 480, 672):
        print(f"  at {bw:>4} GB/s effective bandwidth -> {bw*1e9/b:8.1f} tok/s")
    if args.methodology:
        print("\n" + BYTES_METHODOLOGY)
    return 0


# =====================================================================
# 11. Subcommand: selftest (fixtures -- no binaries, no models, no network)
# =====================================================================
def _fixture_bench(*, model_filename, model_type, model_size, model_n_params,
                   ngl, ncmoe, pp_ts, pp_sd, tg_ts, tg_sd, depth=0,
                   n_prompt=512, n_gen=128):
    """Build a realistic llama-bench `-o json` payload.

    Field set and ordering mirror llama-bench's test::get_fields() as of
    build b10502 (verified against tools/llama-bench/llama-bench.cpp).
    """
    base = dict(
        build_commit="a1b2c3d4", build_number=10502,
        cpu_info="Intel(R) Core(TM) Ultra 9 275HX",
        gpu_info="NVIDIA GeForce RTX 5070 Laptop GPU",
        backends="CUDA,RPC",
        model_filename=model_filename, model_type=model_type,
        model_size=model_size, model_n_params=model_n_params,
        n_batch=2048, n_ubatch=512, n_threads=24,
        cpu_mask="0x0", cpu_strict=False, poll=50,
        type_k="f16", type_v="f16",
        n_gpu_layers=ngl, n_cpu_moe=ncmoe, split_mode="layer",
        main_gpu=0, no_kv_offload=False, flash_attn="on",
        devices="CUDA0", tensor_split="100.00",
        tensor_buft_overrides="none", load_mode="mmap",
        embeddings=False, no_op_offload=False, no_host=False,
        fit_target=0, fit_min_ctx=0,
        n_depth=depth,
    )
    out = []
    for n_p, n_g, ts, sd in ((n_prompt, 0, pp_ts, pp_sd), (0, n_gen, tg_ts, tg_sd)):
        e = dict(base)
        e.update(n_prompt=n_p, n_gen=n_g,
                 test_time="2026-08-19T22:41:07Z",
                 avg_ns=int((n_p or n_g) / ts * 1e9), stddev_ns=int(sd * 1e6),
                 avg_ts=ts, stddev_ts=sd,
                 samples_ns=[int((n_p or n_g) / ts * 1e9)] * 3,
                 samples_ts=[ts - sd, ts, ts + sd])
        out.append(e)
    return json.dumps(out, indent=2)


FIXTURES = {
    # R0 -- Qwen3-0.6B fully resident in VRAM. Fast, small, the bandwidth probe.
    "R0": ("qwen3-0.6b", _fixture_bench(
        model_filename="models/Qwen3-0.6B-Q4_K_M.gguf",
        model_type="qwen3 0.6B Q4_K - Medium",
        model_size=396705472, model_n_params=595984384,
        ngl=99, ncmoe=0, pp_ts=8421.33, pp_sd=112.40,
        tg_ts=214.77, tg_sd=1.86)),
    # R1 -- Qwen3-4B fully resident in VRAM (2.5 GB of 8 GB).
    "R1": ("qwen3-4b", _fixture_bench(
        model_filename="models/Qwen3-4B-Q4_K_M.gguf",
        model_type="qwen3 4B Q4_K - Medium",
        model_size=2497281312, model_n_params=4022272000,
        ngl=99, ncmoe=0, pp_ts=2184.05, pp_sd=41.72,
        tg_ts=41.62, tg_sd=0.35)),
    # R2 -- Qwen3-30B-A3B, all 48 layers of routed experts in system RAM.
    "R2-ncmoe48": ("qwen3-30b-a3b", _fixture_bench(
        model_filename="models/Qwen3-30B-A3B-Q4_K_M.gguf",
        model_type="qwen3moe 30B.A3B Q4_K - Medium",
        model_size=18556686912, model_n_params=30532122624,
        ngl=99, ncmoe=48, pp_ts=311.44, pp_sd=6.02,
        tg_ts=11.83, tg_sd=0.19)),
    # R2 -- 12 of 48 expert layers pulled back onto the GPU.
    "R2-ncmoe36": ("qwen3-30b-a3b", _fixture_bench(
        model_filename="models/Qwen3-30B-A3B-Q4_K_M.gguf",
        model_type="qwen3moe 30B.A3B Q4_K - Medium",
        model_size=18556686912, model_n_params=30532122624,
        ngl=99, ncmoe=36, pp_ts=398.10, pp_sd=8.71,
        tg_ts=15.27, tg_sd=0.24)),
}

# A realistic llama-server /completion response with speculative decoding on.
FIXTURE_SERVER_SPEC = {
    "content": "...",
    "stop": True,
    "timings": {
        "prompt_n": 48, "prompt_ms": 210.4,
        "prompt_per_second": 228.14,
        "predicted_n": 256, "predicted_ms": 11_580.0,
        "predicted_per_second": 22.11,
        "draft_n": 321, "draft_n_accepted": 214,
    },
}
FIXTURE_SERVER_BASE = {
    "content": "...",
    "stop": True,
    "timings": {
        "prompt_n": 48, "prompt_ms": 214.8,
        "prompt_per_second": 223.46,
        "predicted_n": 256, "predicted_ms": 16_760.0,
        "predicted_per_second": 15.27,
    },
}


def cmd_selftest(args) -> int:
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="ladder_selftest_"))
    jsonl, md = tmp / "results.jsonl", tmp / "results.md"
    fixt_dir = tmp / "fixtures"
    fixt_dir.mkdir()

    print("=" * 74)
    print("LADDER SELFTEST -- fixtures only, no binaries/models/network needed")
    print("=" * 74)
    print(f"scratch: {tmp}\n")

    # --- 1. bytes/token, analytic path (no GGUF on disk at build time) -------
    print("--- 1. bytes-per-token (analytic, Method B) ---")
    bpts = {}
    for key in ("qwen3-0.6b", "qwen3-4b", "qwen3-30b-a3b"):
        est = bytes_per_token_analytic(key)
        bpts[key] = est
        d = est["detail"]
        print(f"  {MODELS[key]['label']:<16} total={d['total_params']/1e9:6.2f}B "
              f"active={d['active_params']/1e9:6.2f}B "
              f"({d['active_frac']*100:5.2f}%) "
              f"bpw={d['bits_per_weight']:.3f} -> "
              f"{est['bytes_per_token']/1e6:8.1f} MB/token")
    a = bpts["qwen3-30b-a3b"]["detail"]
    assert 0.09 < a["active_frac"] < 0.11, "MoE active fraction sanity"
    assert bpts["qwen3-0.6b"]["detail"]["active_frac"] > 0.99, "dense reads ~everything"
    print("  OK: MoE active fraction ~10%, dense ~100%\n")

    # --- 2. GGUF parser round-trip on a synthetic MoE GGUF -------------------
    print("--- 2. GGUF tensor-table parser (Method A) on a synthetic MoE file ---")
    synth = fixt_dir / "synthetic-moe-Q4_K_M.gguf"
    _write_synthetic_gguf(synth)
    est_a = bytes_per_token_gguf(synth)
    d = est_a["detail"]
    print(f"  arch={d['arch']} experts={d['n_expert']} used={d['n_expert_used']} "
          f"frac={d['expert_frac']:.4f}")
    print(f"  dense={d['dense_bytes']:,}B  experts_total={d['expert_bytes_total']:,}B"
          f"  experts_active={d['expert_bytes_active']:,}B")
    print(f"  -> bytes/token = {est_a['bytes_per_token']:,}")
    expect = d["dense_bytes"] + round(d["expert_bytes_total"] * d["expert_frac"])
    assert abs(est_a["bytes_per_token"] - expect) <= 1, "Method A arithmetic"
    assert est_a["bytes_per_token"] < d["dense_bytes"] + d["expert_bytes_total"]
    print("  OK: exact accounting scales expert stacks by n_used/n_expert\n")

    # --- 3. parse llama-bench fixtures + crash-safe append -------------------
    print("--- 3. llama-bench parse -> results.jsonl (append + fsync per config) ---")
    for name, (model_key, payload) in FIXTURES.items():
        p = fixt_dir / f"llama-bench_{name}.json"
        p.write_text(payload, encoding="utf-8")
        entries = parse_llama_bench_json(payload)
        groups = group_bench_entries(entries)
        assert len(groups) == 1, f"{name}: expected 1 config group, got {len(groups)}"
        g = groups[0]
        assert g["prefill"] and g["decode"], f"{name}: missing pp or tg"
        rung = name.split("-")[0]
        occ = dict(ram_peak_bytes=int(23.4 * 2**30) if rung == "R2"
                   else int(6.1 * 2**30),
                   ram_total_bytes=int(31.5 * 2**30),
                   vram_peak_mib=6412.0 if rung == "R2" else 3180.0,
                   vram_total_mib=8188.0, gpu_name="NVIDIA GeForce RTX 5070 Laptop GPU",
                   source="fixture")
        row = bench_group_to_row(g, rung=rung, model_key=model_key, variant="base",
                                 bpt=bpts[model_key], occ=occ,
                                 cmd=["llama-bench", "-m", MODELS[model_key]["gguf"]],
                                 notes=f"selftest fixture {name}")
        before = jsonl.stat().st_size if jsonl.exists() else 0
        append_result(row, jsonl)
        after = jsonl.stat().st_size
        assert after > before, "append must grow the file"
        print(f"  {name:<12} pp={g['prefill']:>9.2f}  tg={g['decode']:>8.2f} "
              f"-> file {before:>5}B -> {after:>5}B")
    print("  OK: every config durably on disk before the next one\n")

    # --- 4. llama-server speculative parse ----------------------------------
    print("--- 4. llama-server speculative timings ---")
    for variant, fx, draft in (("server-base", FIXTURE_SERVER_BASE, None),
                               ("server-spec", FIXTURE_SERVER_SPEC, "Qwen3-0.6B")):
        (fixt_dir / f"llama-server_{variant}.json").write_text(
            json.dumps(fx, indent=2), encoding="utf-8")
        t = fx["timings"]
        dn, da = t.get("draft_n"), t.get("draft_n_accepted")
        acc = (da / dn) if dn else None
        dec = t["predicted_n"] / t["predicted_ms"] * 1000.0
        b = bpts["qwen3-30b-a3b"]["bytes_per_token"]
        row = dict(
            ts=_now(), rung="R2", variant=variant, model_key="qwen3-30b-a3b",
            model_label="Qwen3-30B-A3B", gguf=MODELS["qwen3-30b-a3b"]["gguf"],
            tool="llama-server",
            config_label="ngl=99 ncmoe=36" + (" draft-simple" if draft else ""),
            n_gpu_layers=99, n_cpu_moe=36, n_depth=0, flash_attn="on",
            n_threads=24, n_prompt=t["prompt_n"], n_gen=t["predicted_n"],
            prefill_tok_s=t["prompt_n"] / t["prompt_ms"] * 1000.0,
            prefill_stddev=None, decode_tok_s=dec, decode_stddev=None,
            bytes_per_token=b, bytes_method="analytic",
            bytes_detail=bpts["qwen3-30b-a3b"]["detail"],
            implied_bandwidth_gb_s=b * dec / 1e9, bandwidth_is_blended=True,
            occupancy=dict(ram_peak_bytes=int(23.9 * 2**30),
                           ram_total_bytes=int(31.5 * 2**30),
                           vram_peak_mib=7010.0, vram_total_mib=8188.0,
                           source="fixture"),
            build=None, model_size_bytes=MODELS["qwen3-30b-a3b"]["file_bytes"],
            model_n_params=None, cmd=["llama-server", "..."], host=host_info(),
            notes="selftest fixture",
            spec=(dict(draft_n=dn, draft_n_accepted=da, accept_rate=acc,
                       spec_type="draft-simple",
                       draft_gguf="Qwen3-0.6B-Q4_K_M.gguf", per_prompt=[])
                  if draft else None),
            draft_label=draft)
        append_result(row, jsonl)
        print(f"  {variant:<12} decode={dec:6.2f} tok/s" +
              (f"  accept={acc*100:.1f}% ({da}/{dn})" if acc else ""))
    print("  OK\n")

    # --- 5. torn-line tolerance (simulated hard crash mid-write) ------------
    print("--- 5. crash tolerance: torn final line ---")
    n_ok = len(load_results(jsonl))
    with open(jsonl, "a", encoding="utf-8") as f:
        f.write('{"rung": "R2", "variant": "base", "decode_tok')  # power cut here
    n_after = len(load_results(jsonl))
    assert n_after == n_ok, "torn line must be skipped, earlier rows preserved"
    print(f"  {n_ok} rows before, {n_after} rows after a torn append -- "
          f"earlier measurements intact.  OK\n")

    # --- 6. render ----------------------------------------------------------
    print("--- 6. rendered results.md ---\n")
    rows = load_results(jsonl)
    text = render_results_md(rows, md)
    table = text.split("## Ladder")[1].split("## Speculative")[0]
    print(table.strip())
    print()
    print("--- speculative section ---\n")
    print(text.split("## Speculative decoding")[1].split("## Roofline")[0].strip())
    print()

    assert "| R0 |" in text and "| R1 |" in text and "| R2 |" in text
    assert "YES" in text or "no" in text, "win column must render"
    print("=" * 74)
    print(f"SELFTEST PASSED -- {len(rows)} rows parsed, appended, and rendered.")
    print(f"Fixtures written to: {fixt_dir}")
    print("=" * 74)

    if args.keep:
        info(f"kept scratch dir {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


def _write_synthetic_gguf(path: Path) -> None:
    """Write a tiny but structurally valid GGUF (header + tensor table only) for
    a 2-layer MoE, so the Method-A parser can be exercised without a real model.
    Tensor DATA is not written -- read_gguf() only reads the header."""
    def s(x: str) -> bytes:
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    def kv_str(k, v):
        return s(k) + struct.pack("<I", _GT_STR) + s(v)

    def kv_u32(k, v):
        return s(k) + struct.pack("<I", _GT_U32) + struct.pack("<I", v)

    kvs = (kv_str("general.architecture", "qwen3moe")
           + kv_u32("qwen3moe.block_count", 2)
           + kv_u32("qwen3moe.expert_count", 128)
           + kv_u32("qwen3moe.expert_used_count", 8)
           + kv_u32("qwen3moe.embedding_length", 2048))
    n_kv = 5

    # (name, dims, ggml_type). Q4_K=12, Q6_K=14, F32=0.
    tensors = [("token_embd.weight", [2048, 151936], 14),
               ("output.weight", [2048, 151936], 14),
               ("output_norm.weight", [2048], 0)]
    for i in range(2):
        tensors += [
            (f"blk.{i}.attn_norm.weight", [2048], 0),
            (f"blk.{i}.attn_q.weight", [2048, 4096], 12),
            (f"blk.{i}.attn_k.weight", [2048, 512], 12),
            (f"blk.{i}.attn_v.weight", [2048, 512], 14),
            (f"blk.{i}.attn_output.weight", [4096, 2048], 12),
            (f"blk.{i}.ffn_gate_inp.weight", [2048, 128], 0),
            (f"blk.{i}.ffn_gate_exps.weight", [2048, 768, 128], 12),
            (f"blk.{i}.ffn_up_exps.weight", [2048, 768, 128], 12),
            (f"blk.{i}.ffn_down_exps.weight", [768, 2048, 128], 12),
            (f"blk.{i}.ffn_norm.weight", [2048], 0),
        ]

    body = b""
    off = 0
    for name, dims, tt in tensors:
        body += s(name) + struct.pack("<I", len(dims))
        body += struct.pack("<" + "Q" * len(dims), *dims)
        body += struct.pack("<I", tt) + struct.pack("<Q", off)
        off += 32  # dummy, unused by the parser

    path.write_bytes(b"GGUF" + struct.pack("<I", 3)
                     + struct.pack("<Q", len(tensors))
                     + struct.pack("<Q", n_kv) + kvs + body)


# =====================================================================
# 12. CLI
# =====================================================================
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="ladder_bench.py",
        description="The bytes-touched ladder: run a rung, estimate bytes/token, "
                    "append crash-safely to results.jsonl, render results.md.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Start with:  python ladder_bench.py selftest")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common_paths(sp):
        sp.add_argument("--bin", default=str(HERE / "llama.cpp"),
                        help="dir containing llama-bench/llama-server "
                             "(default: ladder/llama.cpp)")
        sp.add_argument("--models", default=str(HERE / "models"),
                        help="dir containing the .gguf files "
                             "(default: ladder/models)")
        sp.add_argument("--threads", type=int, default=None,
                        help="-t threads (default: llama.cpp's choice)")
        sp.add_argument("--flash-attn", default="on", choices=["on", "off", "auto"])
        sp.add_argument("--sample-interval", type=float, default=2.0)
        sp.add_argument("--dry-run", action="store_true",
                        help="print the commands, run nothing")

    sp = sub.add_parser("selftest", help="fixtures end-to-end; no binaries needed")
    sp.add_argument("--keep", action="store_true", help="keep the scratch dir")
    sp.set_defaults(func=cmd_selftest)

    sp = sub.add_parser("run", help="run a rung with llama-bench")
    sp.add_argument("--rung", required=True, choices=sorted(RUNGS))
    common_paths(sp)
    sp.add_argument("--ngl", type=int, default=99)
    sp.add_argument("--ncmoe", default=None,
                    help="comma list overriding the rung default sweep, "
                         "e.g. 48,40,32 (one llama-bench process per value)")
    sp.add_argument("--n-prompt", type=int, default=512)
    sp.add_argument("--n-gen", type=int, default=128)
    sp.add_argument("--depth", default=None, help="-d value(s), e.g. 0,4096")
    sp.add_argument("--reps", type=int, default=3)
    sp.add_argument("--timeout", type=float, default=3600.0)
    sp.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="everything after this is passed straight to llama-bench")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("spec", help="speculative variant via llama-server")
    sp.add_argument("--rung", required=True, choices=sorted(RUNGS))
    common_paths(sp)
    sp.add_argument("--ngl", type=int, default=99)
    sp.add_argument("--ncmoe", type=int, default=None)
    sp.add_argument("--ctx", type=int, default=4096)
    sp.add_argument("--port", type=int, default=8099)
    sp.add_argument("--n-predict", type=int, default=256)
    sp.add_argument("--spec-type", default="draft-simple",
                    choices=["draft-simple", "ngram-mod", "ngram-simple",
                             "ngram-cache", "ngram-map-k", "ngram-map-k4v"])
    sp.add_argument("--draft-n-max", type=int, default=8)
    sp.add_argument("--draft-n-min", type=int, default=0)
    sp.add_argument("--draft-ngl", type=int, default=99)
    sp.add_argument("--load-timeout", type=float, default=900.0,
                    help="seconds to wait for /health (R2 mmaps 18.6 GB)")
    sp.add_argument("--req-timeout", type=float, default=900.0)
    sp.add_argument("--spec-only", action="store_true",
                    help="skip the matched no-speculation baseline")
    sp.set_defaults(func=cmd_spec)

    sp = sub.add_parser("ingest", help="parse a saved llama-bench JSON file")
    sp.add_argument("--json", required=True)
    sp.add_argument("--rung", required=True)
    sp.add_argument("--model", required=True, choices=sorted(MODELS))
    sp.add_argument("--variant", default="base")
    sp.add_argument("--gguf", default=None, help="path to the .gguf for Method A")
    sp.add_argument("--occupancy", default=None, help="JSON file of RAM/VRAM peaks")
    sp.add_argument("--notes", default=None)
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("render", help="re-render results.md from results.jsonl")
    sp.add_argument("--jsonl", default=str(RESULTS_JSONL))
    sp.add_argument("--md", default=str(RESULTS_MD))
    sp.add_argument("--stdout", action="store_true")
    sp.set_defaults(func=cmd_render)

    sp = sub.add_parser("bytes", help="show the bytes/token estimate and the math")
    sp.add_argument("--model", required=True, choices=sorted(MODELS))
    sp.add_argument("--gguf", default=None)
    sp.add_argument("--methodology", action="store_true")
    sp.set_defaults(func=cmd_bytes)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
