# RUNBOOK: Q4_K_M entropy audit

Two steps. Both CPU-only, no GPU touched. Requires Python 3.10+ with
`numpy`, `gguf`, and (for the download helper) `huggingface_hub`.

## 1. Download the subject model (~1.03 GiB)

Verified against the HF API on 2026-08-19: the repo `unsloth/Qwen3-1.7B-GGUF`
publishes the Q4_K_M weights as **`Qwen3-1.7B-Q4_K_M.gguf`** (no shards, no
`-00001-of-0000N` suffix). Downloaded size: **1,107,409,472 bytes**.

Run from this `entropy-audit/` directory:

```bash
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='unsloth/Qwen3-1.7B-GGUF', filename='Qwen3-1.7B-Q4_K_M.gguf', local_dir='.'))"
```

Equivalent one-liner, if you'd rather use curl:

```bash
curl -L -o Qwen3-1.7B-Q4_K_M.gguf https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf
```

(`fetch_model.py` in this folder is the same call wrapped in a script.)

## 2. Run the audit

```bash
python entropy_audit.py Qwen3-1.7B-Q4_K_M.gguf
```

Writes `results-Qwen3-1.7B-Q4_K_M.md` and prints the same report.
**Measured wall time: 28.6 s** on a laptop CPU (budget was 10 min).
Add `--quiet` to drop the per-tensor progress lines.

---

## Already done

Both steps above were executed on 2026-08-19; the committed
[`results-Qwen3-1.7B-Q4_K_M.md`](results-Qwen3-1.7B-Q4_K_M.md) is real output
from the real file, not a placeholder. The commands are kept here for
reproducibility.

## Verification steps (no download needed)

```bash
# Parser vs the gguf package's own Q4_K dequantizer: must agree exactly.
python entropy_audit.py --self-test

# Build the synthetic GGUF and assert the analytically known entropies.
python make_test_gguf.py --smoke-test
```

The smoke test builds `test-q4k.gguf` with index/scale distributions chosen up
front, so the expected entropies are exact rather than approximate. It checks
the exact ones to 1e-9 and the sampling-limited ones to 0.01 bits. Runtime ~2 s.

## Notes

- `entropy_audit.py` runs the parser self-test automatically before it measures
  anything, so a layout misread can never quietly become a number.
- Memory is flat: the GGUF is memory-mapped and walked 16384 super-blocks at a
  time, so a 1 GiB file and a 1 MiB file have the same working set.
- Non-Q4_K tensors (Q6_K / F32) are never modelled. They are counted at full
  stored size in every total and itemised in the report's "Unmodelled bytes"
  table, so the whole-model figure is a floor.
