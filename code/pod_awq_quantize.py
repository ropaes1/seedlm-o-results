"""Produce an AWQ W4A16 group-128 checkpoint for the Phase 3 comparator set.

This is the one comparator that is not a GGUF: `llmcompressor` runs AWQ's
activation-aware scaling search and writes a `compressed-tensors` checkpoint,
which :class:`comparators.AWQDequantizer` unpacks back to dense bf16 so it can
be swapped into the same torch model as everything else.

    python pod_awq_quantize.py --model-dir models/originals/Qwen3-8B \
        --out-dir comparators/Qwen3-8B/awq --group-size 128

**Pod-tested only.**  `llmcompressor` is deliberately not installed on the
laptop (an 8B AWQ pass is ~1 h of GPU time); `run_phase3_pod.sh` installs it and
calls this script.  Everything here is import-guarded so the failure mode on a
machine without the dependency is one clear sentence, not a stack trace.

Calibration corpus: the same wikitext split that calibrates the GGUF imatrix, so
both real comparators see the same data and the head-to-head is not an artefact
of one of them getting a friendlier calibration set.  The choice is
recorded in ``awq_provenance.json`` next to the checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_CALIB_SAMPLES: int = 256
DEFAULT_MAX_SEQ_LEN: int = 2048


def load_calibration(tokenizer, n_samples: int, max_seq_len: int,
                     corpus_path: str | None) -> list:
    """Build the AWQ calibration set.

    Args:
        tokenizer: HF tokenizer for the target model.
        n_samples: number of calibration sequences.
        max_seq_len: tokens per sequence.
        corpus_path: raw text file (the wikitext split the imatrix also uses);
            when ``None`` the HF ``wikitext`` dataset is pulled instead.

    Returns:
        A list of ``{"input_ids": [...]}`` dicts, the shape ``oneshot`` accepts.
    """
    if corpus_path and Path(corpus_path).exists():
        text = Path(corpus_path).read_text(encoding="utf-8", errors="ignore")
    else:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
    ids = tokenizer(text, return_tensors=None)["input_ids"]
    chunks = []
    for i in range(0, len(ids) - max_seq_len, max_seq_len):
        chunks.append({"input_ids": ids[i:i + max_seq_len]})
        if len(chunks) >= n_samples:
            break
    if not chunks:
        raise RuntimeError("calibration corpus too short for one full sequence")
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--calib-samples", type=int, default=DEFAULT_CALIB_SAMPLES)
    ap.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    ap.add_argument("--corpus", default="/workspace/wikitext-2-raw/wiki.test.raw",
                    help="raw text file; falls back to the HF wikitext dataset")
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from llmcompressor import oneshot
        from llmcompressor.modifiers.awq import AWQModifier
    except ImportError as exc:            # pragma: no cover - pod only
        raise SystemExit(
            "pod_awq_quantize.py needs llmcompressor + transformers: "
            "pip install llmcompressor  (pod only; run_phase3_pod.sh installs "
            f"it) [{exc}]") from exc

    out = Path(args.out_dir)
    if out.exists() and any(out.glob("*.safetensors")):
        print(f"SKIP {out} already holds an AWQ checkpoint")
        return

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map="auto")
    calib = load_calibration(tok, args.calib_samples, args.max_seq_len,
                             args.corpus)
    recipe = AWQModifier(targets=["Linear"], scheme="W4A16",
                         ignore=["lm_head"])
    # `group_size` lives on the scheme in recent llmcompressor releases and on
    # the modifier in older ones; set whichever attribute exists so the pod is
    # not hostage to one version.
    for holder in (recipe, getattr(recipe, "config_groups", None)):
        if holder is not None and hasattr(holder, "group_size"):
            holder.group_size = args.group_size

    oneshot(model=model, dataset=calib, recipe=recipe,
            max_seq_length=args.max_seq_len,
            num_calibration_samples=len(calib),
            output_dir=str(out))
    tok.save_pretrained(out)

    (out / "awq_provenance.json").write_text(json.dumps({
        "scheme": "W4A16",
        "group_size": args.group_size,
        "ignored": ["lm_head"],
        "calibration": (args.corpus if Path(args.corpus).exists()
                        else "hf:wikitext/wikitext-2-raw-v1:test"),
        "calibration_sequences": len(calib),
        "max_seq_len": args.max_seq_len,
        "source_model": str(args.model_dir),
    }, indent=2), encoding="utf-8")
    print(f"AWQ checkpoint written to {out}")


if __name__ == "__main__":
    main()
