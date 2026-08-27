# The bytes-touched ladder -- results

_Generated 2026-08-22T10:25:47-04:00 from `results.jsonl` (14 measurements)._

Host: `test-laptop` / Windows-11-10.0.26200-SP0 / 24 logical CPUs

**Win condition: R2 decode >= 12 tok/s.**

## Ladder

| Rung | Model | Variant | Config | prefill tok/s | decode tok/s | bytes/tok | implied BW (GB/s) | RAM peak | VRAM peak | Win |
|---|---|---|---|---:|---:|---:|---:|---:|---:|:--:|
| R0 | Qwen3-0.6B | base | ngl=99 fa=1 | 19,005.5 | 349.03 | 390.8 MB | 136.4 | 23.6 GiB | 608 MiB |  |
| R1 | Qwen3-4B | base | ngl=99 fa=1 | 4,114.4 | 84.95 | 2,491.3 MB | 211.6 | 24.9 GiB | 2,622 MiB |  |
| R1 | Qwen3-4B | server-base | ngl=99 | 893.3 | 77.38 | 2,491.3 MB | 192.8 | 26.1 GiB | 3,164 MiB |  |
| R1 | Qwen3-4B | server-spec | ngl=99 draft-simple | 785.3 | 67.52 | 2,491.3 MB | 168.2 + | 27.4 GiB | 4,026 MiB |  |
| R1 | Qwen3-4B | server-spec | ngl=99 draft-simple | 822.1 | 55.08 | 2,491.3 MB | 137.2 + | 28.9 GiB | 4,026 MiB |  |
| R2 | Qwen3-30B-A3B | base | ngl=99 ncmoe=48 fa=1 | 348.7 | 38.21 | 1,919.6 MB | 73.3 * | 27.5 GiB | 1,386 MiB | YES |
| R2 | Qwen3-30B-A3B | base | ngl=99 ncmoe=44 fa=1 | 395.3 | 42.99 | 1,919.6 MB | 82.5 * | 27.8 GiB | 2,880 MiB | YES |
| R2 | Qwen3-30B-A3B | base | ngl=99 ncmoe=40 fa=1 | 470.1 | 45.97 | 1,919.6 MB | 88.2 * | 28.0 GiB | 4,324 MiB | YES |
| R2 | Qwen3-30B-A3B | base | ngl=99 ncmoe=36 fa=1 | 488.6 | 48.21 | 1,919.6 MB | 92.6 * | 28.3 GiB | 5,666 MiB | YES |
| R2 | Qwen3-30B-A3B | base | ngl=99 ncmoe=32 fa=1 | 544.9 | 52.47 | 1,919.6 MB | 100.7 * | 28.0 GiB | 7,060 MiB | YES |
| R2 | Qwen3-30B-A3B | base | ngl=99 ncmoe=30 fa=1 | 578.0 | 54.19 | 1,919.6 MB | 104.0 * | 28.1 GiB | 7,708 MiB | YES |
| R2 | Qwen3-30B-A3B | base | ngl=99 ncmoe=28 fa=1 | 113.5 | 10.50 | 1,919.6 MB | 20.2 * | 28.7 GiB | 7,874 MiB | no |
| R2 | Qwen3-30B-A3B | server-base | ngl=99 ncmoe=32 | 46.0 | 48.77 | 1,919.6 MB | 93.6 * | 29.3 GiB | 7,272 MiB | YES |
| R2 | Qwen3-30B-A3B | server-spec | ngl=99 ncmoe=32 ngram-mod | 45.9 | 48.10 | 1,919.6 MB | 92.3 * + | 29.6 GiB | 7,272 MiB | YES |

`*` blended bandwidth: weights span GDDR7 (attention/KV on GPU) and DDR5 (routed experts on CPU) -- not comparable to a pure-VRAM row.  
`+` apparent bandwidth under speculative decoding: exceeding real hardware bandwidth here is the amortization win, not an error.

## Speculative decoding

| Rung | Variant | Draft | draft n | accepted | accept rate | decode tok/s | speedup vs matched base |
|---|---|---|---:|---:|---:|---:|---:|
| R1 | server-spec | Qwen3-0.6B | 1,275 | 604 | 47.4% | 67.52 | 0.87x |
| R1 | server-spec | Qwen3-0.6B | 998 | 513 | 51.4% | 55.08 | 0.71x |
| R2 | server-spec | ngram-mod | - | - | 0% | 48.10 | 0.99x |

## Roofline read

Decode is memory-bound at batch 1: `tok/s ~= B_eff / bytes_per_token`.
R0 measures B_eff for pure-VRAM residency. Every later rung's implied
bandwidth should be read as *how close that configuration gets to the
memory tier it actually lives in* -- not as a hardware spec.

<details><summary>bytes/token methodology (click)</summary>

```
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
```
</details>

## Raw rows

Every row above is one line of `results.jsonl`, written and fsync'd
before the next measurement started. If the machine hard-crashes, the
file is complete up to the last finished measurement.

