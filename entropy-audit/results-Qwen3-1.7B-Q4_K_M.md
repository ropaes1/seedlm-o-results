# Q4_K_M entropy audit -- `Qwen3-1.7B-Q4_K_M`

- file: `Qwen3-1.7B-Q4_K_M.gguf`
- file bytes: 1,107,409,472 (1056.1 MiB); tensor bytes: 1,101,457,408
- tensors: 310 (168 Q4_K, 142 other)
- weights: 1,720,574,976 (70.0% in Q4_K super-blocks)
- audit wall time: 28.6 s (CPU only)

## Headline

**recoverable: 0.142 bpw of 5.121 stored (2.8% smaller, ~2.8% decode speedup if bandwidth-bound)**

- stored             : 5.1213 bpw  (1,050.4 MiB)
- order-0 model      : 4.9844 bpw  (1,022.3 MiB, 2.7% smaller)
- order-1 model      : 4.9798 bpw  (1,021.4 MiB, 2.8% smaller)
- order-1, per-tensor: 4.9793 bpw  (1,021.3 MiB, 2.8% smaller)

Q4_K tensors only (61.5% of tensor bytes, the part this audit actually models):

- stored        : 4.5000 bpw (= 4.5 exactly)
- order-0 model : 4.3042 bpw (4.35% smaller)
- order-1 model : 4.2977 bpw (4.50% smaller)
- **recoverable within Q4_K: 0.202 bpw of 4.500**

Index stream alone (the 4.0-bit headline number):

- H0 = **3.8602** bits/index vs 4.0 stored (3.49% slack)
- H1 = **3.8536** bits/index vs 4.0 stored (3.66% slack)
- order-1 gain over order-0: 0.0066 bits/index -- i.e. how much adjacency context is worth.
- 6-bit sub-block scales: H0 = 4.9632 vs 6.0 stored (17.3% slack)
- 6-bit sub-block mins  : H0 = 5.2449 vs 6.0 stored (12.6% slack)
- fp16 d/dmin (32 bits/super-block) charged at full price -- not modelled, so every number above is a floor.

### Model-wide 4-bit index distribution

| index |     0 |     1 |     2 |     3 |     4 |     5 |     6 |     7 |     8 |     9 |    10 |    11 |    12 |    13 |    14 |    15 |
|-------|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| p (%) |  4.16 |  2.59 |  3.77 |  5.22 |  6.85 |  8.44 |  9.72 | 10.41 | 10.31 |  9.46 |  8.07 |  6.45 |  4.85 |  3.46 |  3.00 |  3.24 |

Uniform would be 6.25% everywhere and exactly 4.0 bits; the departure from flat is the entire source of the index slack.

## Per-family

bpw columns are over *all* weights of the family, mixed quant types included; `q4k%` is the share of the family's bytes that were actually modelled (the rest is charged at full stored size).

| family           | qtypes      | tens |  Mweight | stored |  H0bpw |  H1bpw | recov  | %bytes | q4k% |  H0idx |  H1idx | H0sc  | H0min |
|------------------|-------------|-----:|---------:|-------:|-------:|-------:|-------:|-------:|-----:|-------:|-------:|------:|------:|
| token_embd       | Q6_K        |    1 |   311.16 |  6.562 |  6.562 |  6.562 |  0.000 |  23.17 |   0.0 |  0.000 |  0.000 | 0.000 | 0.000 |
| ffn_down         | Q4_K,Q6_K   |   28 |   352.32 |  5.531 |  5.432 |  5.428 |  0.103 |  22.12 |  40.7 |  3.856 |  3.847 | 5.012 | 5.282 |
| ffn_gate         | Q4_K        |   28 |   352.32 |  4.500 |  4.307 |  4.301 |  0.199 |  17.99 | 100.0 |  3.864 |  3.857 | 4.946 | 5.238 |
| ffn_up           | Q4_K        |   28 |   352.32 |  4.500 |  4.302 |  4.296 |  0.204 |  17.99 | 100.0 |  3.860 |  3.854 | 4.923 | 5.219 |
| attn_output      | Q4_K        |   28 |   117.44 |  4.500 |  4.310 |  4.305 |  0.195 |   6.00 | 100.0 |  3.864 |  3.859 | 5.016 | 5.246 |
| attn_q           | Q4_K        |   28 |   117.44 |  4.500 |  4.302 |  4.295 |  0.205 |   6.00 | 100.0 |  3.857 |  3.850 | 4.980 | 5.261 |
| attn_v           | Q4_K,Q6_K   |   28 |    58.72 |  5.531 |  5.429 |  5.425 |  0.107 |   3.69 |  40.7 |  3.850 |  3.842 | 4.978 | 5.266 |
| attn_k           | Q4_K        |   28 |    58.72 |  4.500 |  4.304 |  4.297 |  0.203 |   3.00 | 100.0 |  3.860 |  3.853 | 4.969 | 5.253 |
| attn_norm        | F32         |   28 |     0.06 | 32.000 | 32.000 | 32.000 |  0.000 |   0.02 |   0.0 |  0.000 |  0.000 | 0.000 | 0.000 |
| ffn_norm         | F32         |   28 |     0.06 | 32.000 | 32.000 | 32.000 |  0.000 |   0.02 |   0.0 |  0.000 |  0.000 | 0.000 | 0.000 |
| attn_k_norm      | F32         |   28 |     0.00 | 32.000 | 32.000 | 32.000 |  0.000 |   0.00 |   0.0 |  0.000 |  0.000 | 0.000 | 0.000 |
| attn_q_norm      | F32         |   28 |     0.00 | 32.000 | 32.000 | 32.000 |  0.000 |   0.00 |   0.0 |  0.000 |  0.000 | 0.000 | 0.000 |
| output_norm      | F32         |    1 |     0.00 | 32.000 | 32.000 | 32.000 |  0.000 |   0.00 |   0.0 |  0.000 |  0.000 | 0.000 | 0.000 |
| TOTAL            | F32,Q4_K,Q6_K |  310 |  1720.57 |  5.121 |  4.984 |  4.980 |  0.142 | 100.00 |  61.5 |  3.860 |  3.854 | 4.963 | 5.245 |

Columns: `stored`/`H0bpw`/`H1bpw`/`recov` are whole-tensor bits per weight (indices + 6-bit scales + fp16 d/dmin). `H0idx`/`H1idx` are the index stream alone, against 4.0 stored. `H0sc`/`H0min` are the 6-bit streams, against 6.0 stored.

## Unmodelled bytes (honest-total guard)

| qtype | tensors |       bytes | % of tensor bytes |
|-------|--------:|------------:|------------------:|
| Q6_K  |      29 | 423,843,840 |            38.48% |
| F32   |     113 |     495,616 |             0.04% |

Total unmodelled: 424,339,456 bytes (38.53% of tensor bytes). These are carried at full stored size in every total above; measuring them (Q6_K/Q5_K have their own 4/8-bit index streams) is a follow-up and would only improve the headline.

## Method

- Super-block layout parsed from the ggml `block_q4_K` struct (144 B / 256 weights); the parser is asserted to agree *exactly* with `gguf.quants.Q4_K.dequantize_blocks` at startup (`--self-test`).
- Order-0 and order-1 are plug-in (maximum-likelihood) estimates over the whole family; order-1 conditions on the previous weight and uses only the 255 pairs inside each super-block, so super-blocks stay independently decodable.
- Entropy is the coder-independent bound; a real rANS/arithmetic coder lands within ~0.1-1% of it, and a static table costs a few KB.
- Speedup figure assumes a purely bandwidth-bound decode (bytes moved / bytes moved).

