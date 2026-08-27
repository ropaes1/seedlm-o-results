# RUNBOOK: the bytes-touched ladder

**Three rungs, one number per rung.** Follow this top to bottom.
Every command is copy-pasteable. Nothing here requires a decision.

Verified against **llama.cpp build `b10502`** (released 2026-08-19). Model
filenames, sizes and SHA-256s were checked against Hugging Face on 2026-08-19.

All commands are run from this `ladder/` directory.

---

## 0. Why this ladder exists (read once, 90 seconds)

At batch size 1, which is what a laptop chat session is, decoding is not
compute-bound. It is bound by how many **bytes of weights must be dragged out
of memory to produce one token**:

```
decode_tok_s  ~=  B_eff  /  bytes_touched_per_token
```

Every celebrated inference speedup attacks that fraction from a different side:

| Lever | What it changes | Rung |
|---|---|---|
| Quantization | fewer bytes per weight (numerator) | R0 and R1 hold Q4_K_M constant, so this is the *control* |
| MoE | fewer weights touched per token (numerator) | **R2**: 3.04B active of 30.53B total |
| Speculative decoding | more tokens per weight-read (denominator of the *rate*) | **+S** |

The ladder measures each lever **separately, on the actual target machine**,
so the folklore ("MoE is 10x cheaper!") becomes a measured curve with an
honest error bar.

Background reading the ladder is built on:

- **Roofline mental model**: <https://jax-ml.github.io/scaling-book/> (ch. 1,
  7, 8). Chapter 7 in particular is the arithmetic-intensity argument for why
  batch-1 decode lives on the memory-bandwidth side of the roofline.
- **Speculative decoding in practice**: <https://inco.ai/blog/dflash2/>
  (reports 2.7-3.4x at batch 1 on a 27B; llama.cpp supports the family).

**Win condition: R2 decode >= 12 tok/s.** That is the chosen threshold for
"daily-drivable". Everything else is data.

---

## 1. Machine and the crash protocol

**Target machine:** Windows 11 laptop, RTX 5070 Laptop GPU (Blackwell,
**compute capability 12.0 / sm_120**), 8 GB GDDR7 (8151 MiB reported), 32 GB
RAM (31.5 usable), Intel Ultra 9 275HX.

R2 with `--n-cpu-moe` runs the GPU and all CPU cores hot simultaneously, the
heaviest sustained load in this runbook. The toolchain is built so a crash
costs one rung, not the whole session.

### Crash protocol

- `ladder_bench.py` **appends one line to `results.jsonl` and `fsync()`s it**
  the instant a measurement exists, before the next configuration's process
  even starts. If the machine dies, everything up to the last finished
  measurement is on disk.
- The R2 `-ncmoe` sweep runs **one `llama-bench` process per value**, on
  purpose. A crash kills one config, not the sweep.
- Raw `llama-bench` stdout/stderr for every run is kept in `logs/`.

**After a crash, to resume:**

```powershell
# 1. See what survived:
python ladder_bench.py render --stdout
# 2. Restart the sweep at the value AFTER the one that died, e.g. if
#    ncmoe=40 crashed:
python ladder_bench.py run --rung R2 --ncmoe 36,32
```

`render` rebuilds `results.md` from `results.jsonl` and **tolerates a torn
final line** (the signature of a power-cut mid-write): it skips it and keeps
every earlier row.

### Reduce the risk before you start (do these)

1. Plug in the AC adapter. Windows power mode: **Best performance**.
2. Close: browsers, Docker Desktop, chat apps, cloud-sync clients, any editor
   with a language server. R2 needs ~19 GB of RAM to itself.
3. `wsl --shutdown` (WSL2 reserves a large RAM balloon).
4. **Do not run this ladder concurrently with other GPU work.**
5. Confirm nothing else is on the GPU: `nvidia-smi`.

---

## 2. Step 1: Downloads (~21.9 GB total, do this on a good connection)

Everything below is a one-time fetch.

```powershell
mkdir dl, models -Force | Out-Null
$ProgressPreference = 'SilentlyContinue'
```

### 2a. llama.cpp prebuilt: Windows, CUDA 13.3, x64 (~537 MB)

> **Why 13.3 and not 12.4:** the GPU is Blackwell, compute capability
> **12.0 (sm_120)**. CUDA 12.4 predates sm_120 entirely; that build will not
> run this card. Take the **13.3** zips (requires a CUDA 13-capable driver).

```powershell
$B = "https://github.com/ggml-org/llama.cpp/releases/download/b10502"
Invoke-WebRequest "$B/llama-b10502-bin-win-cuda-13.3-x64.zip"  -OutFile "dl\llama-b10502-bin-win-cuda-13.3-x64.zip"
Invoke-WebRequest "$B/cudart-llama-bin-win-cuda-13.3-x64.zip"  -OutFile "dl\cudart-llama-bin-win-cuda-13.3-x64.zip"
```

| File | Bytes | SHA-256 |
|---|---:|---|
| `llama-b10502-bin-win-cuda-13.3-x64.zip` | 146,813,754 | `657ad104b7c2f3aaf9abac91b48ffb72a2556cb8a6a38d395eaaf64bc1f1f719` |
| `cudart-llama-bin-win-cuda-13.3-x64.zip` | 390,970,417 | `1462a050eb4c684921ba51dcc4cc488a036674c3e73e9945ee705b854808d03e` |

Verify and unpack **both zips into the same folder** (the cudart zip supplies
the CUDA runtime DLLs the binaries need):

```powershell
Get-FileHash dl\llama-b10502-bin-win-cuda-13.3-x64.zip -Algorithm SHA256 | Format-List
Get-FileHash dl\cudart-llama-bin-win-cuda-13.3-x64.zip -Algorithm SHA256 | Format-List

Expand-Archive dl\llama-b10502-bin-win-cuda-13.3-x64.zip  -DestinationPath "llama.cpp" -Force
Expand-Archive dl\cudart-llama-bin-win-cuda-13.3-x64.zip  -DestinationPath "llama.cpp" -Force
```

> If `Expand-Archive` produces a nested folder (e.g. `llama.cpp\build\bin\`),
> move the contents up so that `llama.cpp\llama-bench.exe` exists. Check with:
> `Get-ChildItem llama.cpp\llama-bench.exe`

### 2b. The three GGUFs (~21.4 GB)

All from Unsloth's repos, all Q4_K_M so quantization is held constant across
the ladder.

```powershell
$H = "https://huggingface.co"
Invoke-WebRequest "$H/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf?download=true"           -OutFile "models\Qwen3-0.6B-Q4_K_M.gguf"
Invoke-WebRequest "$H/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf?download=true"               -OutFile "models\Qwen3-4B-Q4_K_M.gguf"
Invoke-WebRequest "$H/unsloth/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf?download=true"     -OutFile "models\Qwen3-30B-A3B-Q4_K_M.gguf"
```

| Rung | File | Bytes | SHA-256 |
|---|---|---:|---|
| R0 / draft | `Qwen3-0.6B-Q4_K_M.gguf` | 396,705,472 | `ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a` |
| R1 | `Qwen3-4B-Q4_K_M.gguf` | 2,497,281,312 | `f6f851777709861056efcdad3af01da38b31223a3ba26e61a4f8bf3a2195813a` |
| R2 | `Qwen3-30B-A3B-Q4_K_M.gguf` | 18,556,686,912 | `9f1a24700a339b09c06009b729b5c809e0b64c213b8af5b711b3dbdfd0c5ba48` |

Verify all three at once (this reads 21 GB off disk and takes a couple of
minutes; it is worth it, a corrupt 18 GB download wastes more than that):

```powershell
Get-ChildItem models\*.gguf | Get-FileHash -Algorithm SHA256 |
  Select-Object @{n='File';e={Split-Path $_.Path -Leaf}}, Hash | Format-Table -AutoSize
```

> **Fast path:** if `curl.exe` or `aria2c` is available, `Invoke-WebRequest` is
> slow for multi-GB files. `curl.exe -L -C - -o models\Qwen3-30B-A3B-Q4_K_M.gguf "<url>"`
> resumes on interruption; use it for the 18 GB file.

---

## 3. Step 2: Sanity checks (2 minutes, do not skip)

```powershell
# The binaries run and see the GPU:
.\llama.cpp\llama-bench.exe --version
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv

# The harness works end-to-end against built-in fixtures (no models needed):
python ladder_bench.py selftest
```

`selftest` must print **`SELFTEST PASSED`**. It exercises the llama-bench JSON
parser, the GGUF tensor-table reader, the bytes/token estimator, the crash-safe
append, torn-line recovery, and the `results.md` renderer.

Now confirm the **exact** bytes/token numbers for the actual files (this reads
the real GGUF headers, Method A, the precise path):

```powershell
python ladder_bench.py bytes --model qwen3-0.6b     --gguf models\Qwen3-0.6B-Q4_K_M.gguf
python ladder_bench.py bytes --model qwen3-4b       --gguf models\Qwen3-4B-Q4_K_M.gguf
python ladder_bench.py bytes --model qwen3-30b-a3b  --gguf models\Qwen3-30B-A3B-Q4_K_M.gguf --methodology
```

**Expected (analytic estimates; the GGUF-exact numbers will be close):**

| Rung | Model | Total params | Active/token | bytes/token |
|---|---|---:|---:|---:|
| R0 | Qwen3-0.6B | 0.60 B | 0.60 B (100.0%) | ~397 MB |
| R1 | Qwen3-4B | 4.02 B | 4.02 B (100.0%) | ~2,497 MB |
| R2 | Qwen3-30B-A3B | 30.53 B | 3.04 B (**9.96%**) | ~1,849 MB |

**That 9.96% is the whole thesis of R2**: a model 7.4x bigger than R1 on disk
touches *fewer* bytes per token than R1 does.

Python note: stdlib only, 3.12. `psutil` is optional; without it the harness
uses Win32 `GlobalMemoryStatusEx` via `ctypes`, which is enough. If you want
per-process RSS: `pip install psutil` (~0.5 MB).

---

## 4. Step 3: R0, Qwen3-0.6B (baseline / bandwidth probe)

**Lever isolated:** none. R0 measures *the machine's effective bandwidth* with
a model small enough (397 MB) to sit entirely in VRAM with room to spare. Every
later rung is read relative to this.

```powershell
python ladder_bench.py run --rung R0
```

Equivalent raw command (what it runs):

```powershell
.\llama.cpp\llama-bench.exe -m models\Qwen3-0.6B-Q4_K_M.gguf `
  -p 512 -n 128 -ngl 99 -r 3 -fa on -o json
```

- **Wall time:** ~2 minutes.
- **Record:** `pp512` (prefill tok/s) and `tg128` (decode tok/s).
- **Expected decode:** ~150-260 tok/s.

> **Read this correctly.** R0 will show an *implied bandwidth well below* the
> GDDR7 spec. That is not a broken measurement. At 0.6B the per-layer kernel
> launch and sync overhead is a large fraction of each token's wall time, so
> R0 is partly latency-bound, not purely bandwidth-bound. R0 is therefore a
> **lower bound** on B_eff. R1 (big enough to amortize launch overhead, still
> fully VRAM-resident) is the better bandwidth estimate.

---

## 5. Step 4: R1, Qwen3-4B (bpw + dense scaling)

**Lever isolated:** dense scaling at constant quantization. 6.3x more bytes
per token than R0, same Q4_K_M, still 100% VRAM-resident (2.5 GB of 8 GB).

```powershell
python ladder_bench.py run --rung R1
```

- **Wall time:** ~4 minutes.
- **Expected decode:** ~55-95 tok/s.
- **The check that matters:** `R0_decode / R1_decode` should land near
  `R1_bytes / R0_bytes = 6.29`. If R1 is *relatively faster* than that ratio
  predicts, you have just measured R0's launch-overhead tax.

R1's implied bandwidth is the **best pure-VRAM B_eff estimate**. Write it
down: R2's numbers get compared against it.

---

## 6. Step 5: R2, Qwen3-30B-A3B (the bytes-TOUCHED rung)

This is the rung the ladder is for, and the one with the memory-pressure
risk. Read this whole section before running it.

### The memory plan

18.6 GB of model, 8 GB of VRAM, 31.5 GB of RAM. The split:

- **Routed expert weights** (`ffn_*_exps`), ~90% of the file, go to
  **system RAM**, executed on CPU.
- **Attention, norms, the router (`ffn_gate_inp`), the output projection, and
  the entire KV cache** stay on the **GPU**, for every layer.

The flag that does this is **`--n-cpu-moe N` / `-ncmoe N`**: *keep the
Mixture-of-Experts weights of the first N layers in CPU memory.* Qwen3-30B-A3B
has **48 layers**.

> **Critical idiom, and the thing that trips everyone up:** keep `-ngl 99`
> (offload *everything*) and control VRAM with `-ncmoe` instead. Do **not**
> lower `-ngl`; that would strand attention and KV on the CPU, which is
> exactly backwards. `-ncmoe` is the modern replacement for hand-writing
> `-ot "\.ffn_.*_exps\.=CPU"`; the `--override-tensor` regex still works if
> you need finer control, but you do not here.

`-ncmoe 48` = every layer's experts on CPU (guaranteed to fit, slowest).
Lower values pull expert layers back onto the GPU until VRAM runs out.
**The sweep finds the knee.**

### Before you run

```powershell
wsl --shutdown
Get-CimInstance Win32_OperatingSystem | Select-Object @{n='FreeRAM_GB';e={[math]::Round($_.FreePhysicalMemory/1MB,1)}}
```

You want **>= 22 GB free**. If you don't have it, close more.

### Run the sweep

```powershell
python ladder_bench.py run --rung R2
```

This runs **five separate `llama-bench` processes**, `-ncmoe` = 48, 44, 40,
36, 32, appending a durable row after each:

```powershell
# what each iteration runs:
.\llama.cpp\llama-bench.exe -m models\Qwen3-30B-A3B-Q4_K_M.gguf `
  -p 512 -n 128 -ngl 99 -ncmoe 48 -r 3 -fa on -o json
```

- **Wall time:** ~45-70 minutes total. The first process is slowest because
  Windows must page 18.6 GB off disk into the file cache; later ones reuse it.
- **Expected decode:** ~9-14 tok/s at `-ncmoe 48`, rising to ~12-18 tok/s
  around `-ncmoe 36`, then **failing** (CUDA OOM) somewhere near 32-28.
- **An OOM at the low end is a successful experiment**, not a failure: it
  locates the VRAM ceiling. `ladder_bench.py` logs the failure, skips that
  config, and continues.

### If it OOMs before `-ncmoe 36`

VRAM headroom is smaller than expected (a display-attached laptop GPU loses
~1-1.5 GB to the desktop compositor). Just run the high end:

```powershell
python ladder_bench.py run --rung R2 --ncmoe 48,46,44,42
```

### If the machine is thrashing (disk pinned, RAM at 100%)

Fall back to the smaller quant, Unsloth's dynamic 4-bit: **0.84 GB smaller**
and generally *better* quality per byte:

```powershell
Invoke-WebRequest "https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-UD-Q4_K_XL.gguf?download=true" -OutFile "models\Qwen3-30B-A3B-UD-Q4_K_XL.gguf"
# 17,715,663,424 bytes
```

Then point the harness at it with `--extra`, or just note it and move on.

### What to record

The decode tok/s at the **best** `-ncmoe`, and whether it clears **12 tok/s**.
`results.md` marks this automatically in the `Win` column.

> **Expect the implied-bandwidth number for R2 to look "wrong", far below
> DDR5 spec. That gap is the finding.** With experts on the CPU, the binding
> constraint at batch 1 is not DDR5 streaming bandwidth; it is the CPU doing
> 48 layers x 8 experts x 3 matmuls = 1,152 small GEMMs per token, which
> neither saturates memory nor the vector units. R2's rows are flagged `*` in
> `results.md` because their bandwidth blends two memory tiers (GDDR7 for
> attention, DDR5 for experts) and must not be compared to R0/R1 directly.

---

## 7. Step 6: +S, speculative decoding

**Lever isolated:** amortization, i.e. more accepted tokens per weight-read.

> ### Two things to know before running this
>
> **1. `llama-bench` does NOT support speculative decoding.** As of b10502
> only **`llama-server`** does. So the +S numbers come from a different tool
> than R0-R2, and a llama-bench tok/s is *not* directly comparable to a
> llama-server tok/s. `ladder_bench.py spec` therefore runs **both** a
> no-speculation `llama-server` baseline **and** the speculative variant over
> the same three prompts, and reports the ratio. Compare +S only to its
> matched `server-base` row.
>
> **2. The flags were renamed.** The old `--draft-max` / `--draft-min` /
> plain `-md` spelling is gone. Current set:
> `--spec-type draft-simple`, `-md/--spec-draft-model`,
> `--spec-draft-n-max`, `--spec-draft-n-min`, `--spec-draft-ngl`.

### What applies to which rung

| Rung | Draft support | Why |
|---|---|---|
| **R0** Qwen3-0.6B | no draft model | Qwen3-0.6B *is* the smallest Qwen3. Nothing smaller shares its tokenizer. Use `--spec-type ngram-mod` if you want a draft-free number. |
| **R1** Qwen3-4B | yes: `draft-simple` + Qwen3-0.6B | Same Qwen3 tokenizer (vocab 151,936). This is the textbook case. |
| **R2** Qwen3-30B-A3B | yes: `draft-simple` + Qwen3-0.6B | Same tokenizer. **Best expected payoff**, see below. |

> **On MTP / EAGLE-3 / DFlash, explicitly, so you don't hunt for it later:**
> **Qwen3-30B-A3B has no MTP head.** `--spec-type draft-mtp` requires a model
> *trained* with multi-token-prediction layers (that is Qwen3.6-35B-A3B, a
> different and later model). Likewise `draft-eagle3`, `draft-dflash` and
> `draft-dspark` need a separately-trained draft head converted via
> `convert_hf_to_gguf.py --target-model-dir`, which means downloading
> full-precision HF weights and running torch. **All of that is out of scope
> here.** `draft-simple` with the 0.6B is the move.

### Run it

```powershell
# R1 + speculation (~10 min): runs matched baseline, then speculative
python ladder_bench.py spec --rung R1

# R2 + speculation (~20 min): use the best -ncmoe you found in step 5
python ladder_bench.py spec --rung R2 --ncmoe 36
```

Equivalent raw server invocation for R2:

```powershell
.\llama.cpp\llama-server.exe -m models\Qwen3-30B-A3B-Q4_K_M.gguf `
  -md models\Qwen3-0.6B-Q4_K_M.gguf `
  --spec-type draft-simple --spec-draft-n-max 8 --spec-draft-ngl 99 `
  -ngl 99 -ncmoe 36 -c 4096 -fa on --host 127.0.0.1 --port 8099
```

- **Expected R1 speedup:** ~1.4-2.0x.
- **Expected R2 speedup:** ~1.3-1.8x, and here is the interesting part:
  **speculation should help R2 *more* than the raw ratio suggests**, because
  R2's verification step batches the draft tokens through the *CPU-resident
  experts*, and CPU GEMM at batch 8 is far more efficient per token than at
  batch 1. R2 is the rung where the CPU-offload penalty and the speculation
  win directly cancel. **If R2 misses 12 tok/s in step 5, this is the step
  that can rescue it.**
- **Also record the acceptance rate** (`draft_n_accepted / draft_n`).
  Healthy is 55-75%. Below ~40% means the draft is too weak and the extra
  work costs more than it saves.
- **Tuning knob if acceptance is low:** drop `--spec-draft-n-max` to 4-5.
  If acceptance is high (>75%), raise it to 12-16.

---

## 8. Step 7: the feel-check (5 minutes, subjective but the real acceptance test)

Numbers say 12 tok/s. Your eyes decide if that is daily-drivable.

```powershell
.\llama.cpp\llama-server.exe -m models\Qwen3-30B-A3B-Q4_K_M.gguf `
  -md models\Qwen3-0.6B-Q4_K_M.gguf --spec-type draft-simple --spec-draft-n-max 8 `
  -ngl 99 -ncmoe 36 -c 8192 -fa on --jinja --port 8080
```

Open <http://127.0.0.1:8080>, ask it three real questions from your actual
work, and answer one question: **would you use this instead of a cloud model
for a first draft?** Write the answer at the bottom of `results.md`. That
sentence is the deliverable, more than any tok/s figure.

(Or `llama-cli` if you prefer a terminal:
`.\llama.cpp\llama-cli.exe -m models\Qwen3-30B-A3B-Q4_K_M.gguf -ngl 99 -ncmoe 36 -c 8192 -fa on --jinja`)

---

## 9. Reading the results

```powershell
python ladder_bench.py render --stdout
```

`results.md` grows a row per measurement, plus a speculative-decoding section
and the full bytes/token methodology. `results.jsonl` is the durable source of
truth: one JSON object per measurement, with the full llama-bench metadata,
the bytes/token derivation, RAM/VRAM peaks, GPU temperature and power ceiling,
and the exact command line.

**The three sentences you should be able to write when it's done:**

1. R1 says this machine sustains ~`X` GB/s of effective weight bandwidth from
   VRAM at batch 1.
2. R2 touches only 9.96% of its weights per token, and hits `Y` tok/s, which
   is `Y/X_predicted` of what a naive roofline predicts, because `<reason>`.
3. Speculation multiplies R2 by `Z`x, at `W`% draft acceptance.

---

## 10. Time budget

| Step | Wall time |
|---|---|
| 1. Downloads (21.9 GB) | connection-bound |
| 2. Sanity checks | 5 min |
| 3. R0 | 2 min |
| 4. R1 | 4 min |
| 5. R2 sweep (5 configs) | 45-70 min |
| 6. +S for R1 and R2 | 30 min |
| 7. Feel-check | 5 min |
| **Total after downloads** | **~1.5-2 hours** |

Optional extension if the machine is stable: context-depth scaling, which
exposes KV-cache traffic (excluded from the bytes/token model by design):

```powershell
python ladder_bench.py run --rung R1 --depth 0,4096,16384
python ladder_bench.py run --rung R2 --ncmoe 36 --depth 0,4096
```

The gap between predicted and measured tok/s *at depth* is the KV traffic.

---

## 11. Cloud-GPU fallback (NOT the default)

Only if the laptop cannot complete the run. This measures a *different
machine*, so it answers "is this model good?" and **not** "is this model
daily-drivable on this laptop?", which is the actual question. Prefer
rerunning locally.

1. Rent a single **RTX 4090 24 GB** or **L40S 48 GB** from any GPU cloud.
   Pick a CUDA 12.8+ PyTorch image.
2. Inside the box:
   ```bash
   apt-get update && apt-get install -y build-essential cmake git libcurl4-openssl-dev
   git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
   cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --config Release
   pip install -U "huggingface_hub[cli]"
   hf download unsloth/Qwen3-30B-A3B-GGUF Qwen3-30B-A3B-Q4_K_M.gguf --local-dir models
   ./build/bin/llama-bench -m models/Qwen3-30B-A3B-Q4_K_M.gguf -p 512 -n 128 -ngl 99 -fa on -o json > r2_cloud.json
   ```
   On 24 GB+ the whole model fits in VRAM, so **no `-ncmoe` at all**, which
   also means it does not measure the CPU-offload lever, i.e. it does not
   measure the thing R2 exists to measure.
3. Bring the JSON home and fold it in, clearly labelled as not-the-laptop:
   ```powershell
   python ladder_bench.py ingest --json r2_cloud.json --rung R2-cloud `
     --model qwen3-30b-a3b --variant cloud-4090 --notes "rented RTX 4090, full VRAM residency"
   ```

---

## 12. Command reference

```
python ladder_bench.py selftest                      # fixtures; needs nothing
python ladder_bench.py run   --rung R0|R1|R2 [--ncmoe 48,40] [--depth 0,4096]
python ladder_bench.py spec  --rung R1|R2 [--ncmoe 36] [--spec-type draft-simple]
python ladder_bench.py bytes --model qwen3-30b-a3b --gguf <path> --methodology
python ladder_bench.py ingest --json <file> --rung R2 --model qwen3-30b-a3b
python ladder_bench.py render --stdout
```

Useful flags on `run`: `--dry-run` (print commands, execute nothing),
`--reps N` (default 3), `--n-prompt/--n-gen` (default 512/128),
`--bin DIR` / `--models DIR`, and `--extra ...` (everything after it goes
straight to `llama-bench`).

---

## Appendix: sources verified 2026-08-19

- llama.cpp release **b10502**, Windows CUDA asset names, sizes and SHA-256
  digests: <https://github.com/ggml-org/llama.cpp/releases>
- `--n-cpu-moe` / `-ncmoe`, `-ot/--override-tensor`, llama-bench option set and
  JSON/markdown output schema:
  <https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md>
  and `tools/llama-bench/llama-bench.cpp`
- Speculative-decoding flag set (`--spec-type`, `--spec-draft-*`), supported
  types, and the fact that **only `llama-server`** implements it:
  <https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md>
- `timings.draft_n` / `timings.draft_n_accepted` in the `/completion` response:
  llama.cpp PR #12603 and `tools/server/README.md`
- Blackwell needs CUDA >= 12.8 (sm_120 absent from 12.4): NVIDIA Blackwell RTX
  software migration guide, and confirmed locally: `compute_cap 12.0`
- GGUF filenames, byte sizes and SHA-256s: Hugging Face API for
  `unsloth/Qwen3-{0.6B,4B,30B-A3B}-GGUF`
- Architecture constants for the bytes/token model: `config.json` for
  `Qwen/Qwen3-{0.6B,4B,30B-A3B}`
- Roofline framing: <https://jax-ml.github.io/scaling-book/>
- Speculative-decoding gains at batch 1: <https://inco.ai/blog/dflash2/>
