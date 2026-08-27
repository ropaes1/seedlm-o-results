# run_phase3_local.ps1 - SeedLM+O Phase 3 Leg A launcher (Windows 11, no WSL).
#
# NOTE: this file is deliberately pure ASCII. Windows PowerShell 5.1 reads .ps1
# as ANSI unless there is a BOM, so non-ASCII punctuation here breaks parsing.
#
# WHAT LEG A IS:
#   Qwen3-1.7B, BOTH seed configs (c8p3 = 4.0 bpw base, c12p4 = 3.0 bpw base),
#   cached-reference-logits evaluation, rtn4 as the only comparator that exists
#   locally. Stage 1 ONLY: --stage fit then --stage eval per config.
#
#   The stage-2 refit (winner @65535 seeds) does NOT run here. It moved to the
#   pod, because the pod must re-measure every verdict row on its own stack
#   anyway (same-stack determinism rule: laptop KL and pod KL are not the same quantity and
#   may never share one gate computation). Leg A's job is therefore exactly one
#   thing: PICK THE WINNER and ship the numbers.
#
#   Nothing multi-GB crosses machines. Fits are deterministic from
#   generator_seed=3407, so the winner's identity is the only input the pod
#   needs; it refits from scratch in less time than a cache upload would take.
#
# BEFORE YOU START THE OVERNIGHT RUN:
#   1. Plug in AC power (the fit stage pins the GPU for hours).
#   2. Close Chrome / anything else holding VRAM. Cached eval mode keeps only
#      ONE 1.7B copy resident (~3.4 GB) on the 8 GB card, but the reference
#      logits pass allocates a [200, 151936] fp32 buffer per prompt.
#   3. Disable sleep:      powercfg /change standby-timeout-ac 0
#      (restore later with e.g. powercfg /change standby-timeout-ac 30)
#   4. Optional: set $env:HF_TOKEN to a fine-grained Hugging Face token
#      and this script uploads results by itself.
#
# BUDGET PROJECTION (from the measured 66 s / 2M params @65535 seeds anchor on
# the 5070; stage 1 runs at 16384 seeds, i.e. ~1/4 of that per tensor):
#   Qwen3-1.7B has 28 layers x 7 families = 196 target tensors, ~1.4 G target
#   params. Per config: seed_only ~1.2-1.6 h + three awq refits (refit-reuse
#   touches only mask-intersecting blocks, ~10-20% of the cost) ~0.5-0.8 h,
#   plus evals ~15 min  =>  ~2-2.5 h per config, ~4-5 h for both. One night with
#   margin. The runner prints a STALL-AWARE ETA after every tensor (gaps > 5 min
#   are excluded from the rate, so a suspend does not poison the projection) and
#   warns if the remaining projection exceeds 8 h.
#
# ONE-TIME DEPENDENCIES (already installed on this machine):
#   uv pip install --python .\.venv\Scripts\python.exe transformers matplotlib accelerate gguf
#   ('accelerate' is required by transformers >= 5 for device_map="cuda";
#    'gguf' is needed by comparators.py for the K-quant dequant path.)
#   lm-eval is deliberately NOT installed locally - that suite is pod-only.
#
# SAFE TO KILL AND RERUN: fits are cached by
# sha1(slug|config|tensor|C|P|n_seeds|rule|p|generator_seed) under
# cache\Qwen3-1.7B\; a rerun prints "SKIP cached" and recomputes nothing.
# Phase 2's cache\ and results\ are a separate namespace and are never touched.
#
# ---------------------------------------------------------------------------
# OPTIONAL (addendum #2): replicate the 1.7B stage 2 locally, in parallel with
# the pod, to get a second INDEPENDENT same-stack verdict plus a cross-stack
# drift table. This is NOT part of the default flow below - run it by hand
# after this script has printed the winner, substituting the winner's config:
#
#   .\.venv\Scripts\python.exe runner.py --stage refit-full `
#       --model-dir models/originals/Qwen3-1.7B --config <winner config>
#   .\.venv\Scripts\python.exe runner.py --stage eval `
#       --model-dir models/originals/Qwen3-1.7B --config <winner config> `
#       --n-seeds 65535
#   .\.venv\Scripts\python.exe runner.py --stage verdict `
#       --model-dir models/originals/Qwen3-1.7B --config <winner config>
#
# Merge the two machines' rows by downloading the pod's phase3_results.json into
# results\Qwen3-1.7B\ and rerunning --stage verdict: every row carries a `stack`
# tag, each stack gets its own self-contained verdict (they are never mixed),
# and summary.md gains a "## cross-stack drift" section reporting per-variant
# KL/top-1 deltas and a REPLICATED / DIVERGED line.
# ---------------------------------------------------------------------------

Set-Location -Path $PSScriptRoot

# Fragmentation guard for the long fit loop (large [S, b, C] buffers).
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
# Keep Python's stdout UTF-8 whatever the console codepage is.
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "venv python not found at $py" }

$modelDir = "models/originals/Qwen3-1.7B"
if (-not (Test-Path (Join-Path $PSScriptRoot $modelDir))) {
    throw "model not found at $modelDir - download it first: $py -c ""from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-1.7B', local_dir='$modelDir')"""
}

$slug = "Qwen3-1.7B"
$resDir = Join-Path $PSScriptRoot "results\$slug"
New-Item -ItemType Directory -Force -Path $resDir | Out-Null
$log = Join-Path $resDir "legA.log"

Write-Host "=== SeedLM+O Phase 3, Leg A (Qwen3-1.7B, stage 1 only) ==="
Write-Host "python : $py"
Write-Host "model  : $modelDir"
Write-Host "log    : $log"
Write-Host "started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# PowerShell 5.1 turns a native command's stderr into ErrorRecords when 2>&1 is
# used; with ErrorActionPreference=Stop that would abort on harmless progress
# output. Keep it Continue and gate on $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
$t0 = Get-Date

# Selftest first: no point burning a night on a broken build.
& $py runner.py --selftest 2>&1 | Tee-Object -FilePath $log
if ($LASTEXITCODE -ne 0) {
    Write-Host "SELFTEST FAILED (exit $LASTEXITCODE) - aborting"
    exit $LASTEXITCODE
}

foreach ($cfg in @("c8p3", "c12p4")) {
    foreach ($stage in @("fit", "eval")) {
        $ts = Get-Date
        Write-Host ""
        Write-Host "--- $slug / $cfg / --stage $stage  (started $(Get-Date -Format 'HH:mm:ss')) ---"
        & $py runner.py --stage $stage --model-dir $modelDir --config $cfg `
            --eval-mode cached 2>&1 | Tee-Object -FilePath $log -Append
        $rc = $LASTEXITCODE
        $mins = [math]::Round(((Get-Date) - $ts).TotalMinutes, 1)
        Write-Host "--- $cfg $stage finished in $mins min (exit $rc) ---"
        if ($rc -ne 0) {
            Write-Host "STAGE FAILED - aborting so the log shows where"
            exit $rc
        }
    }
}

# ---------------------------------------------------------------- the winner
Write-Host ""
Write-Host "=== Leg A result ==="
$winnerLine = (& $py runner.py --model-dir $modelDir --print-winner 2>&1 | Select-Object -Last 1)
$winnerRc = $LASTEXITCODE
Write-Host $winnerLine
if ($winnerRc -ne 0) {
    Write-Host "WARNING: could not determine a winner (exit $winnerRc)."
    Write-Host "         Inspect results\$slug\summary.md before starting Leg B."
}
$totalH = [math]::Round(((Get-Date) - $t0).TotalHours, 2)
Write-Host "total wall clock: $totalH h"
Write-Host "artifacts: results\$slug\{phase3_results.json, summary.md, run.log, legA.log}"

# --------------------------------------------------------------- ship to HF
# The pod needs exactly two things from this run: the winner's identity and the
# stage-1 numbers. Both live in results\$slug\. Nothing else crosses machines.
# ref_logits/ is ~1.5 GB of regenerable fp32 tensors - never worth uploading.
$repo = "<your-hf-user>/seedlm-o-results"
$uploadCmd = "$py hf_sync.py upload --repo $repo --local `"$resDir`" --path-in-repo results/$slug --ignore `"ref_logits/*`""
Write-Host ""
if ($env:HF_TOKEN) {
    Write-Host "HF_TOKEN is set - uploading results\$slug\ to $repo ..."
    & $py hf_sync.py upload --repo $repo --local $resDir --path-in-repo "results/$slug" --ignore "ref_logits/*" 2>&1 |
        Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) {
        Write-Host "UPLOAD FAILED (exit $LASTEXITCODE). Run it by hand:"
        Write-Host "  $uploadCmd"
    } else {
        Write-Host "uploaded. Leg B can now start."
    }
} else {
    Write-Host "HF_TOKEN not set - upload by hand before starting Leg B:"
    Write-Host ""
    Write-Host "  `$env:HF_TOKEN = '<fine-grained token>'"
    Write-Host "  $uploadCmd"
    Write-Host ""
}

Write-Host "finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
exit 0
