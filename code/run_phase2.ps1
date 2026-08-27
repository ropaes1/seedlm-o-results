# run_phase2.ps1 - SeedLM+O Phase 2 overnight launcher (Windows 11, no WSL).
#
# NOTE: this file is deliberately pure ASCII. Windows PowerShell 5.1 reads .ps1
# as ANSI unless there is a BOM, so non-ASCII punctuation here breaks parsing.
#
# BEFORE YOU START THE OVERNIGHT RUN:
#   1. Plug in AC power (the fit stage pins the GPU for hours).
#   2. Close Chrome / anything else holding VRAM - the eval stage needs two
#      Qwen3-0.6B copies resident (2 x 1.2 GB) on an 8 GB card.
#   3. Disable sleep:      powercfg /change standby-timeout-ac 0
#      (restore later with e.g. powercfg /change standby-timeout-ac 30)
#
# ONE-TIME DEPENDENCIES (already installed on this machine):
#   uv pip install --python .\.venv\Scripts\python.exe transformers matplotlib accelerate
#   ('accelerate' is required by transformers >= 5 for device_map="cuda".)
#
# WHAT IT DOES:
#   runner.py --stage all  ==  fit (16 variants x 196 tensors @16384 seeds,
#   cached/resumable) -> eval (teacher-forced KL + 160-token harness per
#   variant) -> pick winner -> refit winner @65535 -> re-eval -> gate verdict.
#   Everything is written to results/ and cache/; models/originals/ is opened
#   read-only and its sha256 is re-verified at the end of every stage.
#
# SAFE TO KILL AND RERUN: fits are cached by
# sha1(tensor|C|P|n_seeds|rule|p|generator_seed); a rerun prints "SKIP cached".

Set-Location -Path $PSScriptRoot

# Fragmentation guard for the long fit loop (large [S, b, C] buffers).
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
# Keep Python's stdout UTF-8 whatever the console codepage is.
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "venv python not found at $py" }

New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "results") | Out-Null
$log = Join-Path $PSScriptRoot "results\overnight.log"

Write-Host "=== SeedLM+O Phase 2 ==="
Write-Host "python : $py"
Write-Host "log    : $log"
Write-Host "started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# PowerShell 5.1 turns a native command's stderr into ErrorRecords when 2>&1 is
# used; with ErrorActionPreference=Stop that would abort on harmless progress
# output. Keep it Continue and gate on $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"

# Selftest first: no point burning a night on a broken build.
& $py runner.py --selftest 2>&1 | Tee-Object -FilePath $log
if ($LASTEXITCODE -ne 0) {
    Write-Host "SELFTEST FAILED (exit $LASTEXITCODE) - aborting"
    exit $LASTEXITCODE
}

& $py runner.py --stage all 2>&1 | Tee-Object -FilePath $log -Append
$rc = $LASTEXITCODE

Write-Host "finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  exit=$rc"
Write-Host "results : results\phase2_results.json, results\summary.md,"
Write-Host "          results\mp_atlas.png, results\spectra.json, results\run.log"
exit $rc
