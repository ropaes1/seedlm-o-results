# run_w1_local.ps1 - SeedLM+O probe W1: activation-weighted fit objective.
#
# NOTE: this file is deliberately pure ASCII. Windows PowerShell 5.1 reads .ps1
# as ANSI unless there is a BOM, so non-ASCII punctuation here breaks parsing.
#
# ---------------------------------------------------------------------------
# HARD SAFETY RULE - READ BEFORE EDITING
#
#   This script runs Qwen3-0.6B and ONLY Qwen3-0.6B. Do not repoint $modelDir
#   at Qwen3-1.7B. The test laptop has a reproducible hardware fault under
#   the c12p4-on-1.7B load profile; such runs were moved to rented GPUs.
#   Qwen3-0.6B has hours of stable precedent from Phase 2. The 1.7B weighted
#   probe runs on the POD.
# ---------------------------------------------------------------------------
#
# WHAT W1 IS:
#   Phase 3 Leg A found that per-tensor rel_err is nearly scale-invariant while
#   behavioural damage explodes with scale. Equal-magnitude error, unequal
#   damage means the error STRUCTURE is the problem: the plain per-block L2 fit
#   is salience-blind and drops residual on high-activation input channels.
#   W1 weights the fit objective by the AWQ activation scale of each input
#   channel, so the residual lands where it costs least.
#
#   AT ZERO BIT COST. The stored format is untouched - same 16-bit seed, same
#   4-bit shared exponent, same P 4-bit coefficients, same outlier side channel
#   - so a weighted row and its unweighted twin sit at bit-identical bpw and the
#   comparison needs no budget matching. act_scales are a fit-time input only;
#   decode never sees them.
#
#   Weighted runs carry a "w"-suffixed config slug (c8p3w, c12p4w). That gives
#   them their own cache namespace and their own results rows, so nothing this
#   script does can overwrite an unweighted fit or an unweighted number.
#
# WHAT THIS SCRIPT DOES:
#   1. runner.py --selftest                       (includes the six W1 units)
#   2. per config in c8p3w, c12p4w:  --stage fit then --stage eval, 65535 seeds
#   3. prints a comparison table: weighted rows against the existing unweighted
#      0.6B rows from Phase 2 and Phase 3
#   4. uploads NOTHING. W1 is a local probe; no HF traffic.
#
# WHAT EXISTS TO COMPARE AGAINST, AND WHAT DOES NOT (read before interpreting):
#   c8p3  - results\phase2_results.json has full-model unweighted 0.6B rows at
#           65535 seeds (seed_only 0.3021, awq_p1.0 0.1850, rtn4 0.1704). The
#           weighted c8p3w rows land at bit-identical bpw next to those, so the
#           pairing is strict on bits and only crosses layout/eval-mode (see the
#           caveat the table prints).
#   c12p4 - there is NO full-model unweighted 0.6B row. The only c12p4 rows in
#           results\Qwen3-0.6B\ are the 2-layer, 512-seed smoke rows, which are
#           not comparable to anything and are filtered out of the table.
#           Pass -WithUnweightedTwin to fit and eval the unweighted twin in the
#           same run, same layout, same seed count, same stack - the only way to
#           get a clean c12p4 delta. It doubles the wall clock.
#
# WHAT IT DELIBERATELY DOES NOT DO:
#   - no --stage all (no automatic stage-2 refit; --n-seeds 65535 IS stage 2)
#   - no comparators (0.6B has no local GGUF/AWQ artifacts; those rows stay
#     absent and the verdict for this model stays INCOMPLETE, which is correct
#     and not a W1 result)
#   - no upload, no run_phase3_pod.sh changes
#
# BUDGET PROJECTION (measured anchor: 66 s per 2M params at 65535 seeds on this
# 5070; the weighted path adds ~10-20% for the per-pattern solves):
#   Qwen3-0.6B has 28 layers x 7 families = 196 target tensors, ~440M target
#   params. Per config: seed_only ~4 h + three awq refits (refit-reuse touches
#   only mask-intersecting blocks) ~1-1.5 h, plus one eval pass ~10 min
#   =>  ~5-6 h per config,  ~10-12 h for both. That is TWO nights, or one night
#   per config. Use -Configs to split:
#
#       .\run_w1_local.ps1 -Configs c12p4          # night 1 (the sub-4 config)
#       .\run_w1_local.ps1 -Configs c8p3           # night 2
#
#   c12p4 first: it is the config the Phase 3 gates care about, and the one
#   where the unweighted numbers were worst. With -WithUnweightedTwin, double
#   every number above (~10-12 h for one config, both objectives).
#
# BEFORE YOU START:
#   1. AC power. The fit stage pins the GPU for hours.
#   2. Close Chrome / anything holding VRAM.
#   3. Disable sleep:  powercfg /change standby-timeout-ac 0
#      (restore later with e.g. powercfg /change standby-timeout-ac 30)
#   4. Nothing else. No HF_TOKEN needed - this script never uploads.
#
# SAFE TO KILL AND RERUN: fits are cached per tensor under cache\Qwen3-0.6B\ by
# sha1(slug|config|tensor|C|P|n_seeds|rule|p|generator_seed) with the weighted
# config slug in the key; a rerun prints "SKIP cached" and recomputes nothing.
#
# NOTE ON results\Qwen3-0.6B\phase3_results.json: --stage eval MERGES its rows
# into that file (by name+stack, newest wins). The weighted rows are new names,
# so nothing existing is overwritten; the older 2-layer smoke rows stay put and
# the comparison table below filters them out by seed count.
#
# WIRING SMOKE TEST (no GPU, no state change):
#       .\run_w1_local.ps1 -DryRun
#   prints every command it would run and exits.

[CmdletBinding()]
param(
    # Which seed configs to probe, in order. Weighting is applied to both.
    [string[]] $Configs = @("c12p4", "c8p3"),
    # Candidate seeds per block. 65535 is the full sweep (stage-2 quality).
    [int] $NSeeds = 65535,
    # Restrict to the first N decoder layers. 0 means the whole model.
    [int] $LayersLimit = 0,
    # Also fit+eval the UNWEIGHTED twin of each config, so the comparison is
    # same-layout / same-seeds / same-stack. Doubles the wall clock. Needed for
    # c12p4, which has no existing full-model unweighted 0.6B row.
    [switch] $WithUnweightedTwin,
    # Print the commands and exit without touching the GPU.
    [switch] $DryRun
)

Set-Location -Path $PSScriptRoot

# No expandable_segments here: the isolation tests on 2026-08-16 ran without it
# and run_phase3_local.ps1's setting is not needed at 0.6B block sizes.
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "venv python not found at $py" }

# 0.6B ONLY. See the hard safety rule at the top of this file.
$slug = "Qwen3-0.6B"
$modelDir = "models/originals/$slug"
if (-not (Test-Path (Join-Path $PSScriptRoot $modelDir))) {
    throw "model not found at $modelDir"
}

foreach ($cfg in $Configs) {
    if ($cfg -ne "c8p3" -and $cfg -ne "c12p4") {
        throw "unknown config '$cfg' - want c8p3 and/or c12p4"
    }
}

$resDir = Join-Path $PSScriptRoot "results\$slug"
New-Item -ItemType Directory -Force -Path $resDir | Out-Null
$log = Join-Path $resDir "w1.log"

Write-Host "=== SeedLM+O probe W1: activation-weighted fit (Qwen3-0.6B) ==="
Write-Host "python  : $py"
Write-Host "model   : $modelDir"
Write-Host "configs : $($Configs -join ', ')  (fitted as $(($Configs | ForEach-Object { $_ + 'w' }) -join ', '))"
Write-Host "n_seeds : $NSeeds"
if ($LayersLimit -gt 0) { Write-Host "layers  : first $LayersLimit only" }
Write-Host "log     : $log"
Write-Host "started : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
if ($DryRun) { Write-Host "MODE    : -DryRun (printing commands only, nothing runs)" }

# PowerShell 5.1 turns a native command's stderr into ErrorRecords when 2>&1 is
# used; with ErrorActionPreference=Stop that would abort on harmless progress
# output. Keep it Continue and gate on $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
$t0 = Get-Date

function Invoke-Runner {
    param([string] $Label, [string[]] $RunnerArgs)

    Write-Host ""
    Write-Host "--- $Label  (started $(Get-Date -Format 'HH:mm:ss')) ---"
    Write-Host "    $py runner.py $($RunnerArgs -join ' ')"
    if ($DryRun) { return 0 }
    $ts = Get-Date
    # Out-Host keeps the lines visible without leaking them into the function's
    # output stream - otherwise `return $rc` comes back as [lines..., rc] and
    # the caller's `-ne 0` check misfires on a PASSING run.
    & $py runner.py @RunnerArgs 2>&1 | Tee-Object -FilePath $log -Append | Out-Host
    $rc = $LASTEXITCODE
    $mins = [math]::Round(((Get-Date) - $ts).TotalMinutes, 1)
    Write-Host "--- $Label finished in $mins min (exit $rc) ---"
    return $rc
}

# Selftest first: no point burning a night on a broken build. The W1 units in
# here are the ones that prove the weighted path is doing what it claims -
# unit weights reproduce the unweighted fit, the weighted objective actually
# goes down, refit-reuse stays exact, and bit accounting does not move.
$rc = Invoke-Runner "selftest" @("--selftest")
if ($rc -ne 0) {
    Write-Host "SELFTEST FAILED (exit $rc) - aborting"
    exit $rc
}

if ($WithUnweightedTwin) {
    $objectives = @("none", "awq")
} else {
    $objectives = @("awq")
}

foreach ($cfg in $Configs) {
    foreach ($obj in $objectives) {
        if ($obj -eq "awq") { $label = "${cfg}w" } else { $label = $cfg }
        foreach ($stage in @("fit", "eval")) {
            $a = @("--stage", $stage, "--model-dir", $modelDir, "--config", $cfg,
                   "--fit-weighting", $obj, "--n-seeds", "$NSeeds",
                   "--eval-mode", "cached")
            if ($LayersLimit -gt 0) { $a += @("--layers-limit", "$LayersLimit") }
            $rc = Invoke-Runner "$slug / $label / --stage $stage" $a
            if ($rc -ne 0) {
                Write-Host "STAGE FAILED - aborting so the log shows where"
                exit $rc
            }
        }
    }
}

# ------------------------------------------------------- the comparison table
$table = @'
"""W1 comparison: activation-weighted rows against their unweighted twins."""
import json
import sys
from pathlib import Path

P3 = Path("results/Qwen3-0.6B/phase3_results.json")
P2 = Path("results/phase2_results.json")
# Only rows from this run's seed count (plus the seedless comparators/controls).
# Drops the older 2-layer smoke rows at 512/2048 seeds, which are not
# comparable to a full-model fit and would silently poison the table.
KEEP_SEEDS = {0, int(sys.argv[1]) if len(sys.argv) > 1 else 65535}


def load(path):
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("variants", [])
    except Exception as exc:
        print("  (could not read %s: %s)" % (path, exc))
        return []


def num(x, nd=4):
    return "-" if x is None else ("%." + str(nd) + "f") % x


rows3 = [r for r in load(P3) if r.get("n_seeds") in KEEP_SEEDS]
rows2 = [r for r in load(P2) if str(r.get("name", "")).endswith("@65535")]

print("")
print("=" * 100)
print("W1: activation-weighted fit vs the unweighted objective, Qwen3-0.6B")
print("=" * 100)
print("")
hdr = ("%-26s %-7s %-6s %9s %10s %10s %10s" %
       ("variant", "config", "fit", "bpw", "mean KL", "rel_err", "wtd_err"))
print(hdr)
print("-" * len(hdr))
for r in sorted(rows3, key=lambda x: (x.get("bpw_total", 0.0), x["name"])):
    print("%-26s %-7s %-6s %9.4f %10s %10s %10s" % (
        r["name"][:26], str(r.get("config", "-"))[:7],
        str(r.get("fit_weighting", "none"))[:6],
        r.get("bpw_total", 0.0), num(r.get("mean_kl"), 6),
        num(r.get("mean_rel_err")), num(r.get("mean_weighted_rel_err"))))

# Pair each weighted row with the unweighted row at the same bits.
pairs = []
by_name = {r["name"]: r for r in rows3}
for r in rows3:
    if r.get("fit_weighting") != "awq":
        continue
    cfg = str(r.get("config", ""))
    twin = by_name.get(r["name"].replace(cfg + "_", cfg[:-1] + "_", 1)) \
        if cfg.endswith("w") else None
    if twin is not None:
        pairs.append((twin, r))
if pairs:
    print("")
    print("paired at identical bpw (the whole point: zero bit cost):")
    print("")
    hdr2 = ("%-26s %9s %12s %12s %9s" %
            ("variant pair", "bpw", "KL unwtd", "KL wtd", "delta"))
    print(hdr2)
    print("-" * len(hdr2))
    for u, w in pairs:
        ku, kw = u.get("mean_kl"), w.get("mean_kl")
        if ku is None or kw is None or not ku:
            delta = "-"
        else:
            delta = "%+.1f%%" % (100.0 * (kw - ku) / ku)
        print("%-26s %9.4f %12s %12s %9s" % (
            u["name"][:26], u.get("bpw_total", 0.0), num(ku, 6), num(kw, 6),
            delta))
else:
    print("")
    print("No unweighted twin rows at this seed count in")
    print("results/Qwen3-0.6B/phase3_results.json, so there is no strict")
    print("same-layout pairing. For c8p3 the Phase 2 rows below are a valid")
    print("equal-bits reference; for c12p4 there is none - rerun with")
    print("-WithUnweightedTwin to produce it.")

if rows2:
    print("")
    print("Phase 2 reference rows (Qwen3-0.6B, c8p3, unweighted, 65535 seeds):")
    print("")
    for r in sorted(rows2, key=lambda x: x.get("bpw_total", 0.0)):
        print("  %-22s bpw %7.4f   mean KL %s   rel_err %s" % (
            r["name"], r.get("bpw_total", 0.0), num(r.get("mean_kl"), 6),
            num(r.get("mean_rel_err"))))
    print("")
    print("  CAVEAT: those Phase 2 rows were measured in the legacy layout with")
    print("  eval_mode=dual, this run uses the Phase 3 layout with")
    print("  eval_mode=cached. The equivalence AC pins those two to within 1e-6")
    print("  mean KL on this model, and the stack is the same laptop, so they")
    print("  are comparable - but they are not one gate computation and must")
    print("  never be merged into one.")
print("")
print("Read the probe like this: KL down a lot at identical bpw means the error")
print("STRUCTURE was the problem and the W2/W3 follow-up (rotation + compensation) is")
print("warranted. Barely moving means the negative result stands, with the")
print("obvious objection now pre-answered. The decision number lives at 1.7B")
print("(pod), not here; 0.6B only shows whether the machinery works at all.")
print("")
'@

Write-Host ""
Write-Host "=== W1 comparison ==="
if ($DryRun) {
    Write-Host "(-DryRun: would pipe the comparison script to `"$py - $NSeeds`")"
} else {
    $table | & $py - $NSeeds 2>&1 | Tee-Object -FilePath $log -Append
}

$totalH = [math]::Round(((Get-Date) - $t0).TotalHours, 2)
Write-Host "total wall clock: $totalH h"
Write-Host "artifacts: results\$slug\{phase3_results.json, summary.md, run.log, w1.log}"
Write-Host "no upload performed (W1 is a local probe)."
Write-Host "finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
exit 0
