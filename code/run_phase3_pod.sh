#!/usr/bin/env bash
# run_phase3_pod.sh - SeedLM+O Phase 3 Leg B launcher (rented RTX 4090 24 GB).
#
# Assumes a pod with Ubuntu + CUDA (RunPod PyTorch 2.x template), a 150 GB volume at
# /workspace, HF_TOKEN exported, and this script pulled into
# /workspace/harness/ from the private results repo. Run it under tmux:
#
#     tmux new -s phase3
#     export HF_TOKEN=<fine-grained token>
#     bash harness/run_phase3_pod.sh 2>&1 | tee phase3.log
#     # Ctrl-B then D to detach; tmux attach -t phase3 to come back
#
# WHAT IT DOES, in order (every step is idempotent and resumable - the pod can
# die at any point, the volume survives, and rerunning this script picks up
# exactly where it stopped, because fits are cached per tensor and the heavy
# external artifacts are guarded by marker files under .state/):
#
#   0. deps: uv venv + torch/transformers/gguf/lm-eval, huggingface_hub CLI
#   1. build llama.cpp (convert_hf_to_gguf.py, llama-imatrix, llama-quantize)
#   2. Qwen3-1.7B FIRST (Leg A takeover): the test laptop hit reproducible
#      hardware faults during c12p4 fits and never finished Leg A. With LEGA_ON_POD=1 (default) this pod runs 1.7B stage 1
#      itself - both configs, fit + eval - and picks the winner on its own
#      stack (fits are deterministic from generator_seed=3407, nothing needs
#      to cross machines). LEGA_ON_POD=0 restores the old behavior (pull the
#      laptop's winner from the results repo).
#   3. Qwen3-1.7B comparators + stage 2: winner refit @65535, comparators
#      (Q4_K_M/Q3_K_M imatrix + AWQ), re-eval every verdict row on this stack,
#      verdict. Then an intermediate upload, so the 1.7B verdict is safe on
#      the repo even if the pod dies later. Running the 1.7B leg before the
#      8B leg produces the cheap go/no-go signal (c12p4 vs Q3_K_M at ~3 bpw)
#      before ~14 pod-hours of 8B work; set STOP_AFTER_17B=1 to halt there
#      and review the verdict before paying for the 8B leg.
#   4. download Qwen3-8B; stage 1: --stage fit + --stage eval for c8p3, c12p4
#   5. Qwen3-8B comparators: f16 GGUF -> imatrix (wikitext) -> Q4_K_M + Q3_K_M,
#      plus AWQ W4A16 g128 via llmcompressor; all dequantized to bf16 and
#      swapped into the same torch model as our own variants (the same-stack rule forbids
#      comparing a llama.cpp runtime against a torch runtime, so we compare the
#      WEIGHTS, not the runtimes)
#   6. Qwen3-8B stage 2: winner refit @65535 + re-eval + lm-eval + verdict
#   7. upload results/ to the private HF repo, print a per-stage wall clock
#
# COMPARATOR SOURCE (added 2026-08-17 after the Leg A takeover
# post-mortem): COMPARATOR_SOURCE=build (default) builds the K-quants
# here with llama.cpp + an imatrix calibrated on wikitext. COMPARATOR_SOURCE=hf
# skips llama.cpp entirely and downloads pre-quantized GGUFs from
# COMPARATOR_HF_REPO instead. The prebuilt route is the escape hatch when the
# local convert/quantize toolchain breaks; its K-quants are
# THIRD-PARTY-calibrated, so comparator_provenance.json records the source repo
# and per-file sha256 and the report must state which source produced them.
#
# BUDGET (4090 ~= 2.2x the 5070 anchor of 66 s / 2M
# params @65535 seeds): 1.7B stage 1 both configs ~2 h (Leg A takeover),
# 8B stage 1 ~6-8 h, 8B stage 2 winner-only ~8-10 h, comparators ~1.5 h,
# evals ~1 h, lm-eval ~1 h, 1.7B stage 2 ~1.5 h
# => ~21-25 pod-hours. TERMINATE THE POD when the DONE
# banner appears - an idle GPU bills the same as
# a busy one.
#
# Exits nonzero on the first stage failure so the tmux scrollback shows where.

set -euo pipefail

# ------------------------------------------------------------------ settings
WORK="${WORK:-/workspace}"
HARNESS="${HARNESS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
RESULTS_REPO="${RESULTS_REPO:-<your-hf-user>/seedlm-o-results}"
MODEL_8B="${MODEL_8B:-Qwen/Qwen3-8B}"
MODEL_17B="${MODEL_17B:-Qwen/Qwen3-1.7B}"
SLUG_8B="Qwen3-8B"
SLUG_17B="Qwen3-1.7B"
STAGE1_SEEDS="${STAGE1_SEEDS:-16384}"
STAGE2_SEEDS="${STAGE2_SEEDS:-65535}"
LMEVAL_TASKS="${LMEVAL_TASKS:-gsm8k,ifeval}"
LMEVAL_LIMIT="${LMEVAL_LIMIT:-200}"
# Leg A takeover: 1 = run 1.7B stage 1 (both configs) on this pod and pick the
# winner here; 0 = pull the laptop's winner from the results repo (pre-crash
# behavior). See header step 2.
LEGA_ON_POD="${LEGA_ON_POD:-1}"
# 1 = exit cleanly after the 1.7B leg + upload, so the 1.7B verdict can be
# reviewed before committing ~14 pod-hours to the 8B leg. Rerun with 0 to
# continue; completed stages are cached/marker-guarded and skip instantly.
STOP_AFTER_17B="${STOP_AFTER_17B:-0}"
# The default calibration corpus for the imatrix. Recorded in the
# provenance file so the report can state which corpus produced the K-quants.
IMATRIX_CORPUS_URL="${IMATRIX_CORPUS_URL:-https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip}"
IMATRIX_CORPUS_NAME="wikitext-2-raw-v1 (test split, llama.cpp default)"
# Where the Q4_K_M / Q3_K_M comparators come from:
#   build = convert_hf_to_gguf.py -> llama-imatrix -> llama-quantize (default,
#           self-built and calibrated on IMATRIX_CORPUS_URL)
#   hf    = download pre-quantized GGUFs; llama.cpp is never built or invoked.
# The hf route exists because a convert/quantize break must not be able to
# block a paid run a second time (2026-08-16: the build failed silently, the
# marker got stamped anyway, and the verdict came out "comparators available:
# none"). Prebuilt K-quants are calibrated by whoever published them - the
# provenance file records repo + filenames + sha256 so the report can say so.
COMPARATOR_SOURCE="${COMPARATOR_SOURCE:-build}"
# Empty repo = derive per model: unsloth/{slug}-GGUF, i.e.
# unsloth/Qwen3-1.7B-GGUF for the 1.7B leg and unsloth/Qwen3-8B-GGUF for the
# 8B leg. Set it explicitly to pin one repo for both legs.
COMPARATOR_HF_REPO="${COMPARATOR_HF_REPO:-}"
# Filenames inside that repo. "{slug}" expands to Qwen3-1.7B / Qwen3-8B, so one
# setting works for both legs; a literal name (no {slug}) is also accepted.
COMPARATOR_HF_Q4_FILE="${COMPARATOR_HF_Q4_FILE:-}"
COMPARATOR_HF_Q3_FILE="${COMPARATOR_HF_Q3_FILE:-}"
[ -n "${COMPARATOR_HF_Q4_FILE}" ] || COMPARATOR_HF_Q4_FILE='{slug}-Q4_K_M.gguf'
[ -n "${COMPARATOR_HF_Q3_FILE}" ] || COMPARATOR_HF_Q3_FILE='{slug}-Q3_K_M.gguf'

case "${COMPARATOR_SOURCE}" in
    build|hf) ;;
    *) printf 'FATAL: COMPARATOR_SOURCE must be "build" or "hf" (got "%s")\n' \
           "${COMPARATOR_SOURCE}" >&2; exit 1 ;;
esac

export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Data root override: harness code lives in ${WORK}/harness but cache/results/
# models must live on the volume root, where this script creates and uploads
# them. Without this, runner.py anchors its layout to the harness dir.
export SEEDLM_ROOT="${WORK}"

STATE="${WORK}/.state"
MODELS="${WORK}/models/originals"
COMPARATORS="${WORK}/comparators"
LLAMA="${WORK}/llama.cpp"
VENV="${WORK}/.venv"
PY="${VENV}/bin/python"
TIMINGS="${WORK}/stage_timings.tsv"

mkdir -p "${STATE}" "${MODELS}" "${COMPARATORS}" "${WORK}/results" "${WORK}/cache"
: >>"${TIMINGS}"

RUN_T0=$(date +%s)

# ----------------------------------------------------------------- helpers
log() { printf '%s | %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S')" "$*"; }

die() { log "FATAL: $*"; exit 1; }

# Record a stage's wall clock so the cost of the pod is attributable per stage.
timed() {
    local name="$1"; shift
    local t0 t1 rc
    t0=$(date +%s)
    log "=== BEGIN ${name} ==="
    set +e
    "$@"
    rc=$?
    set -e
    t1=$(date +%s)
    printf '%s\t%s\t%s\n' "${name}" "$((t1 - t0))" "${rc}" >>"${TIMINGS}"
    if [ "${rc}" -ne 0 ]; then
        log "=== FAILED ${name} after $((t1 - t0))s (exit ${rc}) ==="
        exit "${rc}"
    fi
    log "=== END ${name} in $(( (t1 - t0) / 60 )) min ==="
}

# Marker-guarded step: only the expensive *external* artifacts need this. The
# runner's own stages are internally resumable and are always re-entered.
once() {
    local marker="${STATE}/$1"; shift
    if [ -f "${marker}" ]; then
        log "SKIP $(basename "${marker}") (already done)"
        return 0
    fi
    # timed() runs us under `set +e`, so `set -e` is INERT here: without the
    # explicit `|| return 1` a failing stage would fall through and touch the
    # marker anyway. That is exactly how cmp_17b.done got stamped over a
    # comparator build that produced nothing.
    "$@" || return 1
    touch "${marker}" || return 1
}

# ---------------------------------------------------------- failure plumbing
# EVERY stage function below runs through timed(), which wraps the call in
# `set +e ... set -e`. Inside those functions `set -e` does nothing, so a
# failing middle command does NOT stop the function - execution falls through
# to whatever comes next and the function returns the exit status of its LAST
# command. Failure therefore has to be propagated by hand, on every step that
# must succeed:
#
#     run_step "what this is" cmd arg arg || return 1
#     some_pipeline_or_redirect || { log "FATAL ..."; return 1; }
#
# Soft-fail steps (awq, lm-eval, the Leg A pull) deliberately do NOT do this;
# each one says so at the call site.
run_step() {
    local what="$1"; shift
    local rc=0
    log "  step: ${what}"
    "$@" || rc=$?
    if [ "${rc}" -ne 0 ]; then
        log "  STEP FAILED (exit ${rc}): ${what}"
        return 1
    fi
    return 0
}

# Marker hygiene / self-healing (BRIEF item 3). A `once` marker is only honored
# while the artifacts it certifies still exist and are nonzero; otherwise the
# marker is a lie and gets deleted so the guarded stage reruns. This clears the
# falsely-stamped cmp_17b.done on the existing volume with no manual `rm`.
verify_marker() {
    local name="$1"; shift
    local marker="${STATE}/${name}"
    local f stale=0
    [ -f "${marker}" ] || return 0
    for f in "$@"; do
        if [ ! -s "${f}" ]; then
            log "MARKER STALE: ${name} certifies ${f} - missing or zero bytes"
            stale=1
        fi
    done
    if [ "${stale}" -eq 1 ]; then
        rm -f "${marker}"
        log "MARKER SELF-HEAL: removed ${marker}; the guarded stage will rerun"
    fi
    return 0
}

# Drop a stale cmp_*.done for one model slug (both K-quants must be present).
verify_comparator_marker() {
    local marker="$1" slug="$2"
    verify_marker "${marker}" \
        "${COMPARATORS}/${slug}/Q4_K_M.gguf" \
        "${COMPARATORS}/${slug}/Q3_K_M.gguf"
}

sha256_of() {
    sha256sum "$1" 2>/dev/null | awk '{print $1}' || true
}

runner() {
    (cd "${WORK}" && "${PY}" "${HARNESS}/runner.py" "$@")
}

# ------------------------------------------------------------------- step 0
setup_deps() {
    if ! command -v uv >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh \
            || { log "FATAL uv installer failed"; return 1; }
    fi
    export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
    if [ ! -d "${VENV}" ]; then
        run_step "uv venv 3.12" uv venv --python 3.12 "${VENV}" || return 1
    fi
    run_step "pip install torch" \
        uv pip install --python "${PY}" torch --torch-backend=auto || return 1
    # langdetect/immutabledict are ifeval task deps lm-eval does not pull.
    # sentencepiece: convert_hf_to_gguf.py imports it for Qwen vocab export -
    # its absence was the silent comparator-build killer on 2026-08-16/17.
    run_step "pip install harness deps" \
        uv pip install --python "${PY}" \
            transformers accelerate safetensors numpy matplotlib \
            "huggingface_hub[cli,hf_transfer]" gguf lm-eval llmcompressor \
            langdetect immutabledict sentencepiece || return 1
    run_step "torch/cuda smoke test" \
        "${PY}" -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")' \
        || return 1
    if [ -z "${HF_TOKEN:-}" ]; then
        log "FATAL HF_TOKEN is not set (export a fine-grained HF token first)"
        return 1
    fi
    # No `hf auth login`: huggingface_hub reads HF_TOKEN directly, and the CLI
    # entry-point name has churned across releases (huggingface-cli -> hf, and
    # huggingface_hub.commands was removed in 1.x). Verify the token instead.
    run_step "verify HF_TOKEN" \
        "${PY}" -c 'import os, huggingface_hub as h; print("hf_hub", h.__version__, h.HfApi(token=os.environ["HF_TOKEN"]).whoami()["name"])' \
        || return 1
}

selftest() {
    runner --selftest || return 1
}

# ------------------------------------------------------------------- step 1
build_llama() {
    if [ ! -d "${LLAMA}/.git" ]; then
        run_step "git clone llama.cpp" \
            git clone --depth 1 https://github.com/ggml-org/llama.cpp "${LLAMA}" \
            || return 1
    fi
    run_step "cmake configure (CUDA)" \
        cmake -S "${LLAMA}" -B "${LLAMA}/build" \
            -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON || return 1
    run_step "cmake build llama-quantize llama-imatrix" \
        cmake --build "${LLAMA}/build" --config Release -j "$(nproc)" \
            --target llama-quantize llama-imatrix || return 1
    run_step "pip install llama.cpp requirements" \
        uv pip install --python "${PY}" -r "${LLAMA}/requirements.txt" || return 1
    if [ ! -x "${LLAMA}/build/bin/llama-quantize" ]; then
        log "FATAL llama-quantize not built"
        log "      fallback: rerun with COMPARATOR_SOURCE=hf (no llama.cpp needed)"
        return 1
    fi
    if [ ! -x "${LLAMA}/build/bin/llama-imatrix" ]; then
        log "FATAL llama-imatrix not built"
        log "      fallback: rerun with COMPARATOR_SOURCE=hf (no llama.cpp needed)"
        return 1
    fi
    if [ ! -f "${LLAMA}/convert_hf_to_gguf.py" ]; then
        log "FATAL ${LLAMA}/convert_hf_to_gguf.py missing"
        log "      fallback: rerun with COMPARATOR_SOURCE=hf (no llama.cpp needed)"
        return 1
    fi
}

fetch_imatrix_corpus() {
    local zip="${WORK}/wikitext.zip"
    run_step "download imatrix corpus" \
        curl -fL -o "${zip}" "${IMATRIX_CORPUS_URL}" || return 1
    (cd "${WORK}" && unzip -o "${zip}") \
        || { log "FATAL unzip of ${zip} failed"; return 1; }
    if [ ! -s "${WORK}/wikitext-2-raw/wiki.test.raw" ]; then
        log "FATAL calibration corpus missing after unzip"
        return 1
    fi
}

# ------------------------------------------------------------------- step 2
download_model() {
    local repo="$1" slug="$2"
    run_step "snapshot_download ${repo}" \
        "${PY}" -c 'import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], local_dir=sys.argv[2])' \
        "${repo}" "${MODELS}/${slug}" || return 1
    if [ ! -s "${MODELS}/${slug}/config.json" ]; then
        log "FATAL ${MODELS}/${slug}/config.json missing after download"
        return 1
    fi
}

# --------------------------------------------------------- comparator build
# Produce Q4_K_M and Q3_K_M plus an AWQ W4A16 g128 checkpoint for one model, in
# the layout runner.py auto-discovers:
#   comparators/{slug}/Q4_K_M.gguf, Q3_K_M.gguf, awq/
#
# Control flow contract: this runs under
# timed()'s `set +e`, so every step propagates by hand, the K-quants are
# asserted nonzero BEFORE any provenance is written, and a failure prints a
# loud FATAL with the COMPARATOR_SOURCE=hf fallback instead of returning 0.

# unsloth/{slug}-GGUF unless COMPARATOR_HF_REPO pins one.
comparator_hf_repo() {
    local slug="$1"
    if [ -n "${COMPARATOR_HF_REPO}" ]; then
        printf '%s' "${COMPARATOR_HF_REPO}"
    else
        printf 'unsloth/%s-GGUF' "${slug}"
    fi
}

# Expand the "{slug}" placeholder in a configured GGUF filename.
comparator_hf_file() {
    local slug="$1" quant="$2" tmpl
    case "${quant}" in
        Q4_K_M) tmpl="${COMPARATOR_HF_Q4_FILE}" ;;
        Q3_K_M) tmpl="${COMPARATOR_HF_Q3_FILE}" ;;
        *) return 1 ;;
    esac
    printf '%s' "${tmpl//\{slug\}/${slug}}"
}

# Both K-quants must exist and be nonzero. Everything downstream (provenance,
# the cmp_*.done marker, the comparator eval, the verdict) depends on this.
assert_kquants() {
    local slug="$1" out="${COMPARATORS}/${slug}" q bad=0
    for q in Q4_K_M Q3_K_M; do
        if [ ! -s "${out}/${q}.gguf" ]; then
            log "FATAL comparator artifact missing or zero bytes: ${out}/${q}.gguf"
            bad=1
        else
            log "OK comparator ${q}.gguf ($(stat -c %s "${out}/${q}.gguf") bytes)"
        fi
    done
    return "${bad}"
}

# The loud pointer the 2026-08-16 run did not print.
comparator_fatal() {
    local slug="$1" src="$2"
    log "FATAL comparator stage failed for ${slug} (COMPARATOR_SOURCE=${src})."
    log "      No provenance was written and no cmp marker was stamped, so a"
    log "      rerun retries this stage instead of skipping it."
    if [ "${src}" = "build" ]; then
        log "      FALLBACK: rerun with COMPARATOR_SOURCE=hf to download"
        log "      pre-quantized GGUFs and skip llama.cpp entirely:"
        log "        COMPARATOR_SOURCE=hf STOP_AFTER_17B=1 bash harness/run_phase3_pod.sh"
    else
        log "      Check COMPARATOR_HF_REPO ($(comparator_hf_repo "${slug}")),"
        log "      COMPARATOR_HF_Q4_FILE/COMPARATOR_HF_Q3_FILE and HF_TOKEN;"
        log "      or rerun with COMPARATOR_SOURCE=build."
    fi
}

# --- source: build (llama.cpp, imatrix-calibrated) -------------------------
comparators_from_build() {
    local slug="$1"
    local src="${MODELS}/${slug}"
    local out="${COMPARATORS}/${slug}"
    local f16="${out}/f16.gguf"
    local imat="${out}/imatrix.dat"
    local q

    # Preconditions - fail here rather than three commands later.
    [ -s "${src}/config.json" ] \
        || { log "FATAL model dir not downloaded: ${src}"; return 1; }
    [ -f "${LLAMA}/convert_hf_to_gguf.py" ] \
        || { log "FATAL missing ${LLAMA}/convert_hf_to_gguf.py"; return 1; }
    [ -x "${LLAMA}/build/bin/llama-imatrix" ] \
        || { log "FATAL missing ${LLAMA}/build/bin/llama-imatrix"; return 1; }
    [ -x "${LLAMA}/build/bin/llama-quantize" ] \
        || { log "FATAL missing ${LLAMA}/build/bin/llama-quantize"; return 1; }
    [ -s "${WORK}/wikitext-2-raw/wiki.test.raw" ] \
        || { log "FATAL missing imatrix corpus ${WORK}/wikitext-2-raw/wiki.test.raw"; return 1; }

    # Each step: skip only on a NONZERO artifact (a truncated leftover from a
    # killed pod is deleted and redone), and delete the output on failure so
    # the next run does not inherit a half-written file.
    if [ ! -s "${f16}" ]; then
        rm -f "${f16}"
        run_step "convert_hf_to_gguf ${slug} -> f16" \
            "${PY}" "${LLAMA}/convert_hf_to_gguf.py" "${src}" \
                --outfile "${f16}" --outtype f16 \
            || { rm -f "${f16}"; return 1; }
        [ -s "${f16}" ] \
            || { log "FATAL convert_hf_to_gguf wrote no f16 GGUF"; return 1; }
    fi
    if [ ! -s "${imat}" ]; then
        rm -f "${imat}"
        run_step "llama-imatrix ${slug} (100 chunks wikitext)" \
            "${LLAMA}/build/bin/llama-imatrix" -m "${f16}" \
                -f "${WORK}/wikitext-2-raw/wiki.test.raw" \
                -o "${imat}" -ngl 999 --chunks 100 \
            || { rm -f "${imat}"; return 1; }
        [ -s "${imat}" ] \
            || { log "FATAL llama-imatrix wrote no ${imat}"; return 1; }
    fi
    for q in Q4_K_M Q3_K_M; do
        if [ ! -s "${out}/${q}.gguf" ]; then
            rm -f "${out}/${q}.gguf"
            run_step "llama-quantize ${slug} ${q}" \
                "${LLAMA}/build/bin/llama-quantize" --imatrix "${imat}" \
                    "${f16}" "${out}/${q}.gguf" "${q}" \
                || { rm -f "${out}/${q}.gguf"; return 1; }
        fi
    done
    return 0
}

# --- source: hf (pre-quantized, third-party calibrated) --------------------
# NOTE: llama.cpp is never built or invoked on this path. The K-quants were
# calibrated by whoever published them (imatrix corpus unknown to us), which is
# why the provenance records source repo + sha256 and reports "n/a (prebuilt
# GGUF from <repo>)" for the corpus. The report MUST state which source
# produced the K-quants.
comparators_from_hf() {
    local slug="$1"
    local out="${COMPARATORS}/${slug}"
    local repo file dest q
    repo="$(comparator_hf_repo "${slug}")"
    log "COMPARATOR_SOURCE=hf: pulling prebuilt K-quants from ${repo}"
    for q in Q4_K_M Q3_K_M; do
        dest="${out}/${q}.gguf"
        if [ -s "${dest}" ]; then
            log "  have ${dest} ($(stat -c %s "${dest}") bytes)"
            continue
        fi
        rm -f "${dest}"
        file="$(comparator_hf_file "${slug}" "${q}")" \
            || { log "FATAL no filename template for ${q}"; return 1; }
        log "  step: hf_hub_download ${repo}/${file}"
        if ! "${PY}" - "${repo}" "${file}" "${out}" "${dest}" <<'PYHF'
import os
import sys

from huggingface_hub import hf_hub_download

repo, filename, outdir, dest = sys.argv[1:5]
path = hf_hub_download(repo_id=repo, filename=filename, local_dir=outdir,
                       token=os.environ.get("HF_TOKEN"))
if os.path.abspath(path) != os.path.abspath(dest):
    os.replace(path, dest)
print("downloaded %s/%s -> %s (%d bytes)"
      % (repo, filename, dest, os.path.getsize(dest)))
PYHF
        then
            log "  STEP FAILED: hf_hub_download ${repo}/${file}"
            rm -f "${dest}"
            return 1
        fi
        if [ ! -s "${dest}" ]; then
            log "FATAL download produced no ${dest}"
            return 1
        fi
    done
    return 0
}

# --- soft: AWQ -------------------------------------------------------------
# Deliberately soft-fail: awq_w4 is an informational comparator row - the gates
# use q4km/q3km only. An OOM or llmcompressor version break at hour N must not
# kill the unattended run. (A failed attempt leaves no out/awq dir, so a manual
# rerun after `rm ${STATE}/cmp_*.done` retries it.)
awq_soft() {
    local slug="$1"
    local src="${MODELS}/${slug}"
    local out="${COMPARATORS}/${slug}"
    if [ -d "${out}/awq" ]; then
        return 0
    fi
    if ! "${PY}" "${HARNESS}/pod_awq_quantize.py" \
        --model-dir "${src}" --out-dir "${out}/awq" --group-size 128; then
        log "WARN awq quantize failed for ${slug} - continuing without awq_w4"
        rm -rf "${out}/awq"
    fi
    return 0
}

# --- provenance ------------------------------------------------------------
# Only ever reached after assert_kquants passed, so every stat below has a
# file to stat.
write_comparator_provenance() {
    local slug="$1"
    local out="${COMPARATORS}/${slug}"
    local f16="${out}/f16.gguf"
    local corpus corpus_url repo llama_commit f16_bytes awq_note
    local dest="${WORK}/results/${slug}/comparator_provenance.json"

    if [ "${COMPARATOR_SOURCE}" = "hf" ]; then
        repo="$(comparator_hf_repo "${slug}")"
        corpus="n/a (prebuilt GGUF from ${repo})"
        corpus_url="n/a"
        llama_commit="n/a (llama.cpp not used)"
    else
        repo="n/a (built on this pod)"
        corpus="${IMATRIX_CORPUS_NAME}"
        corpus_url="${IMATRIX_CORPUS_URL}"
        llama_commit="$(git -C "${LLAMA}" rev-parse HEAD 2>/dev/null || echo unknown)"
    fi
    if [ -s "${f16}" ]; then
        f16_bytes="$(stat -c %s "${f16}")"
    else
        f16_bytes="null"
    fi
    if [ -d "${out}/awq" ]; then
        awq_note="llmcompressor W4A16 group_size=128"
    else
        awq_note="absent (quantize failed or skipped)"
    fi

    mkdir -p "${WORK}/results/${slug}" || return 1
    {
        printf '{\n'
        printf '  "comparator_source": "%s",\n' "${COMPARATOR_SOURCE}"
        printf '  "source_repo": "%s",\n' "${repo}"
        printf '  "imatrix_corpus": "%s",\n' "${corpus}"
        printf '  "imatrix_corpus_url": "%s",\n' "${corpus_url}"
        printf '  "llama_cpp_commit": "%s",\n' "${llama_commit}"
        printf '  "awq": "%s",\n' "${awq_note}"
        printf '  "gguf_sha256": {\n'
        printf '    "Q4_K_M": "%s",\n' "$(sha256_of "${out}/Q4_K_M.gguf")"
        printf '    "Q3_K_M": "%s"\n'  "$(sha256_of "${out}/Q3_K_M.gguf")"
        printf '  },\n'
        printf '  "gguf_bytes": {\n'
        printf '    "f16": %s,\n' "${f16_bytes}"
        printf '    "Q4_K_M": %s,\n' "$(stat -c %s "${out}/Q4_K_M.gguf")"
        printf '    "Q3_K_M": %s\n' "$(stat -c %s "${out}/Q3_K_M.gguf")"
        printf '  }\n'
        printf '}\n'
    } >"${dest}" || { log "FATAL could not write ${dest}"; return 1; }
    log "wrote ${dest} (source=${COMPARATOR_SOURCE})"
}

build_comparators() {
    local slug="$1"
    mkdir -p "${COMPARATORS}/${slug}" || return 1

    case "${COMPARATOR_SOURCE}" in
        build)
            comparators_from_build "${slug}" \
                || { comparator_fatal "${slug}" build; return 1; }
            ;;
        hf)
            comparators_from_hf "${slug}" \
                || { comparator_fatal "${slug}" hf; return 1; }
            ;;
        *)
            log "FATAL unknown COMPARATOR_SOURCE=${COMPARATOR_SOURCE}"
            return 1
            ;;
    esac

    # Hard gate: no provenance, no marker, no downstream eval unless BOTH
    # K-quants exist with nonzero size.
    assert_kquants "${slug}" \
        || { comparator_fatal "${slug}" "${COMPARATOR_SOURCE}"; return 1; }

    awq_soft "${slug}"
    write_comparator_provenance "${slug}" || return 1
}

# ------------------------------------------------------- per-model pipelines
stage1_both_configs() {
    local slug="$1"
    local cfg
    for cfg in c8p3 c12p4; do
        run_step "${slug} fit ${cfg} @${STAGE1_SEEDS}" \
            runner --stage fit  --model-dir "${MODELS}/${slug}" --config "${cfg}" \
                --n-seeds "${STAGE1_SEEDS}" --eval-mode cached || return 1
        run_step "${slug} eval ${cfg} @${STAGE1_SEEDS}" \
            runner --stage eval --model-dir "${MODELS}/${slug}" --config "${cfg}" \
                --n-seeds "${STAGE1_SEEDS}" --eval-mode cached || return 1
    done
}

eval_comparators() {
    local slug="$1"
    local q
    # runner.py WARN-skips a missing GGUF and exits 0 (correct for the laptop
    # leg, fatal for this one - that is how the pod produced an INCOMPLETE
    # verdict). Refuse to even start the eval without both artifacts.
    for q in Q4_K_M Q3_K_M; do
        if [ ! -s "${COMPARATORS}/${slug}/${q}.gguf" ]; then
            log "FATAL ${COMPARATORS}/${slug}/${q}.gguf missing or zero bytes;"
            log "      refusing to run a comparator eval that would WARN-skip it."
            return 1
        fi
    done
    local args=(--stage comparators --model-dir "${MODELS}/${slug}"
                --gguf-q4km "${COMPARATORS}/${slug}/Q4_K_M.gguf"
                --gguf-q3km "${COMPARATORS}/${slug}/Q3_K_M.gguf"
                --eval-mode cached)
    # awq is optional (see awq_soft).
    if [ -d "${COMPARATORS}/${slug}/awq" ]; then
        args+=(--awq-dir "${COMPARATORS}/${slug}/awq")
    else
        log "WARN evaluating comparators without awq_w4 for ${slug}"
    fi
    run_step "${slug} comparator eval" runner "${args[@]}" || return 1
}

# Verdict guard (BRIEF item 5). eval_comparators must have landed a q4km AND a
# q3km row in phase3_results.json; without them --stage verdict reports
# "INCOMPLETE - comparators available: none", which is the single output this
# pod run exists to avoid. Checked BEFORE stage 2 and before any upload.
assert_comparator_rows() {
    local slug="$1"
    local json="${WORK}/results/${slug}/phase3_results.json"
    if [ ! -f "${json}" ]; then
        die "verdict guard: ${json} does not exist after the comparator eval"
    fi
    if ! "${PY}" - "${json}" <<'PYROWS'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as fh:
    doc = json.load(fh)
names = {v.get("name") for v in doc.get("variants", [])}
missing = sorted({"q4km", "q3km"} - names)
if missing:
    print("missing comparator rows: " + ", ".join(missing), file=sys.stderr)
    print("rows present: " + ", ".join(sorted(n for n in names if n)),
          file=sys.stderr)
    raise SystemExit(1)
print("verdict guard: q4km and q3km rows present in " + path)
PYROWS
    then
        log "The comparator eval exited 0 but wrote no q4km/q3km rows, so the"
        log "verdict would be INCOMPLETE. Inspect the eval output above, then:"
        log "  rm -f ${STATE}/cmp_*.done   # force a comparator rebuild"
        log "  COMPARATOR_SOURCE=hf bash harness/run_phase3_pod.sh   # or fall back"
        die "verdict guard: no comparator rows in ${json} for ${slug}"
    fi
}

# Refit the winner at full seed budget and re-measure every verdict row on THIS
# stack. `--stage eval` re-evaluates the whole matrix at STAGE2_SEEDS, which is
# what keeps the gate single-stack even when the winner was chosen elsewhere.
stage2_and_verdict() {
    local slug="$1" cfg="$2"
    run_step "${slug} refit-full ${cfg} @${STAGE2_SEEDS}" \
        runner --stage refit-full --model-dir "${MODELS}/${slug}" --config "${cfg}" \
            --n-seeds "${STAGE1_SEEDS}" --stage2-seeds "${STAGE2_SEEDS}" || return 1
    run_step "${slug} eval ${cfg} @${STAGE2_SEEDS}" \
        runner --stage eval --model-dir "${MODELS}/${slug}" --config "${cfg}" \
            --n-seeds "${STAGE2_SEEDS}" --eval-mode cached || return 1
    run_step "${slug} verdict ${cfg}" \
        runner --stage verdict --model-dir "${MODELS}/${slug}" --config "${cfg}" \
        || return 1
}

# Read `WINNER_CONFIG=` out of a results dir the laptop produced.
winner_config_of() {
    local slug="$1" line cfg
    line=$(runner --model-dir "${MODELS}/${slug}" --print-winner 2>/dev/null || true)
    cfg=$(printf '%s\n' "${line}" | tr ' ' '\n' | sed -n 's/^WINNER_CONFIG=//p' | head -1)
    printf '%s' "${cfg}"
}

upload_results() {
    # ref_logits/ is ~2 GB of regenerable fp32 tensors per model - never shipped.
    run_step "upload results to ${RESULTS_REPO}" \
        "${PY}" "${HARNESS}/hf_sync.py" upload --repo "${RESULTS_REPO}" \
            --local "${WORK}/results" --path-in-repo results \
            --ignore '*/ref_logits/*' || return 1
}

# =========================================================================
#                                  MAIN
# =========================================================================
log "SeedLM+O Phase 3 Leg B starting on $(hostname)"
nvidia-smi || die "no GPU visible"
df -h "${WORK}" | tail -1

timed "0-deps"          once deps.done          setup_deps
export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
# The selftest hashes the Qwen3-0.6B originals (always present on the laptop,
# absent on a fresh pod volume), so that download must precede it.
timed "0-download-0.6b" once download_06b.done  download_model "Qwen/Qwen3-0.6B" "Qwen3-0.6B"
timed "0-selftest"      selftest
if [ "${COMPARATOR_SOURCE}" = "hf" ]; then
    # Acceptance rule: the hf path never invokes llama.cpp, so neither the
    # build nor its calibration corpus is fetched. Stage names are unchanged,
    # just not entered.
    log "COMPARATOR_SOURCE=hf: skipping 1-llama.cpp and 1-corpus"
    log "  comparators come prebuilt from ${COMPARATOR_HF_REPO:-unsloth/<slug>-GGUF}"
    log "  (third-party-calibrated - recorded in comparator_provenance.json)"
else
    timed "1-llama.cpp"     once llama.done         build_llama
    timed "1-corpus"        once corpus.done        fetch_imatrix_corpus
fi
# ----------------------------------------------------------- Qwen3-1.7B leg
# FIRST (Leg A takeover - see header step 2): the pod picks the 1.7B winner
# on its own stack, and the 1.7B verdict lands on the repo before any 8B
# money is spent.
timed "6-download-1.7b" once download_17b.done download_model "${MODEL_17B}" "${SLUG_17B}"

pull_laptop_17b() {
    "${PY}" "${HARNESS}/hf_sync.py" download --repo "${RESULTS_REPO}" \
        --allow "results/${SLUG_17B}/*" --local "${WORK}/.laptop" || true
    if [ -f "${WORK}/.laptop/results/${SLUG_17B}/phase3_results.json" ]; then
        mkdir -p "${WORK}/results/${SLUG_17B}"
        cp "${WORK}/.laptop/results/${SLUG_17B}/phase3_results.json" \
           "${WORK}/results/${SLUG_17B}/phase3_results.json"
        log "pulled Leg A results for ${SLUG_17B}"
    else
        log "WARN no Leg A results for ${SLUG_17B} in ${RESULTS_REPO}"
    fi
}

if [ "${LEGA_ON_POD}" = "1" ]; then
    # Takeover: stage 1 both configs, fit + eval, on this stack. The winner
    # is then read from rows THIS POD produced.
    timed "6-1.7b-stage1"   stage1_both_configs "${SLUG_17B}"
else
    timed "6-pull-legA"     pull_laptop_17b
fi

WINNER_CFG_17B="$(winner_config_of "${SLUG_17B}")"
if [ -z "${WINNER_CFG_17B}" ] || [ "${WINNER_CFG_17B}" = "-" ]; then
    if [ "${LEGA_ON_POD}" = "1" ]; then
        die "1.7B stage 1 ran on this pod but no winner could be determined"
    fi
    log "WARN no Leg A stage-1 winner for ${SLUG_17B} - SKIPPING the 1.7B"
    log "     stage 2 rather than failing the run. Rerun this script after"
    log "     Leg A has uploaded results/${SLUG_17B}/ to ${RESULTS_REPO}."
else
    log "1.7B stage-1 winner config: ${WINNER_CFG_17B}"
    if [ "${LEGA_ON_POD}" != "1" ]; then
        # Same-stack determinism: Leg A's rows were measured on the local test machine. Drop them
        # before re-measuring, so this pod's verdict is computed purely from
        # rows this pod produced. (NOT done in takeover mode - there the rows
        # are this pod's own stage-1 rows. The laptop's partial rows stay
        # local to the laptop; --stage verdict shows a cross-stack drift table
        # if both are merged later, by hand.)
        rm -f "${WORK}/results/${SLUG_17B}/phase3_results.json"
    fi
    # Self-heal the marker before honoring it: the 2026-08-16 run stamped
    # cmp_17b.done over a build that produced no GGUFs at all.
    verify_comparator_marker cmp_17b.done "${SLUG_17B}"
    timed "6-1.7b-comparators-build" once cmp_17b.done build_comparators "${SLUG_17B}"
    timed "6-1.7b-stage1-seed"  runner --stage fit --model-dir "${MODELS}/${SLUG_17B}" \
        --config "${WINNER_CFG_17B}" --n-seeds "${STAGE2_SEEDS}"
    timed "6-1.7b-comparators-eval" eval_comparators "${SLUG_17B}"
    assert_comparator_rows "${SLUG_17B}"
    timed "6-1.7b-stage2"       stage2_and_verdict "${SLUG_17B}" "${WINNER_CFG_17B}"
fi

# Intermediate upload: the 1.7B verdict is safe on the repo even if the pod
# dies (or is stopped for review) before or during the 8B leg.
timed "7-upload-17b"    upload_results

if [ "${STOP_AFTER_17B}" = "1" ]; then
    log ""
    log "============ STOPPED AFTER 1.7B (STOP_AFTER_17B=1) ============"
    log "Review results/${SLUG_17B}/summary.md (verdict + gates) on"
    log "https://huggingface.co/${RESULTS_REPO} before the 8B leg."
    log "To continue: rerun this script with STOP_AFTER_17B=0. Completed"
    log "stages are cached/marker-guarded and will be skipped."
    log "An idle pod bills like a busy one - Stop (not Terminate) the pod"
    log "if the review will take long; the volume survives a Stop."
    log "==============================================================="
    exit 0
fi

# ------------------------------------------------------------- Qwen3-8B leg
timed "2-download-8b"   once download_8b.done   download_model "${MODEL_8B}" "${SLUG_8B}"
timed "3-8b-stage1"     stage1_both_configs "${SLUG_8B}"
verify_comparator_marker cmp_8b.done "${SLUG_8B}"
timed "4-8b-comparators-build" once cmp_8b.done build_comparators "${SLUG_8B}"
timed "4-8b-comparators-eval"  eval_comparators "${SLUG_8B}"
assert_comparator_rows "${SLUG_8B}"

WINNER_CFG_8B="$(winner_config_of "${SLUG_8B}")"
if [ -z "${WINNER_CFG_8B}" ] || [ "${WINNER_CFG_8B}" = "-" ]; then
    log "WARN could not read an 8B stage-1 winner config; defaulting to c12p4"
    WINNER_CFG_8B="c12p4"
fi
log "8B stage-1 winner config: ${WINNER_CFG_8B}"
timed "5-8b-stage2"     stage2_and_verdict "${SLUG_8B}" "${WINNER_CFG_8B}"

# Soft-fail: lm-eval is reported-not-gating. A task-dep or
# API break after ~18 pod-hours must not abort before the final upload.
lmeval_soft() {
    if ! runner --stage lmeval --model-dir "${MODELS}/${SLUG_8B}" \
        --config "${WINNER_CFG_8B}" --n-seeds "${STAGE2_SEEDS}" \
        --lmeval-tasks "${LMEVAL_TASKS}" --lmeval-limit "${LMEVAL_LIMIT}"; then
        log "WARN lm-eval failed (non-gating) - continuing without benchmark rows"
    fi
}
timed "5-8b-lmeval"     lmeval_soft

# ------------------------------------------------------------------- step 7
timed "7-upload"        upload_results

RUN_T1=$(date +%s)
log ""
log "================= DONE ================="
log "per-stage wall clock (seconds):"
awk -F'\t' '{printf "  %-32s %8d s  (%5.2f h)  exit=%s\n", $1, $2, $2/3600, $3}' "${TIMINGS}"
log "total: $(( (RUN_T1 - RUN_T0) / 60 )) min  ($(awk -v s="$((RUN_T1 - RUN_T0))" 'BEGIN{printf "%.2f", s/3600}') h)"
log "results uploaded to https://huggingface.co/${RESULTS_REPO}"
log ""
log "NOW TERMINATE THE POD:"
log "  1. confirm results/${SLUG_8B}/summary.md is in the repo"
log "  2. Pod page -> Terminate (NOT Stop), delete the volume"
log "  3. revoke the fine-grained HF token"
log "========================================"
