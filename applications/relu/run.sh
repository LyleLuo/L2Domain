#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${APP_DIR}/../.." && pwd)"
source "${ROOT_DIR}/scripts/app_runner.sh"

usage() {
  printf 'Usage: ./run.sh [common options] [--suite main|occupancy|cutlass|all]\n\n'
  sidedgpu_app_print_common_options \
    '  --suite main|occupancy|cutlass|all  Figure 9, occupancy, CUTLASS, or every suite (default: main)'
}

SUITE="main"
parse_app_option() {
  case "$1" in
    --suite)
      sidedgpu_app_require_value "$@"
      SUITE="$2"
      SIDEDGPU_APP_OPTION_CONSUMED=2
      ;;
    *) return 1 ;;
  esac
}

SIDEDGPU_APP_OPTION_PARSER=parse_app_option
sidedgpu_app_parse_options usage "$@"
unset SIDEDGPU_APP_OPTION_PARSER

case "${SUITE}" in main|occupancy|cutlass|all) ;; *) echo "--suite must be main, occupancy, cutlass, or all" >&2; exit 2 ;; esac
case "${PRESET}" in
  quick)
    MAIN_DATA_MBS="16"; MAIN_RUNS=1
    CUTLASS_DATA_MBS="64"; CUTLASS_RUNS=1
    OCCUPANCY_DATA_MB=16; OCCUPANCY_BLOCKS_PER_SM="2"
    ;;
  paper)
    MAIN_DATA_MBS="16 64 256 1024"; MAIN_RUNS=10
    CUTLASS_DATA_MBS="64 128 256 512 1024 2048"; CUTLASS_RUNS=5
    OCCUPANCY_DATA_MB=1024; OCCUPANCY_BLOCKS_PER_SM="1 2 4 8"
    ;;
  *) echo "--preset must be quick or paper" >&2; exit 2 ;;
esac

MAIN_THREADS=128
CACHE_MODE=1
sidedgpu_app_begin "${APP_DIR}" relu "suite=${SUITE}"

case "${GPU_LABEL}:${CUDA_ARCH}" in
  b200:*|custom:sm_100*) CUTLASS_THREADS=128 ;;
  a100:*|h100:*|custom:sm_80|custom:sm_90a) CUTLASS_THREADS=1024 ;;
  *) CUTLASS_THREADS=1024 ;;
esac

CUTLASS_INC="${ROOT_DIR}/thirdparty/cutlass/include"
MAIN_BIN="${BIN_DIR}/relu_main"
PREDICATE_BIN="${BIN_DIR}/relu_persistent_predicate"
CUTLASS_BIN="${BIN_DIR}/relu_cutlass_epilogue_style"

if [[ "${SUITE}" == main || "${SUITE}" == occupancy || "${SUITE}" == all ]]; then
  sidedgpu_run nvcc -std=c++20 -O3 -arch="${CUDA_ARCH}" -I"${ROOT_DIR}/utility" \
    "${APP_DIR}/src/relu_sideaware_benchmark.cu" -o "${MAIN_BIN}"
fi

if [[ "${SUITE}" == cutlass || "${SUITE}" == all ]]; then
  if (( ! DRY_RUN )) && [[ ! -f "${CUTLASS_INC}/cutlass/cutlass.h" ]]; then
    echo "CUTLASS headers not found; run: git submodule update --init thirdparty/cutlass" >&2
    exit 1
  fi
  sidedgpu_run nvcc -std=c++20 -O3 -arch="${CUDA_ARCH}" -I"${ROOT_DIR}/utility" \
    "${APP_DIR}/src/relu_sideaware_predicate.cu" -o "${PREDICATE_BIN}"
  sidedgpu_run nvcc -std=c++20 -O3 -arch="${CUDA_ARCH}" -I"${ROOT_DIR}/utility" -I"${CUTLASS_INC}" \
    "${APP_DIR}/src/relu_cutlass_epilogue_style.cu" -o "${CUTLASS_BIN}"
fi

(( BUILD_ONLY )) && exit 0

for_each_main_case() {
  local callback="$1" data_mb workflow assignment persistent swapped
  for data_mb in ${MAIN_DATA_MBS}; do
    for workflow in persistent green_context; do
      [[ "${workflow}" == persistent ]] && persistent=1 || persistent=0
      for assignment in aware swapped; do
        [[ "${assignment}" == aware ]] && swapped=0 || swapped=1
        "${callback}" "${data_mb}" "${workflow}" "${assignment}" "${persistent}" "${swapped}"
      done
    done
  done
}

for_each_cutlass_case() {
  local callback="$1" data_mb
  for data_mb in ${CUTLASS_DATA_MBS}; do
    "${callback}" "${data_mb}"
  done
}

dry_main_case() {
  sidedgpu_run_gpu "${MAIN_BIN}" "$1" "${MAIN_RUNS}" "${MAIN_THREADS}" "$4" "$5" 1
}

dry_occupancy_case() {
  sidedgpu_run_gpu "${MAIN_BIN}" "${OCCUPANCY_DATA_MB}" "${MAIN_RUNS}" "${MAIN_THREADS}" 0 0 "$1"
}

dry_cutlass_case() {
  sidedgpu_run_gpu "${PREDICATE_BIN}" "$1" "${CUTLASS_RUNS}" "${CUTLASS_THREADS}" "${CACHE_MODE}"
  sidedgpu_run_gpu "${CUTLASS_BIN}" "$1" "${CUTLASS_RUNS}" "${CUTLASS_THREADS}" PERSISTENT_BLOCKS_FROM_PREDICATE 5 "${CACHE_MODE}"
}

if (( DRY_RUN )); then
  [[ "${SUITE}" == main || "${SUITE}" == all ]] && for_each_main_case dry_main_case
  if [[ "${SUITE}" == occupancy || "${SUITE}" == all ]]; then
    for blocks_per_sm in ${OCCUPANCY_BLOCKS_PER_SM}; do
      dry_occupancy_case "${blocks_per_sm}"
    done
  fi
  [[ "${SUITE}" == cutlass || "${SUITE}" == all ]] && for_each_cutlass_case dry_cutlass_case
  exit 0
fi

FAILURES=0

run_main_case() {
  local data_mb="$1" workflow="$2" assignment="$3" persistent="$4" swapped="$5"
  local log status baseline_ms mode_ms speedup correct pass
  log="${OUT_DIR}/.relu_main_${GPU_LABEL}_${data_mb}_${workflow}_${assignment}.log"
  echo "===== Figure 9: data_mb=${data_mb} workflow=${workflow} assignment=${assignment} ====="
  set +e
  sidedgpu_run_gpu "${MAIN_BIN}" "${data_mb}" "${MAIN_RUNS}" "${MAIN_THREADS}" "${persistent}" "${swapped}" 1 2>&1 | tee "${log}"
  status="${PIPESTATUS[0]}"
  set -e
  baseline_ms="$(awk '/Speedup=/{for(i=1;i<=NF;++i)if($i~/^Baseline=/){sub("Baseline=","",$i);print $i;exit}}' "${log}")"
  mode_ms="$(awk '/Speedup=/{for(i=1;i<=NF;++i)if($i~/^Ours=/){sub("Ours=","",$i);print $i;exit}}' "${log}")"
  speedup="$(awk '/Speedup=/{for(i=1;i<=NF;++i)if($i~/^Speedup=/){sub("Speedup=","",$i);sub("x","",$i);print $i;exit}}' "${log}")"
  correct="$(awk '/Baseline vs Ours=/{print $NF;exit}' "${log}")"
  pass="$([[ "${status}" == 0 && "${correct:-}" == Ours=PASS ]] && echo PASS || echo FAIL)"
  echo "${GPU_LABEL},${SIDEDGPU_GPU_INDEX},${data_mb},${MAIN_THREADS},${MAIN_RUNS},${workflow},${assignment},${baseline_ms},${mode_ms},${speedup},${pass}" >> "${RELU_MAIN_CSV}"
  [[ "${pass}" == PASS ]] || FAILURES=$((FAILURES + 1))
  rm -f "${log}"
}

run_main_suite() {
  RELU_MAIN_CSV="${OUT_DIR}/relu_figure9_${GPU_LABEL}_${PRESET}.csv"
  echo "gpu_label,gpu_index,data_mb,threads,runs,workflow,assignment,baseline_ms,mode_ms,speedup,pass" > "${RELU_MAIN_CSV}"
  for_each_main_case run_main_case
  echo "CSV: ${RELU_MAIN_CSV}"
}

run_occupancy_case() {
  local blocks_per_sm="$1" log status baseline_ms mode_ms speedup correct pass
  log="${OUT_DIR}/.relu_occupancy_${GPU_LABEL}_${blocks_per_sm}.log"
  echo "===== Occupancy: data_mb=${OCCUPANCY_DATA_MB} threads=${MAIN_THREADS} blocks_per_sm=${blocks_per_sm} ====="
  set +e
  sidedgpu_run_gpu "${MAIN_BIN}" "${OCCUPANCY_DATA_MB}" "${MAIN_RUNS}" "${MAIN_THREADS}" 0 0 "${blocks_per_sm}" 2>&1 | tee "${log}"
  status="${PIPESTATUS[0]}"
  set -e
  baseline_ms="$(awk '/Speedup=/{for(i=1;i<=NF;++i)if($i~/^Baseline=/){sub("Baseline=","",$i);print $i;exit}}' "${log}")"
  mode_ms="$(awk '/Speedup=/{for(i=1;i<=NF;++i)if($i~/^Ours=/){sub("Ours=","",$i);print $i;exit}}' "${log}")"
  speedup="$(awk '/Speedup=/{for(i=1;i<=NF;++i)if($i~/^Speedup=/){sub("Speedup=","",$i);sub("x","",$i);print $i;exit}}' "${log}")"
  correct="$(awk '/Baseline vs Ours=/{print $NF;exit}' "${log}")"
  pass="$([[ "${status}" == 0 && "${correct:-}" == Ours=PASS ]] && echo PASS || echo FAIL)"
  echo "${GPU_LABEL},${SIDEDGPU_GPU_INDEX},${OCCUPANCY_DATA_MB},${MAIN_THREADS},${blocks_per_sm},${MAIN_RUNS},green_context,aware,${baseline_ms},${mode_ms},${speedup},${pass}" >> "${RELU_OCCUPANCY_CSV}"
  [[ "${pass}" == PASS ]] || FAILURES=$((FAILURES + 1))
  rm -f "${log}"
}

run_occupancy_suite() {
  local blocks_per_sm
  RELU_OCCUPANCY_CSV="${OUT_DIR}/relu_occupancy_${GPU_LABEL}_${PRESET}.csv"
  echo "gpu_label,gpu_index,data_mb,threads,blocks_per_sm,runs,workflow,assignment,baseline_ms,mode_ms,speedup,pass" > "${RELU_OCCUPANCY_CSV}"
  for blocks_per_sm in ${OCCUPANCY_BLOCKS_PER_SM}; do
    run_occupancy_case "${blocks_per_sm}"
  done
  echo "CSV: ${RELU_OCCUPANCY_CSV}"
}

run_cutlass_case() {
  local data_mb="$1" predicate_log cutlass_log status_predicate status_cutlass
  local persistent_blocks our_ms cutlass_ms speedup correct cutlass_correct pass
  predicate_log="${OUT_DIR}/.relu_cutlass_${GPU_LABEL}_${data_mb}_predicate.log"
  cutlass_log="${OUT_DIR}/.relu_cutlass_${GPU_LABEL}_${data_mb}_cutlass.log"
  echo "===== CUTLASS comparison: data_mb=${data_mb} threads=${CUTLASS_THREADS} ====="
  set +e
  sidedgpu_run_gpu "${PREDICATE_BIN}" "${data_mb}" "${CUTLASS_RUNS}" "${CUTLASS_THREADS}" "${CACHE_MODE}" 2>&1 | tee "${predicate_log}"
  status_predicate="${PIPESTATUS[0]}"
  set -e
  persistent_blocks="$(awk '/^config /{for(i=1;i<=NF;++i)if($i~/^persistent_blocks=/){sub("persistent_blocks=","",$i);print $i;exit}}' "${predicate_log}")"
  our_ms="$(awk '/^our_ms=/{sub("our_ms=","");print;exit}' "${predicate_log}")"
  correct="$(awk -F= '/^correct=/{print $2;exit}' "${predicate_log}")"
  [[ "${persistent_blocks}" =~ ^[1-9][0-9]*$ ]] || status_predicate=1

  set +e
  sidedgpu_run_gpu "${CUTLASS_BIN}" "${data_mb}" "${CUTLASS_RUNS}" "${CUTLASS_THREADS}" "${persistent_blocks:-0}" 5 "${CACHE_MODE}" 2>&1 | tee "${cutlass_log}"
  status_cutlass="${PIPESTATUS[0]}"
  set -e
  cutlass_ms="$(awk '/^cutlass_epilogue_relu/{print $2;exit}' "${cutlass_log}")"
  cutlass_correct="$(awk -F= '/^correct=/{print $2;exit}' "${cutlass_log}")"
  speedup="$(awk -v c="${cutlass_ms}" -v o="${our_ms}" 'BEGIN{if(o>0)printf "%.6f",c/o}')"
  pass="$([[ "${status_predicate}" == 0 && "${status_cutlass}" == 0 && "${correct:-}" == PASS && "${cutlass_correct:-}" == PASS && -n "${our_ms}" && -n "${cutlass_ms}" ]] && echo PASS || echo FAIL)"
  echo "${GPU_LABEL},${SIDEDGPU_GPU_INDEX},${data_mb},${CUTLASS_THREADS},${persistent_blocks},${CUTLASS_RUNS},${CACHE_MODE},${our_ms},${cutlass_ms},${speedup},${pass}" >> "${RELU_CUTLASS_CSV}"
  [[ "${pass}" == PASS ]] || FAILURES=$((FAILURES + 1))
  rm -f "${predicate_log}" "${cutlass_log}"
}

run_cutlass_suite() {
  RELU_CUTLASS_CSV="${OUT_DIR}/relu_cutlass_${GPU_LABEL}_${PRESET}.csv"
  echo "gpu_label,gpu_index,data_mb,threads,persistent_blocks,runs,cache_mode,our_ms,cutlass_ms,speedup_vs_cutlass,pass" > "${RELU_CUTLASS_CSV}"
  for_each_cutlass_case run_cutlass_case
  echo "CSV: ${RELU_CUTLASS_CSV}"
}

[[ "${SUITE}" == main || "${SUITE}" == all ]] && run_main_suite
[[ "${SUITE}" == occupancy || "${SUITE}" == all ]] && run_occupancy_suite
[[ "${SUITE}" == cutlass || "${SUITE}" == all ]] && run_cutlass_suite
sidedgpu_app_exit_on_failures "${FAILURES}"
