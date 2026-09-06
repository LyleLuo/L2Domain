#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${APP_DIR}/../.." && pwd)"
source "${ROOT_DIR}/scripts/app_runner.sh"

usage() {
  printf 'Usage: ./run.sh [common options]\n\n'
  sidedgpu_app_print_common_options
}

sidedgpu_app_parse_options usage "$@"

case "${PRESET}" in
  quick) DATA_MBS="32"; RUNS=1 ;;
  paper) DATA_MBS="32 64 128 512"; RUNS=10 ;;
  *) echo "--preset must be quick or paper" >&2; exit 2 ;;
esac
THREADS=128
BLOCKS_PER_SM=1
CACHE_MODE=0

sidedgpu_app_begin "${APP_DIR}" histogram

RUN_BIN="${BIN_DIR}/histogram256_sideaware"

sidedgpu_run nvcc -std=c++20 -O3 -arch="${CUDA_ARCH}" \
  -I"${ROOT_DIR}/utility" \
  "${APP_DIR}/src/histogram256_sideaware.cu" -o "${RUN_BIN}"

if (( BUILD_ONLY )); then
  exit 0
fi

for_each_case() {
  local callback="$1"
  local data_mb
  for data_mb in ${DATA_MBS}; do
    "${callback}" "${data_mb}"
  done
}

dry_case() {
  sidedgpu_run_gpu "${RUN_BIN}" "$1" "${RUNS}" "${THREADS}" "${BLOCKS_PER_SM}" "${CACHE_MODE}"
}

if (( DRY_RUN )); then
  for_each_case dry_case
  exit 0
fi

CSV="${OUT_DIR}/histogram_${GPU_LABEL}_${PRESET}.csv"
echo "gpu_label,gpu_index,data_mb,threads,blocks_per_sm,total_blocks,persistent_blocks,runs,cache_mode,baseline_kernel_ms,green_aware_kernel_ms,green_swapped_kernel_ms,persistent_aware_kernel_ms,persistent_swapped_kernel_ms,green_aware_kernel_speedup,green_swapped_kernel_speedup,persistent_aware_kernel_speedup,persistent_swapped_kernel_speedup,pass" > "${CSV}"
FAILURES=0

run_case() {
  local data_mb="$1"
  local log status total_blocks persistent_blocks baseline_kernel_ms
  local green_aware_kernel_ms green_swapped_kernel_ms persistent_aware_kernel_ms persistent_swapped_kernel_ms
  local baseline_ok green_aware_ok green_swapped_ok persistent_aware_ok persistent_swapped_ok
  local green_aware_kernel_speedup green_swapped_kernel_speedup persistent_aware_kernel_speedup persistent_swapped_kernel_speedup pass
  log="${OUT_DIR}/.histogram_${GPU_LABEL}_${data_mb}.log"
  echo "===== data_mb=${data_mb} threads=${THREADS} blocks_per_sm=${BLOCKS_PER_SM} cache_mode=${CACHE_MODE} ====="
  set +e
  sidedgpu_run_gpu "${RUN_BIN}" "${data_mb}" "${RUNS}" "${THREADS}" "${BLOCKS_PER_SM}" "${CACHE_MODE}" 2>&1 | tee "${log}"
  status="${PIPESTATUS[0]}"
  set -e

  total_blocks="$(awk 'match($0, /baseline_blocks=[0-9]+/) { value=substr($0, RSTART+16, RLENGTH-16); print value; exit }' "${log}")"
  persistent_blocks="$(awk 'match($0, /persistent_blocks=[0-9]+/) { value=substr($0, RSTART+18, RLENGTH-18); print value; exit }' "${log}")"
  baseline_kernel_ms="$(awk '/^baseline_kernel_ms/{print $2; exit}' "${log}")"
  green_aware_kernel_ms="$(awk '/^green_aware_kernel_ms/{print $2; exit}' "${log}")"
  green_swapped_kernel_ms="$(awk '/^green_swapped_kernel_ms/{print $2; exit}' "${log}")"
  persistent_aware_kernel_ms="$(awk '/^persistent_aware_kernel_ms/{print $2; exit}' "${log}")"
  persistent_swapped_kernel_ms="$(awk '/^persistent_swapped_kernel_ms/{print $2; exit}' "${log}")"
  baseline_ok=""; green_aware_ok=""; green_swapped_ok=""; persistent_aware_ok=""; persistent_swapped_ok=""
  read -r baseline_ok green_aware_ok green_swapped_ok persistent_aware_ok persistent_swapped_ok < <(awk '/^correct/{print $2, $3, $4, $5, $6; exit}' "${log}") || true
  green_aware_kernel_speedup="$(awk -v b="${baseline_kernel_ms}" -v v="${green_aware_kernel_ms}" 'BEGIN { if (v > 0) printf "%.6f", b / v }')"
  green_swapped_kernel_speedup="$(awk -v b="${baseline_kernel_ms}" -v v="${green_swapped_kernel_ms}" 'BEGIN { if (v > 0) printf "%.6f", b / v }')"
  persistent_aware_kernel_speedup="$(awk -v b="${baseline_kernel_ms}" -v v="${persistent_aware_kernel_ms}" 'BEGIN { if (v > 0) printf "%.6f", b / v }')"
  persistent_swapped_kernel_speedup="$(awk -v b="${baseline_kernel_ms}" -v v="${persistent_swapped_kernel_ms}" 'BEGIN { if (v > 0) printf "%.6f", b / v }')"
  pass="$([[ "${status}" == "0" && "${baseline_ok:-}" == "baseline=PASS" && "${green_aware_ok:-}" == "green_aware=PASS" && "${green_swapped_ok:-}" == "green_swapped=PASS" && "${persistent_aware_ok:-}" == "persistent_aware=PASS" && "${persistent_swapped_ok:-}" == "persistent_swapped=PASS" ]] && echo PASS || echo FAIL)"
  echo "${GPU_LABEL},${SIDEDGPU_GPU_INDEX},${data_mb},${THREADS},${BLOCKS_PER_SM},${total_blocks},${persistent_blocks},${RUNS},${CACHE_MODE},${baseline_kernel_ms},${green_aware_kernel_ms},${green_swapped_kernel_ms},${persistent_aware_kernel_ms},${persistent_swapped_kernel_ms},${green_aware_kernel_speedup},${green_swapped_kernel_speedup},${persistent_aware_kernel_speedup},${persistent_swapped_kernel_speedup},${pass}" >> "${CSV}"
  if [[ "${pass}" == FAIL ]]; then
    FAILURES=$((FAILURES + 1))
  fi
  rm -f "${log}"
}

for_each_case run_case

echo "CSV: ${CSV}"
sidedgpu_app_exit_on_failures "${FAILURES}"
