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
  quick)
    VERTICES=65536; DEGREE=8; SOURCES=128; DEPTH=2; DST_LOCAL=95
    HOT_PER_SIDE=4096; RUNS=1; SEED=8201
    ;;
  paper)
    VERTICES=2000000; DEGREE=64; SOURCES=4096; DEPTH=4; DST_LOCAL=95
    HOT_PER_SIDE=200000; RUNS=5; SEED=8201
    ;;
  *) echo "--preset must be quick or paper" >&2; exit 2 ;;
esac

sidedgpu_app_begin "${APP_DIR}" bfs

RUN_BIN="${BIN_DIR}/bfs"

sidedgpu_run nvcc -std=c++20 -O3 -arch="${CUDA_ARCH}" \
  -I"${ROOT_DIR}/utility" \
  "${APP_DIR}/src/bfs.cu" -o "${RUN_BIN}"

if (( BUILD_ONLY )); then
  exit 0
fi

for_each_case() {
  local callback="$1"
  if [[ "${PRESET}" == quick ]]; then
    "${callback}" quick default "${VERTICES}" "${DEGREE}" "${SOURCES}" "${DEPTH}" "${DST_LOCAL}" "${HOT_PER_SIDE}" "${RUNS}" "${SEED}"
    return
  fi

  # Table 10: four degrees, two additional source counts, and two additional
  # depths. The degree-64/source-4096/depth-4 default is emitted only once.
  "${callback}" degree 16 "${VERTICES}" 16 "${SOURCES}" "${DEPTH}" "${DST_LOCAL}" "${HOT_PER_SIDE}" "${RUNS}" "${SEED}"
  "${callback}" degree 32 "${VERTICES}" 32 "${SOURCES}" "${DEPTH}" "${DST_LOCAL}" "${HOT_PER_SIDE}" "${RUNS}" "${SEED}"
  "${callback}" degree 64 "${VERTICES}" 64 "${SOURCES}" "${DEPTH}" "${DST_LOCAL}" "${HOT_PER_SIDE}" "${RUNS}" "${SEED}"
  "${callback}" degree 128 "${VERTICES}" 128 "${SOURCES}" "${DEPTH}" "${DST_LOCAL}" "${HOT_PER_SIDE}" "${RUNS}" "${SEED}"
  "${callback}" sources 512 "${VERTICES}" "${DEGREE}" 512 "${DEPTH}" "${DST_LOCAL}" "${HOT_PER_SIDE}" "${RUNS}" "${SEED}"
  "${callback}" sources 16384 "${VERTICES}" "${DEGREE}" 16384 "${DEPTH}" "${DST_LOCAL}" "${HOT_PER_SIDE}" "${RUNS}" "${SEED}"
  "${callback}" depth 2 "${VERTICES}" "${DEGREE}" "${SOURCES}" 2 "${DST_LOCAL}" "${HOT_PER_SIDE}" "${RUNS}" "${SEED}"
  "${callback}" depth 8 "${VERTICES}" "${DEGREE}" "${SOURCES}" 8 "${DST_LOCAL}" "${HOT_PER_SIDE}" "${RUNS}" "${SEED}"
}

dry_case() {
  shift 2
  sidedgpu_run_gpu "${RUN_BIN}" "$@"
}

if (( DRY_RUN )); then
  for_each_case dry_case
  exit 0
fi

CSV="${OUT_DIR}/bfs_${GPU_LABEL}_${PRESET}.csv"
echo "sweep,value,gpu_label,gpu_index,vertices,degree,sources,depth,dst_local_pct,hot_per_side,runs,seed,visited,aware_bad,swapped_bad,overflow,baseline_med_ms,aware_med_ms,swapped_med_ms,aware_speedup,swapped_speedup,pass" > "${CSV}"
FAILURES=0

run_case() {
  local sweep="$1" value="$2" vertices="$3" degree="$4" sources="$5" depth="$6"
  local dst_local="$7" hot_per_side="$8" runs="$9" seed="${10}"
  local log status visited aware_bad swapped_bad overflow
  local base_med aware_med swapped_med aware_speedup swapped_speedup pass
  log="${OUT_DIR}/.bfs_${GPU_LABEL}_${sweep}_${value}.log"
  echo "===== ${sweep}=${value} degree=${degree} sources=${sources} depth=${depth} dst_local=${dst_local} ====="
  set +e
  sidedgpu_run_gpu "${RUN_BIN}" "${vertices}" "${degree}" "${sources}" "${depth}" \
    "${dst_local}" "${hot_per_side}" "${runs}" "${seed}" 2>&1 | tee "${log}"
  status="${PIPESTATUS[0]}"
  set -e

  visited=""; aware_bad=""; swapped_bad=""; overflow=""
  read -r visited aware_bad swapped_bad overflow < <(awk -F'[ =]' '/^visited=/{print $2, $4, $6, $8; exit}' "${log}") || true
  base_med="$(awk -F'|' '/^\| baseline / {gsub(/^[ \t]+|[ \t]+$/, "", $3); print $3; exit}' "${log}")"
  aware_med="$(awk -F'|' '/^\| aware / {gsub(/^[ \t]+|[ \t]+$/, "", $3); print $3; exit}' "${log}")"
  swapped_med="$(awk -F'|' '/^\| swapped / {gsub(/^[ \t]+|[ \t]+$/, "", $3); print $3; exit}' "${log}")"
  aware_speedup="$(awk -v b="${base_med}" -v a="${aware_med}" 'BEGIN { if (a > 0) printf "%.6f", b / a }')"
  swapped_speedup="$(awk -v b="${base_med}" -v s="${swapped_med}" 'BEGIN { if (s > 0) printf "%.6f", b / s }')"
  pass="$([[ "${status}" == "0" && "${aware_bad:-1}" == "0" && "${swapped_bad:-1}" == "0" && "${overflow:-1}" == "0" ]] && echo PASS || echo FAIL)"
  echo "${sweep},${value},${GPU_LABEL},${SIDEDGPU_GPU_INDEX},${vertices},${degree},${sources},${depth},${dst_local},${hot_per_side},${runs},${seed},${visited:-},${aware_bad:-},${swapped_bad:-},${overflow:-},${base_med},${aware_med},${swapped_med},${aware_speedup},${swapped_speedup},${pass}" >> "${CSV}"
  if [[ "${pass}" == FAIL ]]; then
    FAILURES=$((FAILURES + 1))
  fi
  rm -f "${log}"
}

for_each_case run_case

echo "CSV: ${CSV}"
sidedgpu_app_exit_on_failures "${FAILURES}"
