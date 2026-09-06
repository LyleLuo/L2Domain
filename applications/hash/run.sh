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
  quick) RUNS=1 ;;
  paper) RUNS=7 ;;
  *) echo "--preset must be quick or paper" >&2; exit 2 ;;
esac

sidedgpu_app_begin "${APP_DIR}" hash

RUN_BIN="${BIN_DIR}/hash_probe_dep"
sidedgpu_run nvcc -std=c++20 -O3 -arch="${CUDA_ARCH}" \
  -I"${ROOT_DIR}/utility" \
  "${APP_DIR}/src/hash_probe_dep.cu" -lcuda -o "${RUN_BIN}"

if (( BUILD_ONLY )); then
  exit 0
fi

for_each_case() {
  local callback="$1"
  if [[ "${PRESET}" == quick ]]; then
    "${callback}" 1048576 524288 2097152 4 80 16384
    return
  fi

  # Table 8: default plus one-factor variations of L, C, and h.
  "${callback}" 32000000 8000000 67108864 1 80 100000
  "${callback}" 32000000 8000000 67108864 2 80 100000
  "${callback}" 32000000 8000000 67108864 4 80 100000
  "${callback}" 32000000 8000000 67108864 16 80 100000
  "${callback}" 32000000 8000000 33554432 4 80 100000
  "${callback}" 32000000 8000000 67108864 4 70 100000
  "${callback}" 32000000 8000000 67108864 4 99 100000
}

dry_case() {
  sidedgpu_run_gpu "${RUN_BIN}" "$@" "${RUNS}"
}

if (( DRY_RUN )); then
  for_each_case dry_case
  exit 0
fi

CSV="${OUT_DIR}/hash_${GPU_LABEL}_${PRESET}.csv"
echo "K,Q,C,effective_load,L,h,H,baseline_ms,aware_ms,swapped_ms,aware_speedup,swapped_speedup,pass" > "${CSV}"
FAILURES=0

result_value() {
  local log="$1" key="$2"
  awk -v key="${key}" '$1 == "RESULT" {for (i=2; i<=NF; i++) {split($i, part, "="); if (part[1] == key) {print part[2]; exit}}}' "${log}"
}

run_case() {
  local K="$1" Q="$2" C="$3" L="$4" h="$5" H="$6"
  local tag="K${K}_Q${Q}_C${C}_L${L}_h${h}_H${H}"
  local log="${OUT_DIR}/${tag}.log"
  local status effective_load baseline_ms aware_ms swapped_ms aware_speedup swapped_speedup result_pass pass

  echo "===== K=${K} Q=${Q} C=${C} L=${L} h=${h} H=${H} ====="
  set +e
  sidedgpu_run_gpu "${RUN_BIN}" "${K}" "${Q}" "${C}" "${L}" "${h}" "${H}" "${RUNS}" > "${log}" 2>&1
  status="$?"
  set -e
  cat "${log}"

  effective_load="$(awk -v k="${K}" -v c="${C}" 'BEGIN {printf "%.6f", 100*k/c}')"
  baseline_ms="$(result_value "${log}" baseline_ms)"
  aware_ms="$(result_value "${log}" aware_ms)"
  swapped_ms="$(result_value "${log}" swapped_ms)"
  aware_speedup="$(result_value "${log}" aware_speedup)"
  swapped_speedup="$(result_value "${log}" swapped_speedup)"
  result_pass="$(result_value "${log}" pass)"
  if [[ "${status}" == 0 && "${result_pass}" == PASS ]]; then
    pass=PASS
  else
    pass=FAIL
    FAILURES=$((FAILURES + 1))
  fi
  echo "${K},${Q},${C},${effective_load},${L},${h},${H},${baseline_ms},${aware_ms},${swapped_ms},${aware_speedup},${swapped_speedup},${pass}" >> "${CSV}"
  rm -f "${log}"
}

for_each_case run_case

echo "CSV: ${CSV}"
sidedgpu_app_exit_on_failures "${FAILURES}"
