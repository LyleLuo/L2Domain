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
  quick) CHUNKS_VALUES="64"; CHUNK_BYTES_VALUES="4096"; RUNS=1 ;;
  paper) CHUNKS_VALUES="64 256 1024 4096 16384"; CHUNK_BYTES_VALUES="4096"; RUNS=5 ;;
  *) echo "--preset must be quick or paper" >&2; exit 2 ;;
esac

sidedgpu_app_begin "${APP_DIR}" batched_memcpy
CONFIG_DIR="${OUT_DIR}/config"

RUN_BIN="${BIN_DIR}/batched_memcpy"

sidedgpu_run nvcc -O2 -std=c++20 -arch="${CUDA_ARCH}" \
  -I"${ROOT_DIR}/utility" \
  "${APP_DIR}/src/batched_memcpy.cu" "${ROOT_DIR}/utility/sidedgpu.cu" \
  -lcuda -o "${RUN_BIN}"

if (( BUILD_ONLY )); then
  sidedgpu_build_calibration_tools "${BIN_DIR}" "${CUDA_ARCH}"
  exit 0
fi

run_benchmark() {
  SIDEDGPU_CONFIG_DIR="${CONFIG_DIR}" sidedgpu_run_gpu "${RUN_BIN}" "$@"
}

sidedgpu_prepare_calibration "${CONFIG_DIR}" "${BIN_DIR}" "${CUDA_ARCH}"

for_each_case() {
  local callback="$1"
  local chunks chunk_bytes
  for chunks in ${CHUNKS_VALUES}; do
    for chunk_bytes in ${CHUNK_BYTES_VALUES}; do
      "${callback}" "${chunks}" "${chunk_bytes}"
    done
  done
}

dry_case() {
  run_benchmark "$1" "$2" "${RUNS}"
}

if (( DRY_RUN )); then
  for_each_case dry_case
  exit 0
fi

CSV="${OUT_DIR}/batched_memcpy_${GPU_LABEL}_${PRESET}.csv"
echo "gpu_label,gpu_index,chunks,chunk_bytes,total_mib,runs,allocation_side,kernel_ctas,library_ms,aware_ms,swapped_ms,aware_speedup,swapped_speedup,pass" > "${CSV}"
FAILURES=0

run_case() {
    local chunks="$1" chunk_bytes="$2"
    local total_mib log status parsed_rows case_status
    total_mib="$(awk -v c="${chunks}" -v b="${chunk_bytes}" 'BEGIN { printf "%.6f", c * b / 1048576.0 }')"
    log="${OUT_DIR}/.batched_memcpy_${GPU_LABEL}_${chunks}_${chunk_bytes}.log"
    echo "===== chunks=${chunks} chunk_bytes=${chunk_bytes} runs=${RUNS} ====="
    set +e
    run_benchmark "${chunks}" "${chunk_bytes}" "${RUNS}" 2>&1 | tee "${log}"
    status="${PIPESTATUS[0]}"
    set -e

    parsed_rows="$(awk -F, '/^RESULT,/ { ++count } END { print count + 0 }' "${log}")"
    case_status="${status}"
    if (( parsed_rows != 1 )); then
      echo "expected one result row, parsed ${parsed_rows}" >&2
      case_status=1
    fi

    awk -v gpu_label="${GPU_LABEL}" -v gpu="${SIDEDGPU_GPU_INDEX}" \
        -v chunks="${chunks}" -v chunk_bytes="${chunk_bytes}" \
        -v total_mib="${total_mib}" -v runs="${RUNS}" \
        -v pass="$([[ "${case_status}" == "0" ]] && echo PASS || echo FAIL)" \
        'BEGIN { FS="," }
         /^RESULT,/ {
           library=$4; aware=$5; swapped=$6;
           print gpu_label "," gpu "," chunks "," chunk_bytes "," total_mib "," runs "," $2 "," $3 "," library "," aware "," swapped "," library/aware "," library/swapped "," pass;
         }' "${log}" >> "${CSV}"
    if (( case_status != 0 )); then
      FAILURES=$((FAILURES + 1))
    fi
    rm -f "${log}"
}

for_each_case run_case

echo "CSV: ${CSV}"
sidedgpu_app_exit_on_failures "${FAILURES}"
