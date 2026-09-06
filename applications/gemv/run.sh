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
  quick) M_VALUES="2048"; OUR_RUNS=5; CUBLAS_RUNS=5; WARMUPS=1 ;;
  paper) M_VALUES="2048 3072 4096 5120 6144 7168 8192 9216 10240"; OUR_RUNS=50; CUBLAS_RUNS=20; WARMUPS=5 ;;
  *) echo "--preset must be quick or paper" >&2; exit 2 ;;
esac

sidedgpu_app_begin "${APP_DIR}" gemv
case "${SIDEDGPU_GPU_NAME}" in
  *A100*PCI*) GPU_CSV_NAME="A100_PCIe" ;;
  *H100*PCI*|*H100*NVL*) GPU_LABEL="custom"; GPU_CSV_NAME="CUSTOM" ;;
  *H100*) GPU_CSV_NAME="H100_SXM" ;;
  *" B200"*) GPU_CSV_NAME="B200" ;;
  *) GPU_LABEL="custom"; GPU_CSV_NAME="CUSTOM" ;;
esac

OURS_BIN="${BIN_DIR}/gemv_sidegpu_l32_w16"
CUBLAS_BIN="${BIN_DIR}/gemv_cublas_benchmark"

sidedgpu_run nvcc -std=c++20 -O3 -arch="${CUDA_ARCH}" -I"${ROOT_DIR}/utility" \
  -DLANES_PER_ROW=32 -DWARPS_PER_BLOCK=16 \
  "${APP_DIR}/src/gemv_k1024_persistent_subwarp.cu" -o "${OURS_BIN}"
sidedgpu_run nvcc -std=c++20 -O3 -arch="${CUDA_ARCH}" -I"${ROOT_DIR}/utility" \
  "${APP_DIR}/src/gemv_cublas_benchmark.cu" -lcublas -o "${CUBLAS_BIN}"

if (( BUILD_ONLY )); then
  exit 0
fi

for_each_case() {
  local callback="$1"
  local m
  for m in ${M_VALUES}; do
    "${callback}" "${m}"
  done
}

dry_case() {
  sidedgpu_run_gpu "${OURS_BIN}" "$1" "${OUR_RUNS}" 4242 1
  sidedgpu_run_gpu "${CUBLAS_BIN}" "$1" 1024 "${CUBLAS_RUNS}" "${WARMUPS}" 1
}

if (( DRY_RUN )); then
  for_each_case dry_case
  sidedgpu_run "$(sidedgpu_python)" "${APP_DIR}/plot_gemv.py" --data-dir "${OUT_DIR}" --output-dir "${OUT_DIR}"
  exit 0
fi

CSV="${OUT_DIR}/final_gemv_${GPU_CSV_NAME}.csv"
echo "M,K,lanes_per_row,warps_per_block,baseline_ms,aware_ms,swapped_ms,cublas_ms,max_abs_error,aware_speedup,swapped_speedup,aware_vs_cublas,pass" > "${CSV}"
FAILURES=0

parse_value() {
  local pattern="$1" file="$2"
  awk -v pattern="${pattern}" '$1 == pattern {print $2; exit}' "${file}"
}

run_case() {
  local m="$1"
  local ours_log cublas_log ours_status cublas_status
  local baseline aware swapped cublas_ms max_error
  local aware_speedup swapped_speedup aware_vs_cublas pass
  ours_log="${OUT_DIR}/.gemv_${GPU_LABEL}_${m}_ours.log"
  cublas_log="${OUT_DIR}/.gemv_${GPU_LABEL}_${m}_cublas.log"
  echo "===== M=${m} K=1024 ====="
  set +e
  sidedgpu_run_gpu "${OURS_BIN}" "${m}" "${OUR_RUNS}" 4242 1 2>&1 | tee "${ours_log}"
  ours_status="${PIPESTATUS[0]}"
  sidedgpu_run_gpu "${CUBLAS_BIN}" "${m}" 1024 "${CUBLAS_RUNS}" "${WARMUPS}" 1 2>&1 | tee "${cublas_log}"
  cublas_status="${PIPESTATUS[0]}"
  set -e

  baseline="$(parse_value baseline_ms "${ours_log}")"
  aware="$(parse_value aware_ms "${ours_log}")"
  swapped="$(parse_value swapped_ms "${ours_log}")"
  cublas_ms="$(parse_value cublas_ms "${cublas_log}")"
  max_error="$(awk '/correctness max_abs_error/{max=0; for(i=1;i<=NF;i++) if($i~/=/){split($i,a,"="); if((a[2]+0)>max)max=a[2]+0} print max; exit}' "${ours_log}")"
  aware_speedup="$(awk -v b="${baseline}" -v a="${aware}" 'BEGIN{if(a>0)printf "%.9f",b/a}')"
  swapped_speedup="$(awk -v b="${baseline}" -v s="${swapped}" 'BEGIN{if(s>0)printf "%.9f",b/s}')"
  aware_vs_cublas="$(awk -v c="${cublas_ms}" -v a="${aware}" 'BEGIN{if(a>0)printf "%.9f",c/a}')"
  pass="$(awk -v os="${ours_status}" -v cs="${cublas_status}" \
    -v b="${baseline}" -v a="${aware}" -v s="${swapped}" \
    -v c="${cublas_ms}" -v e="${max_error}" \
    'BEGIN { ok=(os==0 && cs==0 && b>0 && a>0 && s>0 && c>0 && e!="" && e+0<=0.001); print ok ? "PASS" : "FAIL" }')"
  echo "${m},1024,32,16,${baseline},${aware},${swapped},${cublas_ms},${max_error},${aware_speedup},${swapped_speedup},${aware_vs_cublas},${pass}" >> "${CSV}"
  if [[ "${pass}" == FAIL ]]; then
    FAILURES=$((FAILURES + 1))
  fi
  rm -f "${ours_log}" "${cublas_log}"
}

for_each_case run_case

if (( FAILURES > 0 )); then
  echo "CSV: ${CSV}"
fi
sidedgpu_app_exit_on_failures "${FAILURES}"

PYTHON_BIN="$(sidedgpu_python)"
sidedgpu_run "${PYTHON_BIN}" "${APP_DIR}/plot_gemv.py" \
  --data-dir "${OUT_DIR}" --output-dir "${OUT_DIR}"

echo "CSV: ${CSV}"
