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
    RUNS=1
    ;;
  paper)
    RUNS=5
    ;;
  *) echo "--preset must be quick or paper" >&2; exit 2 ;;
esac
SEED=4242
CTAS_PER_SM=1
THREADS=64

sidedgpu_app_begin "${APP_DIR}" bvh
CONFIG_DIR="${OUT_DIR}/config"

RUN_BIN="${BIN_DIR}/bvh"

sidedgpu_run nvcc -O3 -std=c++20 -arch="${CUDA_ARCH}" -DBVH_THREADS="${THREADS}" \
  -I"${ROOT_DIR}/utility" \
  "${APP_DIR}/src/bvh.cu" "${ROOT_DIR}/utility/sidedgpu.cu" \
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
  if [[ "${PRESET}" == quick ]]; then
    "${callback}" 65536 65536 16 100 50 1 1
    return
  fi

  # Table 9: three traversal depths, three skew points, and two larger sizes.
  # The default 1M/step-32/50% point is emitted only once.
  "${callback}" 1048576 1048576 16 100 50 5 3
  "${callback}" 1048576 1048576 32 100 50 5 3
  "${callback}" 1048576 1048576 64 100 50 5 3
  "${callback}" 1048576 1048576 32 100 65 5 3
  "${callback}" 1048576 1048576 32 100 80 5 3
  "${callback}" 1048576 1048576 32 100 95 5 3
  "${callback}" 2097152 2097152 32 100 50 5 3
  "${callback}" 4194304 4194304 32 100 50 5 3
}

dry_case() {
  run_benchmark "$1" "$2" "$3" "$4" "$5" "${RUNS}" "${SEED}" "${CTAS_PER_SM}" "$6" "$7"
}

if (( DRY_RUN )); then
  for_each_case dry_case
  exit 0
fi

read -r GREEN_SIDE0_SMS GREEN_SIDE1_SMS < <(
  awk -F, '{
    for (i = 1; i <= NF; ++i) {
      if ($i == "0") ++side0;
      else if ($i == "1") ++side1;
    }
  } END { print side0 + 0, side1 + 0 }' "${CONFIG_DIR}/smid2side.csv"
)
if (( GREEN_SIDE0_SMS <= 0 || GREEN_SIDE1_SMS <= 0 )); then
  echo "invalid side map: expected non-empty side 0 and side 1" >&2
  exit 1
fi
REGULAR_CTAS=$(((GREEN_SIDE0_SMS + GREEN_SIDE1_SMS) * CTAS_PER_SM))
SIDE_CTAS=$(((GREEN_SIDE0_SMS + GREEN_SIDE1_SMS) / 2 * CTAS_PER_SM))

CSV="${OUT_DIR}/bvh_${GPU_LABEL}_${PRESET}.csv"
echo "gpu_label,gpu_index,nodes,rays,steps,locality,side0_pct,frames,bounces,runs,seed,regular_ctas,side_ctas,green_side0_sms,green_side1_sms,threads,worklist0,worklist1,mode,mean_ms,speedup_vs_regular,pass" > "${CSV}"
FAILURES=0

run_case() {
  local nodes="$1" rays="$2" steps="$3" locality="$4" side0_pct="$5" bounces="$6" frames="$7"
  local log status wl0=0 wl1=0
  log="${OUT_DIR}/.bvh_${GPU_LABEL}_${nodes}_${rays}_${steps}_${locality}_${side0_pct}_${frames}_${bounces}.log"
  echo "===== nodes=${nodes} rays=${rays} steps=${steps} locality=${locality} side0_pct=${side0_pct} frames=${frames} bounces=${bounces} ====="
  set +e
  run_benchmark "${nodes}" "${rays}" "${steps}" "${locality}" "${side0_pct}" "${RUNS}" \
    "${SEED}" "${CTAS_PER_SM}" "${bounces}" "${frames}" 2>&1 | tee "${log}"
  status="${PIPESTATUS[0]}"
  set -e

  read -r wl0 wl1 < <(awk '/worklist_counts=/ {
    for (i = 1; i <= NF; ++i) if ($i ~ /^worklist_counts=/) {
      split($i, a, "="); split(a[2], b, "/"); print b[1], b[2]; exit
    }
  }' "${log}") || true

  local parsed_rows case_status
  parsed_rows="$(awk '/^\| (regular|aware|adaptive|swapped) / { ++count } END { print count + 0 }' "${log}")"
  case_status="${status}"
  if (( parsed_rows != 4 )); then
    echo "expected 4 result rows, parsed ${parsed_rows}" >&2
    case_status=1
  fi

  awk -v gpu_label="${GPU_LABEL}" -v gpu="${SIDEDGPU_GPU_INDEX}" \
      -v nodes="${nodes}" -v rays="${rays}" -v steps="${steps}" \
      -v locality="${locality}" -v side0_pct="${side0_pct}" \
      -v frames="${frames}" -v bounces="${bounces}" -v runs="${RUNS}" \
      -v seed="${SEED}" -v regular_ctas="${REGULAR_CTAS}" -v side_ctas="${SIDE_CTAS}" \
      -v green_side0_sms="${GREEN_SIDE0_SMS}" -v green_side1_sms="${GREEN_SIDE1_SMS}" \
      -v threads="${THREADS}" \
      -v wl0="${wl0:-0}" -v wl1="${wl1:-0}" \
      -v pass="$([[ "${case_status}" == "0" ]] && echo PASS || echo FAIL)" \
      'BEGIN { FS="|" }
       /^\| (regular|aware|adaptive|swapped) / {
         for (i = 2; i <= 4; ++i) gsub(/^[ \t]+|[ \t]+$/, "", $i);
         print gpu_label "," gpu "," nodes "," rays "," steps "," locality "," side0_pct "," frames "," bounces "," runs "," seed "," regular_ctas "," side_ctas "," green_side0_sms "," green_side1_sms "," threads "," wl0 "," wl1 "," $2 "," $3 "," $4 "," pass;
       }' "${log}" >> "${CSV}"
  if (( case_status != 0 )); then
    FAILURES=$((FAILURES + 1))
  fi
  rm -f "${log}"
}

for_each_case run_case

echo "CSV: ${CSV}"
sidedgpu_app_exit_on_failures "${FAILURES}"
