#!/usr/bin/env bash
set -euo pipefail

ARCHITECTURE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${ARCHITECTURE_DIR}/../scripts/common.sh"

figure="${1:-}"
program="${2:-}"
if [[ ! "${figure}" =~ ^[1-5]$ || -z "${program}" ]]; then
    sidedgpu_die "internal error: expected FIGURE and public entrypoint" || true
    exit 2
fi
shift 2
printf -v tag '%02d' "${figure}"

usage() {
    cat <<EOF
Usage: ${program} [options]

Build, measure, and plot architecture Figure ${figure}.

  --device VALUE       auto, physical index, or full GPU UUID (default: auto)
EOF
    if [[ "${figure}" == 2 || "${figure}" == 4 || "${figure}" == 5 ]]; then
        cat <<EOF
  --peer A,B           cross-domain SM pair (default: auto-discover)
EOF
    fi
    cat <<EOF
  --arch VALUE         sm_80, sm_90a, sm_100, sm_120, or auto (default: auto)
  --output-dir DIR     output root (default: architecture/out/figure_${tag}; use . to update references)
  --build-only         compile the CUDA code required by this Figure, then stop
  --dry-run            print commands without querying a GPU or writing files
  --wait-timeout SEC   maximum idle-GPU wait (default: 1800)
  --poll-interval SEC  idle-GPU polling interval (default: 10)
  -h, --help           show this help
EOF
}

if sidedgpu_parse_entry_options "$@"; then
    :
else
    status=$?
    usage >&2
    exit "${status}"
fi
if (( SIDEDGPU_OPT_HELP )); then usage; exit 0; fi
if [[ "${figure}" == 1 || "${figure}" == 3 ]] && [[ "${SIDEDGPU_OPT_PEER}" != auto ]]; then
    sidedgpu_die "--peer is only supported by Figures 2, 4, and 5" || true
    exit 2
fi

if [[ -n "${SIDEDGPU_OPT_OUTPUT_DIR}" ]]; then
    artifact_root="$(sidedgpu_resolve_path "${ARCHITECTURE_DIR}" "${SIDEDGPU_OPT_OUTPUT_DIR}")"
    data_dir="${artifact_root}/data"
    figure_dir="${artifact_root}/figures"
    build_root="${artifact_root}/out"
else
    artifact_root="${ARCHITECTURE_DIR}/out/figure_${tag}"
    data_dir="${artifact_root}/data"
    figure_dir="${artifact_root}/figures"
    build_root="${artifact_root}"
    sidedgpu_log "using generated-output directory ${artifact_root}; tracked references are unchanged"
fi
build_dir="${build_root}/bin"
metadata_dir="${build_root}/metadata"

sidedgpu_begin_gpu_context
cuda_arch="${SIDEDGPU_GPU_ARCH}"
sidedgpu_ensure_dir "${data_dir}" "${figure_dir}" "${build_dir}"
(( SIDEDGPU_OPT_DRY_RUN )) || sidedgpu_require_command nvcc

binary="${build_dir}/architecture_probe_${cuda_arch}"
csv_path="${data_dir}/figure_${tag}.csv"
png_path="${figure_dir}/figure_${tag}.png"
source_path="${ARCHITECTURE_DIR}/src/architecture_probe.cu"
measurement_csv="${csv_path}"
sidedgpu_run nvcc -O3 -std=c++20 -lineinfo "-arch=${cuda_arch}" "${source_path}" -o "${binary}"

calibration_dir=""
side_map_path=""
need_side_map=0
if [[ ( "${figure}" == 2 || "${figure}" == 4 || "${figure}" == 5 ) && "${SIDEDGPU_OPT_PEER}" == auto ]]; then
    need_side_map=1
    calibration_dir="${build_root}/config/figure_${tag}"
    side_map_path="${calibration_dir}/smid2side.csv"
fi
if (( SIDEDGPU_OPT_BUILD_ONLY )); then
    (( ! need_side_map )) || sidedgpu_build_side_map_tool "${build_dir}" "${cuda_arch}"
    sidedgpu_log "build-only complete: ${binary}"
    exit 0
fi
(( ! need_side_map )) || sidedgpu_prepare_side_map "${calibration_dir}" "${build_dir}" "${cuda_arch}"

probe_command=("${binary}" --figure "${figure}" --device 0 --output "${measurement_csv}")
if [[ "${figure}" == 2 || "${figure}" == 4 || "${figure}" == 5 ]]; then
    probe_command+=(--peer "${SIDEDGPU_OPT_PEER}")
fi
if [[ -n "${side_map_path}" ]]; then
    probe_command+=(--side-map "${side_map_path}")
fi
printf -v SIDEDGPU_METADATA_COMMAND '%q ' "${program}" "${SIDEDGPU_OPT_ORIGINAL_ARGS[@]}"
SIDEDGPU_METADATA_COMMAND="${SIDEDGPU_METADATA_COMMAND% }"
sidedgpu_run_gpu "${probe_command[@]}"

python_bin="$(sidedgpu_python)"
plot_command=("${python_bin}" "${ARCHITECTURE_DIR}/plot.py" \
    --figure "${figure}" --input "${measurement_csv}" --output "${png_path}")
sidedgpu_run "${plot_command[@]}"

sidedgpu_ensure_dir "${metadata_dir}"
sidedgpu_write_metadata "${metadata_dir}/figure_${tag}.json" \
    "figure=${tag}" "peer_request=${SIDEDGPU_OPT_PEER}" "cuda_arch=${cuda_arch}" \
    "side_map_path=${side_map_path}" \
    "csv=${csv_path}" "png=${png_path}"

sidedgpu_log "Figure ${tag} CSV: ${csv_path}"
sidedgpu_log "Figure ${tag} PNG: ${png_path}"
sidedgpu_log "Figure ${tag} metadata: ${metadata_dir}/figure_${tag}.json"
