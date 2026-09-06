#!/usr/bin/env bash
# Shared helpers for every sidedgpu GPU entrypoint.

if [[ -n "${SIDEDGPU_COMMON_SH_LOADED:-}" ]]; then
    return 0 2>/dev/null || exit 0
fi
SIDEDGPU_COMMON_SH_LOADED=1

SIDEDGPU_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIDEDGPU_ROOT_DIR="$(cd "${SIDEDGPU_COMMON_DIR}/.." && pwd)"
SIDEDGPU_SUPPORT_PY="${SIDEDGPU_COMMON_DIR}/_support.py"
if [[ -z "${SIDEDGPU_METADATA_COMMAND:-}" ]]; then
    printf -v SIDEDGPU_METADATA_COMMAND '%q ' "$0" "$@"
    SIDEDGPU_METADATA_COMMAND="${SIDEDGPU_METADATA_COMMAND% }"
fi

SIDEDGPU_DEFAULT_DEVICE="auto"
SIDEDGPU_DEFAULT_WAIT_TIMEOUT=1800
SIDEDGPU_DEFAULT_POLL_INTERVAL=10
SIDEDGPU_IDLE_SAMPLES=3
SIDEDGPU_IDLE_MAX_UTIL=5
SIDEDGPU_IDLE_MAX_MEMORY_MIB=1024
SIDEDGPU_DRY_RUN_AUTO_ARCH="sm_AUTO_DETECTED"

SIDEDGPU_GPU_INDEX=""
SIDEDGPU_GPU_UUID=""
SIDEDGPU_GPU_NAME=""
SIDEDGPU_GPU_ARCH=""
SIDEDGPU_GPU_LOCK_FD=""
SIDEDGPU_GPU_LOCK_PATH=""
SIDEDGPU_GPU_MEMORY_USED_MIB=""
SIDEDGPU_GPU_MAX_UTIL=""
SIDEDGPU_GPU_NO_COMPUTE_PROCESSES=""
SIDEDGPU_GPU_LOCK_HELD=""

sidedgpu_log() { printf '[sidedgpu] %s\n' "$*" >&2; }
sidedgpu_warn() { printf '[sidedgpu] warning: %s\n' "$*" >&2; }
sidedgpu_die() { printf '[sidedgpu] error: %s\n' "$*" >&2; return 1; }

sidedgpu_print_command() {
    local arg
    printf '[sidedgpu] +'
    for arg in "$@"; do printf ' %q' "${arg}"; done
    printf '\n'
}

sidedgpu_run() {
    sidedgpu_print_command "$@" >&2
    [[ "${SIDEDGPU_DRY_RUN:-0}" == "1" ]] || "$@"
}

sidedgpu_run_gpu() {
    if [[ -z "${SIDEDGPU_GPU_INDEX:-}" ]]; then
        sidedgpu_die "no GPU is locked; call sidedgpu_acquire_gpu first"
        return 1
    fi
    local visible="${SIDEDGPU_GPU_INDEX}"
    if [[ "${SIDEDGPU_DRY_RUN:-0}" == "1" && "${visible}" == "auto" ]]; then
        visible="<auto-selected-physical-index>"
    fi
    local -a scoped=("CUDA_VISIBLE_DEVICES=${visible}")
    [[ -z "${SIDEDGPU_CONFIG_DIR:-}" ]] || scoped+=("SIDEDGPU_CONFIG_DIR=${SIDEDGPU_CONFIG_DIR}")
    sidedgpu_run env "${scoped[@]}" "$@"
}

sidedgpu_resolve_path() {
    if [[ "$2" == /* ]]; then realpath -m -- "$2"; else realpath -m -- "$1/$2"; fi
}

sidedgpu_ensure_dir() {
    sidedgpu_run mkdir -p -- "$@"
}

sidedgpu_require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        sidedgpu_die "required command not found: $1"
        return 1
    fi
}

sidedgpu_python() {
    local candidate="" command_name
    if [[ -x "${SIDEDGPU_ROOT_DIR}/.venv/bin/python" ]]; then
        candidate="${SIDEDGPU_ROOT_DIR}/.venv/bin/python"
    else
        for command_name in python3.13 python3.12 python3.11 python3; do
            if command -v "${command_name}" >/dev/null 2>&1; then
                candidate="$(command -v "${command_name}")"
                break
            fi
        done
    fi
    if [[ -z "${candidate}" ]] || ! "${candidate}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
        sidedgpu_die "Python 3.11 or newer is required; create .venv with a supported interpreter"
        return 1
    fi
    printf '%s\n' "${candidate}"
}

_sidedgpu_is_uint() { [[ "$1" =~ ^[0-9]+$ ]]; }

sidedgpu_validate_device_selector() {
    local selector="${1:-}"
    if [[ "${selector}" != auto && ! "${selector}" =~ ^[0-9]+$ && ! "${selector}" =~ ^GPU-[A-Za-z0-9-]+$ ]]; then
        sidedgpu_die "--device must be auto, a physical GPU index, or a full GPU UUID"
        return 2
    fi
}

sidedgpu_resolve_arch() {
    local override="${1:-auto}"
    if [[ -n "${override}" && "${override}" != auto ]]; then
        [[ ! "${override}" =~ ^[0-9]{2,3}a?$ ]] || override="sm_${override}"
        if [[ ! "${override}" =~ ^sm_[0-9]{2,3}a?$ ]]; then
            sidedgpu_die "invalid CUDA architecture '${override}' (expected auto or sm_NN[a])"
            return 2
        fi
        printf '%s\n' "${override}"
        return 0
    fi
    case "${SIDEDGPU_GPU_NAME:-}" in
        *A100*) printf '%s\n' sm_80 ;;
        *H100*) printf '%s\n' sm_90a ;;
        *" B200"*) printf '%s\n' sm_100 ;;
        *RTX*5090*) printf '%s\n' sm_120 ;;
        *)
            sidedgpu_die "cannot infer CUDA architecture from GPU name '${SIDEDGPU_GPU_NAME:-unknown}'; pass an explicit architecture"
            return 1
            ;;
    esac
}

sidedgpu_build_side_map_tool() {
    local bin_dir="$1" arch="$2"
    [[ "${SIDEDGPU_DRY_RUN:-0}" == 1 ]] || sidedgpu_require_command nvcc || return
    sidedgpu_ensure_dir "${bin_dir}" || return
    sidedgpu_run nvcc -O2 -std=c++20 "-arch=${arch}" \
        -o "${bin_dir}/get_smid2side" "${SIDEDGPU_ROOT_DIR}/utility/get_smid2side.cu"
}

sidedgpu_build_calibration_tools() {
    local bin_dir="$1" arch="$2"
    sidedgpu_build_side_map_tool "${bin_dir}" "${arch}" || return
    sidedgpu_run nvcc -O2 -std=c++20 "-arch=${arch}" \
        -o "${bin_dir}/get_mask" "${SIDEDGPU_ROOT_DIR}/utility/get_mask.cu"
}

sidedgpu_prepare_side_map() {
    local config_dir="$1" bin_dir="$2" arch="$3" status=0
    if [[ "${SIDEDGPU_DRY_RUN:-0}" != 1 && ( -z "${SIDEDGPU_GPU_INDEX:-}" || -z "${SIDEDGPU_GPU_LOCK_FD:-}" ) ]]; then
        sidedgpu_die "side-map calibration requires a GPU lock; call sidedgpu_acquire_gpu first"
        return 1
    fi
    sidedgpu_ensure_dir "${config_dir}" "${bin_dir}" || status=$?
    (( status != 0 )) || sidedgpu_build_side_map_tool "${bin_dir}" "${arch}" || status=$?
    (( status != 0 )) || SIDEDGPU_CONFIG_DIR="${config_dir}" sidedgpu_run_gpu "${bin_dir}/get_smid2side" || status=$?
    if (( status == 0 )) && [[ "${SIDEDGPU_DRY_RUN:-0}" != 1 ]] && [[ ! -s "${config_dir}/smid2side.csv" ]]; then
        sidedgpu_die "side-map calibration did not produce smid2side.csv under ${config_dir}" || true
        status=1
    fi
    return "${status}"
}

sidedgpu_prepare_calibration() {
    local config_dir="$1" bin_dir="$2" arch="$3" status=0
    if [[ "${SIDEDGPU_DRY_RUN:-0}" != 1 && ( -z "${SIDEDGPU_GPU_INDEX:-}" || -z "${SIDEDGPU_GPU_LOCK_FD:-}" ) ]]; then
        sidedgpu_die "calibration requires a GPU lock; call sidedgpu_acquire_gpu first"
        return 1
    fi
    sidedgpu_ensure_dir "${config_dir}" "${bin_dir}" || status=$?
    (( status != 0 )) || sidedgpu_build_calibration_tools "${bin_dir}" "${arch}" || status=$?
    (( status != 0 )) || SIDEDGPU_CONFIG_DIR="${config_dir}" sidedgpu_run_gpu "${bin_dir}/get_smid2side" || status=$?
    (( status != 0 )) || SIDEDGPU_CONFIG_DIR="${config_dir}" sidedgpu_run_gpu "${bin_dir}/get_mask" || status=$?
    if (( status == 0 )) && [[ "${SIDEDGPU_DRY_RUN:-0}" != 1 ]] \
        && [[ ! -s "${config_dir}/smid2side.csv" || ! -s "${config_dir}/mask.csv" ]]; then
        sidedgpu_die "calibration did not produce smid2side.csv and mask.csv under ${config_dir}" || true
        status=1
    fi
    return "${status}"
}

# Parse the architecture CLI once. Results are shell globals and are never exported.
sidedgpu_parse_entry_options() {
    SIDEDGPU_OPT_ORIGINAL_ARGS=("$@")
    SIDEDGPU_OPT_DEVICE=auto
    SIDEDGPU_OPT_ARCH=auto
    SIDEDGPU_OPT_OUTPUT_DIR=""
    SIDEDGPU_OPT_BUILD_ONLY=0
    SIDEDGPU_OPT_DRY_RUN=0
    SIDEDGPU_OPT_WAIT_TIMEOUT="${SIDEDGPU_DEFAULT_WAIT_TIMEOUT}"
    SIDEDGPU_OPT_POLL_INTERVAL="${SIDEDGPU_DEFAULT_POLL_INTERVAL}"
    SIDEDGPU_OPT_PEER=auto
    SIDEDGPU_OPT_HELP=0
    while (($#)); do
        case "$1" in
            --device|--arch|--output-dir|--wait-timeout|--poll-interval|--peer)
                if [[ $# -lt 2 ]]; then
                    sidedgpu_die "$1 requires a value" || true
                    return 2
                fi
                local option="$1" value="$2"
                shift 2
                ;;
            --device=*|--arch=*|--output-dir=*|--wait-timeout=*|--poll-interval=*|--peer=*)
                local option="${1%%=*}" value="${1#*=}"
                shift
                ;;
            --build-only) SIDEDGPU_OPT_BUILD_ONLY=1; shift; continue ;;
            --dry-run) SIDEDGPU_OPT_DRY_RUN=1; shift; continue ;;
            -h|--help) SIDEDGPU_OPT_HELP=1; return 0 ;;
            --)
                if [[ $# -eq 1 ]]; then shift; continue; fi
                sidedgpu_die "unknown option: $1" || true
                return 2
                ;;
            *)
                sidedgpu_die "unknown option: $1" || true
                return 2
                ;;
        esac
        case "${option}" in
            --device) SIDEDGPU_OPT_DEVICE="${value}" ;;
            --arch) SIDEDGPU_OPT_ARCH="${value}" ;;
            --output-dir) SIDEDGPU_OPT_OUTPUT_DIR="${value}" ;;
            --wait-timeout) SIDEDGPU_OPT_WAIT_TIMEOUT="${value}" ;;
            --poll-interval) SIDEDGPU_OPT_POLL_INTERVAL="${value}" ;;
            --peer) SIDEDGPU_OPT_PEER="${value}" ;;
        esac
    done
    sidedgpu_validate_device_selector "${SIDEDGPU_OPT_DEVICE}" || return $?
    if [[ "${SIDEDGPU_OPT_ARCH}" != auto ]]; then
        sidedgpu_resolve_arch "${SIDEDGPU_OPT_ARCH}" >/dev/null || return $?
    fi
    if ! _sidedgpu_is_uint "${SIDEDGPU_OPT_WAIT_TIMEOUT}"; then
        sidedgpu_die "--wait-timeout must be a non-negative integer" || true
        return 2
    fi
    if ! _sidedgpu_is_uint "${SIDEDGPU_OPT_POLL_INTERVAL}"; then
        sidedgpu_die "--poll-interval must be a non-negative integer" || true
        return 2
    fi
    if [[ "${SIDEDGPU_OPT_PEER}" != auto && ! "${SIDEDGPU_OPT_PEER}" =~ ^[0-9]+,[0-9]+$ ]]; then
        sidedgpu_die "--peer must be auto or A,B with non-negative SM IDs" || true
        return 2
    fi
}

sidedgpu_begin_gpu_context() {
    local device="${1:-${SIDEDGPU_OPT_DEVICE:-auto}}"
    local arch="${2:-${SIDEDGPU_OPT_ARCH:-auto}}"
    local dry_run="${3:-${SIDEDGPU_OPT_DRY_RUN:-0}}"
    local wait_timeout="${4:-${SIDEDGPU_OPT_WAIT_TIMEOUT:-1800}}"
    local poll_interval="${5:-${SIDEDGPU_OPT_POLL_INTERVAL:-10}}"

    SIDEDGPU_DRY_RUN="${dry_run}"
    if (( dry_run )); then
        SIDEDGPU_GPU_INDEX="${device}"
        [[ "${SIDEDGPU_GPU_INDEX}" != auto ]] || SIDEDGPU_GPU_INDEX="<auto-selected-physical-index>"
        SIDEDGPU_GPU_UUID=dry-run
        SIDEDGPU_GPU_NAME=""
        if [[ "${arch}" == auto ]]; then
            SIDEDGPU_GPU_ARCH="${SIDEDGPU_DRY_RUN_AUTO_ARCH}"
        else
            SIDEDGPU_GPU_ARCH="$(sidedgpu_resolve_arch "${arch}")" || return
        fi
    else
        sidedgpu_acquire_gpu "${device}" "${wait_timeout}" "${poll_interval}" || return
        trap 'sidedgpu_release_gpu' EXIT
        SIDEDGPU_GPU_ARCH="$(sidedgpu_resolve_arch "${arch}")" || return
    fi
}

sidedgpu_write_metadata() {
    local output="$1" extra
    shift
    if [[ "${SIDEDGPU_DRY_RUN:-0}" == 1 ]]; then
        sidedgpu_log "dry-run: would write metadata to ${output}"
        return 0
    fi
    local python smi=""
    python="$(sidedgpu_python)" || return
    [[ -f "${SIDEDGPU_SUPPORT_PY}" ]] || { sidedgpu_die "missing support helper: ${SIDEDGPU_SUPPORT_PY}"; return 1; }
    smi="$(command -v nvidia-smi 2>/dev/null || true)"
    local -a command=(
        "${python}" "${SIDEDGPU_SUPPORT_PY}" write-metadata
        "--output=${output}"
        "--repo-root=${SIDEDGPU_ROOT_DIR}"
        "--command=${SIDEDGPU_METADATA_COMMAND:-}"
        "--gpu-index=${SIDEDGPU_GPU_INDEX:-}"
        "--gpu-uuid=${SIDEDGPU_GPU_UUID:-}"
        "--gpu-name=${SIDEDGPU_GPU_NAME:-}"
        "--gpu-arch=${SIDEDGPU_GPU_ARCH:-}"
        "--gpu-memory=${SIDEDGPU_GPU_MEMORY_USED_MIB:-}"
        "--gpu-util=${SIDEDGPU_GPU_MAX_UTIL:-}"
        "--no-compute=${SIDEDGPU_GPU_NO_COMPUTE_PROCESSES:-}"
        "--samples=${SIDEDGPU_IDLE_SAMPLES}"
        "--max-memory=${SIDEDGPU_IDLE_MAX_MEMORY_MIB}"
        "--max-util=${SIDEDGPU_IDLE_MAX_UTIL}"
        "--lock-path=${SIDEDGPU_GPU_LOCK_PATH:-}"
        "--lock-held=${SIDEDGPU_GPU_LOCK_HELD:-}"
        "--nvidia-smi=${smi}"
    )
    for extra in "$@"; do command+=("--extra=${extra}"); done
    "${command[@]}"
}

_sidedgpu_lock_path_for_uuid() {
    local safe="${1//[^A-Za-z0-9_.-]/_}"
    printf '/tmp/sidedgpu-gpu-%s.lock\n' "${safe}"
}

_sidedgpu_close_fd() {
    [[ "$1" =~ ^[0-9]+$ ]] || return 0
    flock -u "$1" >/dev/null 2>&1 || true
    eval "exec $1>&-"
}

_sidedgpu_snapshot() {
    local selector="$1" smi python
    smi="$(command -v nvidia-smi)" || return 1
    if [[ ! -x "${smi}" ]]; then
        sidedgpu_die "nvidia-smi is not executable: ${smi}"
        return 1
    fi
    python="$(sidedgpu_python)" || return 1
    "${python}" "${SIDEDGPU_SUPPORT_PY}" gpu-snapshot \
        "--nvidia-smi=${smi}" "--selector=${selector}" \
        "--samples=${SIDEDGPU_IDLE_SAMPLES}" \
        "--max-memory=${SIDEDGPU_IDLE_MAX_MEMORY_MIB}" \
        "--max-util=${SIDEDGPU_IDLE_MAX_UTIL}"
}

_sidedgpu_print_gpu_diagnostics() {
    local smi python records
    smi="$(command -v nvidia-smi)" || { sidedgpu_warn "GPU diagnostics unavailable"; return 0; }
    [[ -x "${smi}" ]] || { sidedgpu_warn "GPU diagnostics unavailable"; return 0; }
    python="$(sidedgpu_python)" || { sidedgpu_warn "GPU diagnostics unavailable"; return 0; }
    if ! records="$("${python}" "${SIDEDGPU_SUPPORT_PY}" gpu-diagnostics "--nvidia-smi=${smi}")"; then
        sidedgpu_warn "GPU diagnostics unavailable"
        return 0
    fi
    sidedgpu_warn "GPU state at timeout:"
    local index uuid name memory util pids path fd state
    while IFS=$'\t' read -r index uuid name memory util pids; do
        [[ -n "${index}" ]] || continue
        path="$(_sidedgpu_lock_path_for_uuid "${uuid}")"
        exec {fd}>"${path}"
        if flock -n "${fd}"; then state=free; else state=held; fi
        _sidedgpu_close_fd "${fd}"
        sidedgpu_warn "  index=${index} name='${name}' memory=${memory}MiB util=${util}% compute_pids=${pids} lock=${state} lock_path=${path}"
    done <<<"${records}"
}

_sidedgpu_deadline_reached() {
    local started="$1" timeout="$2"
    (( timeout > 0 && $(date +%s) - started >= timeout ))
}

_sidedgpu_report_timeout() {
    local selector="$1" started="$2" elapsed
    elapsed=$(($(date +%s) - started))
    _sidedgpu_print_gpu_diagnostics
    sidedgpu_warn "timed out after ${elapsed}s waiting for idle GPU '${selector}'"
}

sidedgpu_acquire_gpu() {
    local selector="${1:-${SIDEDGPU_DEFAULT_DEVICE}}"
    local timeout="${2:-${SIDEDGPU_DEFAULT_WAIT_TIMEOUT}}"
    local poll="${3:-${SIDEDGPU_DEFAULT_POLL_INTERVAL}}"
    sidedgpu_validate_device_selector "${selector}" || return
    if ! _sidedgpu_is_uint "${timeout}"; then sidedgpu_die "wait timeout must be a non-negative integer number of seconds"; return 2; fi
    if ! _sidedgpu_is_uint "${poll}"; then sidedgpu_die "poll interval must be a non-negative integer number of seconds"; return 2; fi
    if [[ -n "${SIDEDGPU_GPU_LOCK_FD:-}" ]]; then sidedgpu_die "a GPU lock is already held by this shell"; return 1; fi
    sidedgpu_require_command flock || return
    local smi_path
    smi_path="$(command -v nvidia-smi)" || { sidedgpu_die "nvidia-smi not found"; return 1; }
    [[ -x "${smi_path}" ]] || { sidedgpu_die "nvidia-smi is not executable: ${smi_path}"; return 1; }
    sidedgpu_python >/dev/null || return

    local started attempt_started now elapsed remaining sleep_for status candidates
    started="$(date +%s)"
    while true; do
        attempt_started="$(date +%s)"
        if candidates="$(_sidedgpu_snapshot "${selector}")"; then status=0; else status=$?; fi
        (( status != 2 )) || return 2
        if _sidedgpu_deadline_reached "${started}" "${timeout}"; then
            _sidedgpu_report_timeout "${selector}" "${started}"
            return 1
        fi

        local memory util index uuid name path fd rechecked re_memory re_util re_index re_uuid re_name
        while IFS=$'\t' read -r memory util index uuid name; do
            [[ -n "${index}" ]] || continue
            path="$(_sidedgpu_lock_path_for_uuid "${uuid}")"
            exec {fd}>"${path}"
            if ! flock -n "${fd}"; then
                _sidedgpu_close_fd "${fd}"
                if _sidedgpu_deadline_reached "${started}" "${timeout}"; then
                    _sidedgpu_report_timeout "${selector}" "${started}"
                    return 1
                fi
                continue
            fi
            if rechecked="$(_sidedgpu_snapshot "${index}")"; then
                IFS=$'\t' read -r re_memory re_util re_index re_uuid re_name <<<"${rechecked}"
                if _sidedgpu_deadline_reached "${started}" "${timeout}"; then
                    _sidedgpu_close_fd "${fd}"
                    _sidedgpu_report_timeout "${selector}" "${started}"
                    return 1
                fi
                if [[ "${re_index}" == "${index}" && "${re_uuid}" == "${uuid}" ]]; then
                    SIDEDGPU_GPU_INDEX="${re_index}"
                    SIDEDGPU_GPU_UUID="${re_uuid}"
                    SIDEDGPU_GPU_NAME="${re_name}"
                    SIDEDGPU_GPU_MEMORY_USED_MIB="${re_memory}"
                    SIDEDGPU_GPU_MAX_UTIL="${re_util}"
                    SIDEDGPU_GPU_LOCK_FD="${fd}"
                    SIDEDGPU_GPU_LOCK_PATH="${path}"
                    SIDEDGPU_GPU_NO_COMPUTE_PROCESSES=true
                    SIDEDGPU_GPU_LOCK_HELD=true
                    SIDEDGPU_GPU_ARCH="$(sidedgpu_resolve_arch auto 2>/dev/null || printf unknown)"
                    sidedgpu_log "locked GPU ${re_index} (${re_name}, ${re_uuid}, ${SIDEDGPU_GPU_ARCH}); memory=${re_memory} MiB, max-util=${re_util}%"
                    return 0
                fi
            fi
            _sidedgpu_close_fd "${fd}"
            if _sidedgpu_deadline_reached "${started}" "${timeout}"; then
                _sidedgpu_report_timeout "${selector}" "${started}"
                return 1
            fi
        done <<<"${candidates}"

        if (( timeout == 0 )) || _sidedgpu_deadline_reached "${started}" "${timeout}"; then
            _sidedgpu_report_timeout "${selector}" "${started}"
            return 1
        fi
        now="$(date +%s)"
        elapsed=$((now - started))
        remaining=$((timeout - elapsed))
        sleep_for=$((poll - (now - attempt_started)))
        (( sleep_for >= 0 )) || sleep_for=0
        (( sleep_for <= remaining )) || sleep_for="${remaining}"
        sidedgpu_log "no strictly idle unlocked GPU matched '${selector}'; retrying in ${sleep_for}s (${remaining}s remaining)"
        (( sleep_for == 0 )) || sleep "${sleep_for}"
    done
}

sidedgpu_release_gpu() {
    [[ -z "${SIDEDGPU_GPU_LOCK_FD:-}" ]] || _sidedgpu_close_fd "${SIDEDGPU_GPU_LOCK_FD}"
    SIDEDGPU_GPU_INDEX=""
    SIDEDGPU_GPU_UUID=""
    SIDEDGPU_GPU_NAME=""
    SIDEDGPU_GPU_ARCH=""
    SIDEDGPU_GPU_LOCK_FD=""
    SIDEDGPU_GPU_LOCK_PATH=""
    SIDEDGPU_GPU_MEMORY_USED_MIB=""
    SIDEDGPU_GPU_MAX_UTIL=""
    SIDEDGPU_GPU_NO_COMPUTE_PROCESSES=""
    SIDEDGPU_GPU_LOCK_HELD=""
}
