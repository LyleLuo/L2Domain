#!/usr/bin/env bash
# Shared application-runner lifecycle. This file is sourced by applications/*/run.sh.

if [[ -n "${SIDEDGPU_APP_RUNNER_SH_LOADED:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi
SIDEDGPU_APP_RUNNER_SH_LOADED=1

if [[ -z "${SIDEDGPU_COMMON_SH_LOADED:-}" ]]; then
  SIDEDGPU_APP_RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  # shellcheck source=common.sh
  source "${SIDEDGPU_APP_RUNNER_DIR}/common.sh"
fi

sidedgpu_app_print_common_options() {
  cat <<'EOF'
  --device auto|INDEX|UUID  GPU selection (default: auto)
  --preset quick|paper     Smoke point or paper sweep (default: quick)
EOF
  if (( $# > 0 )); then
    printf '%s\n' "$@"
  fi
  cat <<'EOF'
  --build-only             Compile without running
  --dry-run                Print commands without executing them
  --arch auto|sm_XX        Override CUDA architecture detection
  --wait-timeout SECONDS   Wait for a free GPU (default: 1800)
  --poll-interval SECONDS  GPU polling interval (default: 10)
  --output-dir PATH        Runtime output directory (default: out)
EOF
}

sidedgpu_app_require_value() {
  if (( $# < 2 )); then
    echo "missing value for $1" >&2
    exit 2
  fi
}

# Parse the options common to every application. An application may set
# SIDEDGPU_APP_OPTION_PARSER to a function that consumes an application-specific
# option and stores the positive token count in SIDEDGPU_APP_OPTION_CONSUMED.
sidedgpu_app_parse_options() {
  local usage_function="$1"
  shift

  DEVICE="auto"
  PRESET="quick"
  BUILD_ONLY=0
  DRY_RUN=0
  ARCH_OVERRIDE="auto"
  WAIT_TIMEOUT="${SIDEDGPU_DEFAULT_WAIT_TIMEOUT}"
  POLL_INTERVAL="${SIDEDGPU_DEFAULT_POLL_INTERVAL}"
  OUTPUT_ARG="out"

  while (( $# > 0 )); do
    case "$1" in
      --device) sidedgpu_app_require_value "$@"; DEVICE="$2"; shift 2 ;;
      --preset) sidedgpu_app_require_value "$@"; PRESET="$2"; shift 2 ;;
      --build-only) BUILD_ONLY=1; shift ;;
      --dry-run) DRY_RUN=1; shift ;;
      --arch) sidedgpu_app_require_value "$@"; ARCH_OVERRIDE="$2"; shift 2 ;;
      --wait-timeout) sidedgpu_app_require_value "$@"; WAIT_TIMEOUT="$2"; shift 2 ;;
      --poll-interval) sidedgpu_app_require_value "$@"; POLL_INTERVAL="$2"; shift 2 ;;
      --output-dir) sidedgpu_app_require_value "$@"; OUTPUT_ARG="$2"; shift 2 ;;
      -h|--help) "${usage_function}"; exit 0 ;;
      *)
        SIDEDGPU_APP_OPTION_CONSUMED=0
        if [[ -n "${SIDEDGPU_APP_OPTION_PARSER:-}" ]] \
          && "${SIDEDGPU_APP_OPTION_PARSER}" "$@"; then
          if ! _sidedgpu_is_uint "${SIDEDGPU_APP_OPTION_CONSUMED}" \
            || (( SIDEDGPU_APP_OPTION_CONSUMED == 0 || SIDEDGPU_APP_OPTION_CONSUMED > $# )); then
            sidedgpu_die "application option parser returned an invalid consumed-token count"
            return 2
          fi
          shift "${SIDEDGPU_APP_OPTION_CONSUMED}"
        else
          echo "unknown option: $1" >&2
          "${usage_function}" >&2
          exit 2
        fi
        ;;
    esac
  done

  if ! _sidedgpu_is_uint "${WAIT_TIMEOUT}"; then
    echo "--wait-timeout must be a non-negative integer" >&2
    exit 2
  fi
  if ! _sidedgpu_is_uint "${POLL_INTERVAL}"; then
    echo "--poll-interval must be a non-negative integer" >&2
    exit 2
  fi
  if ! sidedgpu_validate_device_selector "${DEVICE}"; then
    exit 2
  fi
  if [[ -z "${OUTPUT_ARG}" ]]; then
    echo "--output-dir must not be empty" >&2
    exit 2
  fi
  if [[ "${ARCH_OVERRIDE}" != auto ]] \
    && ! sidedgpu_resolve_arch "${ARCH_OVERRIDE}" >/dev/null; then
    exit 2
  fi
}

# Acquire and hold the GPU lock, resolve paths/architecture, and write the
# metadata common to every application. Additional arguments are metadata
# key=value pairs (for example, ReLU's selected suite).
sidedgpu_app_begin() {
  local app_dir="$1"
  local application="$2"
  shift 2

  sidedgpu_begin_gpu_context \
    "${DEVICE}" "${ARCH_OVERRIDE}" "${DRY_RUN}" \
    "${WAIT_TIMEOUT}" "${POLL_INTERVAL}"
  CUDA_ARCH="${SIDEDGPU_GPU_ARCH}"

  OUT_DIR="$(sidedgpu_resolve_path "${app_dir}" "${OUTPUT_ARG}")"
  BIN_DIR="${OUT_DIR}/bin"
  sidedgpu_ensure_dir "${OUT_DIR}" "${BIN_DIR}"

  case "${SIDEDGPU_GPU_NAME:-unknown}" in
    *A100*) GPU_LABEL="a100" ;;
    *H100*) GPU_LABEL="h100" ;;
    *" B200"*) GPU_LABEL="b200" ;;
    *) GPU_LABEL="custom" ;;
  esac

  sidedgpu_write_metadata "${OUT_DIR}/metadata.json" \
    "application=${application}" "preset=${PRESET}" "$@"
}

sidedgpu_app_exit_on_failures() {
  local failures="${1:-0}"
  if (( failures > 0 )); then
    echo "${failures} benchmark case(s) failed" >&2
    exit 1
  fi
}
