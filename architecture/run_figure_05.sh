#!/usr/bin/env bash
set -euo pipefail

ARCHITECTURE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${ARCHITECTURE_DIR}/_run_figure.sh" 5 "$0" "$@"
