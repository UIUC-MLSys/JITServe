#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_DIR="${ROOT_DIR}/third_party/vllm"
EXPECTED_VENV="${ROOT_DIR}/.venv"
DEFAULT_VLLM_COMMIT="32176fee733b76b295346870d717d44cb7102944"
VLLM_COMMIT="${VLLM_COMMIT:-$DEFAULT_VLLM_COMMIT}"

usage() {
  cat <<'EOF'
Usage: scripts/setup_vllm.sh

The script creates/uses .venv in the repository root and installs through uv.

Environment:
  VLLM_COMMIT     Override the vLLM commit used for the prebuilt wheel.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is required but not found. Install it first: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

if [[ ! -d "$VLLM_DIR" ]]; then
  echo "Error: vLLM directory not found: $VLLM_DIR" >&2
  exit 1
fi

if [[ ! -d "$EXPECTED_VENV" ]]; then
  (cd "$ROOT_DIR" && uv venv --seed)
fi

PYTHON_BIN="${EXPECTED_VENV}/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Error: python not found in venv: $PYTHON_BIN" >&2
  exit 1
fi

pushd "$VLLM_DIR" >/dev/null

echo "Installing prebuilt vLLM wheel for commit: $VLLM_COMMIT"
UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install \
  --python "$PYTHON_BIN" \
  "https://vllm-wheels.s3.us-west-2.amazonaws.com/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl"

mkdir -p vllm/vllm_flash_attn
"$PYTHON_BIN" python_only_dev.py

popd >/dev/null

pushd "$ROOT_DIR" >/dev/null
uv pip install --python "$PYTHON_BIN" -e .
popd >/dev/null

echo "JITServe editable install completed."
