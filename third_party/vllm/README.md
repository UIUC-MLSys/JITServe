# vLLM (Vendored Dependency)

This directory contains a **vendored snapshot of vLLM** used by
JITServe for artifact evaluation.

⚠️ **Important**
- Please **DO NOT** run `pip install vllm` or upgrade vLLM via pip.
- The code under `third_party/vllm/` is used **only through JITServe**.

## Versioning

This artifact vendors vLLM at commit: [32176fe (2024-10-27)](https://github.com/vllm-project/vllm/commit/32176fe). The snapshot is included verbatim, except for
**minimal integration hooks** required by JITServe.

## Modifications

Changes relative to upstream vLLM are limited to:
- Import hooks allowing JITServe to register its scheduler
- Lightweight scheduler interface extensions

All JITServe-specific scheduling logic resides in the top-level
`jitserve/` directory and **not** in vLLM.

## Installation

Use the setup script from the JITServe repository root:

```bash
./scripts/setup_vllm.sh
```

If you want the script to create a virtual environment first:

```bash
./scripts/setup_vllm.sh --create-venv
```

The manual step-by-step process is:

1. [Optional] Install `uv` from https://docs.astral.sh/uv/getting-started/installation/#installation-methods and create a venv:
   ```bash
   uv venv --seed
   ```
2. Enter the vendored vLLM directory:
   ```bash
   cd third_party/vllm
   ```
3. Set the expected vLLM commit:
   ```bash
   export VLLM_COMMIT=32176fee733b76b295346870d717d44cb7102944
   ```
4. Install the wheel built at that commit:
   ```bash
   UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install \
     https://vllm-wheels.s3.us-west-2.amazonaws.com/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
   ```
5. Create the local flash-attn stub directory:
   ```bash
   mkdir -p vllm/vllm_flash_attn
   ```
6. Run:
   ```bash
   python python_only_dev.py
   ```
   This script will:
   - find the installed vLLM package in the current environment
   - copy built files to this local directory
   - rename the installed vLLM package
   - symbolically link this local directory to the installed vLLM package
7. Return to JITServe root and install JITServe editable:
   ```bash
   cd ../..
   uv pip install -e .
   ```

## Pulling Upstream vLLM Updates (For Developers Only)
We intend to synchronize with upstream vLLM in the future.

To pull upstream updates:
```bash
git subtree pull --prefix=third_party/vllm upstream main --squash
```

If upstream uses a different branch, replace `main` with the branch name.
