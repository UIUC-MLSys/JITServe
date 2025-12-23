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

Install vLLM from this directory:

```bash
pip install -e .
```

## Pulling Upstream vLLM Updates (For Developers Only)
We intend to synchronize with upstream vLLM in the future.

To pull upstream updates:
```bash
git subtree pull --prefix=third_party/vllm upstream main --squash
```

If upstream uses a different branch, replace `main` with the branch name.