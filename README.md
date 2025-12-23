## Overview

This repository contains the artifact for [**JITServe** NSDI'26](https://arxiv.org/abs/2504.20068), an SLO-aware
LLM serving system designed to support heterogeneous workloads.

The artifact is organized to support **reproducible evaluation** of the
scheduling and preemption mechanisms described in the paper.


## Getting Started

### Prerequisites
- Python 3.10+
- CUDA 12+
- No preinstalled vLLM (to avoid import conflicts)

If you have an existing vLLM installation, please remove it first:
```bash
pip uninstall vllm -y
```
---

### Download Required Assets (QRF Predictors)

Before installation, please download the required models and prediction
artifacts following the instructions in:
```
assets/README.md
```
---

### vLLM Dependency (Vendored)

JITServe relies on a vendored and locally modified snapshot of vLLM
located at:
```
third_party/vllm/
```

This vLLM copy is NOT installed automatically

Please follow the instructions in:
```
third_party/vllm/README.md
```

⚠️ Do NOT install vLLM via pip install vllm, as upstream versions are
incompatible with JITServe’s scheduler integration.

---

### Install JITServe

From the repository root, install JITServe as an editable package:
```bash
pip install -e .
```

This installs JITServe into the Python environment and exposes it as a
package.

## Structure Overview

- `jitserve/`: SLO tracking, scheduling, and server entrypoint.
- `third_party/vllm/`: vendored vLLM snapshot with minimal integration hooks.
- `scripts/`: benchmark scripts and experiment runners.
- `benchmark/`: trace generate tools, and benchmark clients.
- `traces/`: datasets and traces.
- `assets/`: qrf models and vectorizer.

## Run Experiments

QRF prediction server:
```bash
python jitserve/request_analyzer/prediction.py
```

Mixed workload benchmark:
```bash
bash scripts/benchmark_mixed.sh
```

Notes:
- QRF model paths default to `assets/qrf/`.
- Model weights are not included in this repository due to licensing
and size constraints.

## Citation
If you use these artifacts, please consider to cite our paper:
```
@misc{zhang2025jitservesloawarellmserving,
      title={JITServe: SLO-aware LLM Serving with Imprecise Request Information}, 
      author={Wei Zhang and Zhiyu Wu and Yi Mu and Rui Ning and Banruo Liu and Nikhil Sarda and Myungjin Lee and Fan Lai},
      year={2025},
      eprint={2504.20068},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2504.20068}, 
}
```

## Contact
Wei Zhang [(zhangw2@illinois.edu)](zhangw2@illinois.edu) and Zhiyu Wu [(zhiyuwu2@illinois.edu)](zhiyuwu2@illinois.edu)