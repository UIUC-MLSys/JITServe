## Overview

This repository contains the artifact for [**JITServe** NSDI'26](https://arxiv.org/abs/2504.20068), an SLO-aware
LLM serving system designed to support heterogeneous workloads.

The artifact is organized to support **reproducible evaluation** of the
scheduling and preemption mechanisms described in the paper.

## Structure Overview

- `jitserve/`: SLO tracking, scheduling, and server entrypoint.
- `third_party/vllm/`: vendored vLLM snapshot with minimal integration hooks.
- `scripts/`: benchmark scripts and experiment runners.
- `benchmark/`: trace generate tools, and benchmark clients.
- `traces/`: datasets and traces.
- `assets/`: qrf models and vectorizer.


## Getting Started

### Prerequisites
- Python 3.10+
- CUDA 12+
- No preinstalled vLLM (to avoid import conflicts)

If you have an existing vLLM installation, please remove it first:
```bash
pip uninstall vllm -y
```

### Download Required Assets (QRF Predictors)

Before installation, please download the required models and prediction
artifacts following the instructions in:
```
assets/README.md
```

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

### Install JITServe

From the repository root, install JITServe as an editable package:
```bash
pip install -e .
```

This installs JITServe into the Python environment and exposes it as a
package.

### Run Experiments
1. Start QRF Prediction Server:
```Bash
python jitserve/request_analyzer/prediction.py
```

Note on Other Scripts: Additional experiment scripts are available in the scripts/ directory. For ease of reproduction, the two experiments listed below are selected as they require the minimum time and computing resources while demonstrating the core functionality of JITServe.

2. End-to-End Oracle Comparison (Section 6.2, Figure 13)

```Bash
# 1. Run benchmark (3 hours)
bash scripts/e2e/benchmark_oracle.sh

# 2. Generate figure at figure/e2e/plot_oracle
python -m figure.e2e.plot_oracle
```

3. Ablation with SLO-target baseline (Section 6.4, Figure 21)

```Bash
# 1. Run benchmark (3 hours)
bash scripts/ablation/benchmark_slosserve.sh

# 2. Generate figure at figure/ablation/plot_slosserve
python -m figure.ablation.plot_slosserve
```

Notes:
- QRF model paths default to `assets/qrf/`.
- Model weights are not included in this repository due to licensing
and size constraints.
- More benchmark and plot scripts will be released in the future.

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

## License

This project is released under the Apache License 2.0.
See the [LICENSE](LICENSE) file for details.