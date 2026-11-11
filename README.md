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

### Install (vendored vLLM + JITServe)

JITServe relies on a vendored and locally modified snapshot of vLLM in
`third_party/vllm/`. One script installs it (prebuilt wheel, no CUDA build)
together with JITServe into `.venv` using [uv](https://docs.astral.sh/uv/):
```bash
bash scripts/setup_vllm.sh      # creates .venv with Python 3.10
source .venv/bin/activate
python assets/download_qrf.py   # QRF predictors, see assets/README.md
```

⚠️ Do NOT install vLLM via pip install vllm, as upstream versions are
incompatible with JITServe’s scheduler integration.

### Run Experiments
The QRF prediction server (`python -m jitserve.request_analyzer.prediction`)
is started automatically by `jitserve.server` when it is not already running.
The experiment scripts wait for `/health` before sending requests.

1. Script-to-figure map

| Script | Figure |
|---|---|
| `scripts/e2e/benchmark_timeline.sh` | Fig 11, 12 (online timeline) |
| `scripts/e2e/benchmark_oracle.sh` | Fig 13 |
| `scripts/e2e/benchmark_throughput.sh` | Fig 14 |
| `scripts/e2e/benchmark_rps.sh` | Fig 15 (load sweep) |
| `scripts/e2e/benchmark_multi_replica.sh` | Fig 18 (multiple replicas, power-of-K) |
| `scripts/ablation/benchmark_slosserve.sh` | Fig 21 |

`benchmark_timeline.sh`, `benchmark_rps.sh` and `benchmark_multi_replica.sh`
run every policy with the same `--max-num-seqs 128`; JITServe additionally
uses a GMAX window of `--gmax-window 32`. The timeline and RPS scripts run one
policy at a time on a single GPU.

2. Recommended runs (Llama-3.1-8B-Instruct, one A100)

These are the shortest runs that show the main result. With the updated code
they reproduce the paper's ranking: JITServe > LTR >> Sarathi-Serve ≈ Autellix > vLLM.
`meta-llama/Llama-3.1-8B-Instruct` is gated on Hugging Face. Log in first, or
point `model=` at a local copy.

```Bash
# Online timeline (Section 6.2, Figures 11 and 12), 20 minutes of arrivals at
# RPS 6 for each of the 5 policies (~30 min per policy, including the drain).
NUM_PROMPTS=7200 bash scripts/e2e/benchmark_timeline.sh
python -m figure.e2e.plot_tokengpt_timeline
python -m figure.e2e.plot_requestgpt_timeline

# The paper's full timeline: 13000 prompts, about 36 min of arrivals per policy.
bash scripts/e2e/benchmark_timeline.sh

# Single load point of Figure 15 (RPS 6, 3000 prompts, ~15 min per policy).
# Drop RATES to run the full sweep 4.0-6.5.
RATES=6.0 bash scripts/e2e/benchmark_rps.sh
```

3. End-to-End Oracle Comparison (Section 6.2, Figure 13)

```Bash
# 1. Run benchmark (3 hours)
bash scripts/e2e/benchmark_oracle.sh

# 2. Generate figure at figure/e2e/plot_oracle
python -m figure.e2e.plot_oracle
```

4. Ablation with SLO-target baseline (Section 6.4, Figure 21)

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