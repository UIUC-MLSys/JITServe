# Traces

This directory contains request arrival traces used for evaluating **JITServe** under
mixed SLO workloads. All traces are **trace-driven**, reproducible, and configurable.

---

## Overview

We generate evaluation workloads by re-labeling requests from a real-world
LMSYS trace (`lmsys.json`) to construct mixtures of:

- **Latency-sensitive requests**
- **Throughput-oriented requests**
- **Collective (compound) requests**, including Tree-of-Thought (ToT) and
  DeepResearch-style workloads

The resulting traces are used to stress-test SLO-aware scheduling under
heterogeneous and realistic request patterns.

---

## Source Trace

- `lmsys.json`  
  A base trace derived from real-world chatbot workloads ([lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)).
  Each entry provides a realistic prompt and request template.

This base trace is **not modified structurally**; instead, requests are
re-labeled and re-composed to form mixed workloads.

---

## Trace Generation

Traces are generated using the script:

```bash
benchmarks/create_trace.py --num_prompts 1000 --request_ratio 3,5,2
```
### Parameters
--num_prompts N
Total number of requests in the generated trace.

--request_ratio a,b,c
Ratio of latency : throughput : collective requests.

## Collective and DeepResearch Requests

The `create_trace.py` script only explicitly generates latency-sensitive, throughput-oriented, and ToT-style requests. DeepResearch-style requests are not included in the generated trace file.

Instead, DeepResearch workloads are specified separately using:
`deepresearch_filter.jsonl`

During evaluation, the DeepResearch trace is loaded independently and executed through a dedicated DeepResearch pipeline. The DeepResearch requests are then combined with the generated trace at runtime.

Together, the generated trace (latency / throughput / ToT) and the
DeepResearch trace form a complete workload containing exactly
`num_prompts` requests.

This separation reflects the fact that DeepResearch workloads involve distinct execution semantics and DAG-style dependencies that are better handled by a specialized execution pipeline.
