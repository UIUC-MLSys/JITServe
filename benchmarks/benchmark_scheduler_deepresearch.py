"""Benchmark scheduler for DeepResearch collective requests."""
import argparse
import asyncio
import copy
import json
import numpy as np
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Collection, Dict, List, Optional, Tuple
from pathlib import Path
from tqdm.asyncio import tqdm as async_tqdm
from transformers import PreTrainedTokenizerBase
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind

# Import DeepResearch modules
from trace_deepresearch import (
    DeepResearchTrace, DeepResearchCollectiveRequest, RequestFormat, RequestType
)
from client_deepresearch import (
    deepresearch_client_simulator, CollectiveOutput, StageOutput, RequestOutput
)

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer
try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser


@dataclass
class DeepResearchMetrics:
    duration: float
    total_collections: int
    completed_collections: int
    failed_collections: int
    total_stages: int
    completed_stages: int
    failed_stages: int
    total_requests: int
    completed_requests: int
    failed_requests: int
    total_input_tokens: int
    total_output_tokens: int
    input_throughput: float
    output_throughput: float
    total_throughput: float
    mean_collection_latency: float
    median_collection_latency: float
    percentiles_collection_latency: List[Tuple[float, float]]
    mean_stage_latency: float
    median_stage_latency: float
    percentiles_stage_latency: List[Tuple[float, float]]
    mean_request_latency: float
    median_request_latency: float
    percentiles_request_latency: List[Tuple[float, float]]
    mean_ttft_ms: float
    median_ttft_ms: float
    percentiles_ttft_ms: List[Tuple[float, float]]
    slo_met_collections: int
    slo_met_ratio: float
    slo_met_requests: int  # Number of individual requests that met SLO
    slo_goodput: float  # SLO goodput: requests meeting SLO per second
    total_service_gain: float


def calculate_deepresearch_metrics(
    collective_outputs: List[CollectiveOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentiles: List[float],
    slo_constraint: Optional[Tuple[float, float, float]] = None,
) -> DeepResearchMetrics:
    """Calculate metrics for DeepResearch collective requests."""
    
    # Initialize counters
    total_collections = len(collective_outputs)
    completed_collections = sum(1 for co in collective_outputs if co.success)
    failed_collections = total_collections - completed_collections
    
    total_stages = 0
    completed_stages = 0
    failed_stages = 0
    total_requests = 0
    completed_requests = 0
    failed_requests = 0
    
    total_input_tokens = 0
    total_output_tokens = 0
    total_service_gain = 0
    
    collection_latencies = []
    stage_latencies = []
    request_latencies = []
    ttft_values = []
    slo_met_collections = 0
    slo_met_requests = 0  # Track individual requests meeting SLO
    
    for collective_output in collective_outputs:
        if collective_output.success:
            collection_latencies.append(collective_output.total_latency)
            
            # Check SLO for collection
            if slo_constraint and collective_output.total_latency <= slo_constraint[2] * len(collective_output.stage_outputs):
                slo_met_collections += 1
        
        for stage_output in collective_output.stage_outputs:
            total_stages += 1
            if stage_output.success:
                completed_stages += 1
                stage_latencies.append(stage_output.stage_latency)
            else:
                failed_stages += 1
            
            for request_output in stage_output.stage_outputs:
                total_requests += 1
                if request_output.success:
                    completed_requests += 1
                    request_latencies.append(request_output.request_latency)
                    
                    # Check if individual request meets SLO
                    if slo_constraint and request_output.request_latency <= slo_constraint[2]:
                        slo_met_requests += 1
                    
                    # Calculate tokens
                    input_tokens = len(tokenizer(request_output.request_input, add_special_tokens=False).input_ids)
                    output_tokens = len(tokenizer(request_output.request_output, add_special_tokens=False).input_ids)
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens
                    
                    # Collect metrics
                    if request_output.request_ttft > 0:
                        ttft_values.append(request_output.request_ttft)
                    total_service_gain += request_output.request_service_gain
                else:
                    failed_requests += 1
    
    # Calculate statistics
    def compute_percentiles(data: List[float], percentiles: List[float]) -> List[Tuple[float, float]]:
        if not data:
            return [(p, 0.0) for p in percentiles]
        return [(p, np.percentile(data, p)) for p in percentiles]
    
    metrics = DeepResearchMetrics(
        duration=dur_s,
        total_collections=total_collections,
        completed_collections=completed_collections,
        failed_collections=failed_collections,
        total_stages=total_stages,
        completed_stages=completed_stages,
        failed_stages=failed_stages,
        total_requests=total_requests,
        completed_requests=completed_requests,
        failed_requests=failed_requests,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        input_throughput=total_input_tokens / dur_s if dur_s > 0 else 0,
        output_throughput=total_output_tokens / dur_s if dur_s > 0 else 0,
        total_throughput=(total_input_tokens + total_output_tokens) / dur_s if dur_s > 0 else 0,
        mean_collection_latency=np.mean(collection_latencies) if collection_latencies else 0,
        median_collection_latency=np.median(collection_latencies) if collection_latencies else 0,
        percentiles_collection_latency=compute_percentiles(collection_latencies, selected_percentiles),
        mean_stage_latency=np.mean(stage_latencies) if stage_latencies else 0,
        median_stage_latency=np.median(stage_latencies) if stage_latencies else 0,
        percentiles_stage_latency=compute_percentiles(stage_latencies, selected_percentiles),
        mean_request_latency=np.mean(request_latencies) if request_latencies else 0,
        median_request_latency=np.median(request_latencies) if request_latencies else 0,
        percentiles_request_latency=compute_percentiles(request_latencies, selected_percentiles),
        mean_ttft_ms=np.mean(ttft_values) * 1000 if ttft_values else 0,
        median_ttft_ms=np.median(ttft_values) * 1000 if ttft_values else 0,
        percentiles_ttft_ms=[(p, np.percentile(ttft_values, p) * 1000) for p in selected_percentiles] if ttft_values else [(p, 0) for p in selected_percentiles],
        slo_met_collections=slo_met_collections,
        slo_met_ratio=slo_met_collections / completed_collections if completed_collections > 0 else 0,
        slo_met_requests=slo_met_requests,
        slo_goodput=slo_met_requests / dur_s if dur_s > 0 else 0,
        total_service_gain=total_service_gain,
    )
    
    return metrics


def print_deepresearch_results(metrics: DeepResearchMetrics) -> None:
    """Print benchmark results for DeepResearch collective requests."""
    separator = "{s:{c}^{n}}".format(s='', n=80, c='-')
    header_separator = "{s:{c}^{n}}".format(s='', n=80, c='=')
    
    print(header_separator)
    print("{s:{c}^{n}}".format(s=' DeepResearch Benchmark Results ', n=80, c='='))
    print(header_separator)
    
    # Basic statistics
    print("\n{:<40} {:<20}".format("Metric", "Value"))
    print(separator)
    print("{:<40} {:<20.2f}".format("Duration (seconds)", metrics.duration))
    print("{:<40} {:<20}".format("Total Collections", metrics.total_collections))
    print("{:<40} {:<20}".format("Completed Collections", metrics.completed_collections))
    print("{:<40} {:<20}".format("Failed Collections", metrics.failed_collections))
    print("{:<40} {:<20}".format("Total Stages", metrics.total_stages))
    print("{:<40} {:<20}".format("Completed Stages", metrics.completed_stages))
    print("{:<40} {:<20}".format("Failed Stages", metrics.failed_stages))
    print("{:<40} {:<20}".format("Total Requests", metrics.total_requests))
    print("{:<40} {:<20}".format("Completed Requests", metrics.completed_requests))
    print("{:<40} {:<20}".format("Failed Requests", metrics.failed_requests))
    print(separator)
    
    # Token throughput
    print("\nToken Throughput:")
    print(separator)
    print("{:<40} {:<20}".format("Total Input Tokens", metrics.total_input_tokens))
    print("{:<40} {:<20}".format("Total Output Tokens", metrics.total_output_tokens))
    print("{:<40} {:<20.2f}".format("Input Throughput (tokens/s)", metrics.input_throughput))
    print("{:<40} {:<20.2f}".format("Output Throughput (tokens/s)", metrics.output_throughput))
    print("{:<40} {:<20.2f}".format("Total Throughput (tokens/s)", metrics.total_throughput))
    print(separator)
    
    # Latency statistics
    print("\nLatency Statistics:")
    print(separator)
    
    # Collection latencies
    print("\nCollection Latencies (seconds):")
    print("{:<20} {:<15} {:<15}".format("Metric", "Mean", "Median"))
    print("{:<20} {:<15.3f} {:<15.3f}".format(
        "Collection", metrics.mean_collection_latency, metrics.median_collection_latency))
    print("\nCollection Latency Percentiles:")
    for p, val in metrics.percentiles_collection_latency:
        print("  P{:<3}: {:.3f}s".format(int(p), val))
    
    # Stage latencies
    print("\nStage Latencies (seconds):")
    print("{:<20} {:<15} {:<15}".format("Metric", "Mean", "Median"))
    print("{:<20} {:<15.3f} {:<15.3f}".format(
        "Stage", metrics.mean_stage_latency, metrics.median_stage_latency))
    print("\nStage Latency Percentiles:")
    for p, val in metrics.percentiles_stage_latency:
        print("  P{:<3}: {:.3f}s".format(int(p), val))
    
    # Request latencies
    print("\nRequest Latencies (seconds):")
    print("{:<20} {:<15} {:<15}".format("Metric", "Mean", "Median"))
    print("{:<20} {:<15.3f} {:<15.3f}".format(
        "Request", metrics.mean_request_latency, metrics.median_request_latency))
    print("\nRequest Latency Percentiles:")
    for p, val in metrics.percentiles_request_latency:
        print("  P{:<3}: {:.3f}s".format(int(p), val))
    
    # TTFT statistics
    print("\nTime to First Token (TTFT):")
    print("{:<20} {:<15} {:<15}".format("Metric", "Mean (ms)", "Median (ms)"))
    print("{:<20} {:<15.2f} {:<15.2f}".format(
        "TTFT", metrics.mean_ttft_ms, metrics.median_ttft_ms))
    print("\nTTFT Percentiles:")
    for p, val in metrics.percentiles_ttft_ms:
        print("  P{:<3}: {:.2f}ms".format(int(p), val))
    
    print(separator)
    
    # SLO and service gain
    print("\nSLO and Service Metrics:")
    print(separator)
    print("{:<40} {:<20}".format("Collections Meeting SLO", metrics.slo_met_collections))
    print("{:<40} {:<20.2%}".format("Collection SLO Met Ratio", metrics.slo_met_ratio))
    print("{:<40} {:<20}".format("Requests Meeting SLO", metrics.slo_met_requests))
    print("{:<40} {:<20.2f} req/s".format("SLO Goodput", metrics.slo_goodput))
    print("{:<40} {:<20.2f}".format("Total Service Gain", metrics.total_service_gain))
    
    print(header_separator)


async def benchmark_deepresearch(
    api_url: str,
    base_url: str,
    model: str,
    tokenizer: PreTrainedTokenizerBase,
    collective_requests: List[DeepResearchCollectiveRequest],
    logprobs: Optional[int],
    num_prompts: int,
    n: int,
    best_of: int,
    request_rate: float,
    disable_tqdm: bool,
    poisson_lambda: Optional[int],
    burst: Optional[bool],
    selected_percentiles: List[float],
    ignore_eos: bool,
    max_concurrency: Optional[int],
    max_output_len: int,
    slo_constraint: Tuple[float, float, float],
    penalty_factor: int,
):
    """Run DeepResearch benchmark."""
    
    # Limit to specified number of prompts
    if num_prompts < len(collective_requests):
        collective_requests = collective_requests[:num_prompts]
    
    print(f"Running benchmark with {len(collective_requests)} collective requests")
    
    # Test with first request
    print("Starting initial test run...")
    test_request = collective_requests[0]
    
    sampling_params = SamplingParams(
        n=n,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        max_tokens=max_output_len,
        logprobs=logprobs,
        best_of=best_of,
        ignore_eos=ignore_eos,
    )
    
    # Run test
    from client_deepresearch import send_deepresearch_collective_request
    
    test_output = await send_deepresearch_collective_request(
        collective_request=test_request,
        slo_constraint=slo_constraint,
        sampling_params=sampling_params,
        client_id=0,
        api_url=api_url,
        model_name=model,
        penalty_factor=penalty_factor,
        client_deadline=200,
    )
    
    if not test_output.success:
        raise ValueError(f"Initial test run failed. Please check your configuration.")
    else:
        print("Initial test run completed successfully. Starting main benchmark...")
    
    # Setup progress bar
    pbar = None if disable_tqdm else async_tqdm(total=len(collective_requests))
    
    # Run benchmark
    benchmark_start_time = time.perf_counter()
    
    collective_outputs = await deepresearch_client_simulator(
        collective_requests=collective_requests,
        slo_constraint=slo_constraint,
        penalty_factor=penalty_factor,
        poisson_lambda=poisson_lambda,
        burst=burst,
        sampling_params=sampling_params,
        client_id=0,
        client_deadline=200,
        api_url=api_url,
        model_name=model,
        pbar=pbar,
    )
    
    benchmark_duration = time.perf_counter() - benchmark_start_time
    
    if pbar:
        pbar.close()
    
    # Calculate and print metrics
    metrics = calculate_deepresearch_metrics(
        collective_outputs=collective_outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentiles=selected_percentiles,
        slo_constraint=slo_constraint,
    )
    
    print_deepresearch_results(metrics)
    
    return metrics


def main(args: argparse.Namespace):
    """Main function for DeepResearch benchmark."""
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    
    # Load DeepResearch trace
    collective_requests = DeepResearchTrace.load_trace(args.trace_path)
    print(f"Loaded {len(collective_requests)} collective requests from trace")
    
    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base_url = f"http://{args.host}:{args.port}"
    
    tokenizer = get_tokenizer(tokenizer_id, trust_remote_code=args.trust_remote_code)
    
    # Parse SLO constraint
    slo_constraint = tuple(map(float, args.slo_constraint.split(",")))
    
    # Calculate poisson lambda
    if args.arrival_rate:
        poisson_lambda = int(1 / args.arrival_rate * 1000)
    else:
        poisson_lambda = 1000  # Default 1 request per second
    
    exp_setting = {
        "date": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "model_id": model_id,
        "tokenizer_id": tokenizer_id,
        "best_of": args.best_of,
        "num_prompts": args.num_prompts,
        "arrival_rate": args.arrival_rate,
        "poisson_lambda": poisson_lambda,
        "burst": args.burst,
        "policy": args.policy,
        "slo_constraint": args.slo_constraint,
        "penalty_factor": args.penalty_factor,
        "max_output_len": args.max_output_len,
        "trace_path": args.trace_path,
    }
    
    print("Experiment Setting: ", exp_setting)
    
    asyncio.run(
        benchmark_deepresearch(
            api_url=api_url,
            base_url=base_url,
            model=model_id,
            tokenizer=tokenizer,
            collective_requests=collective_requests,
            logprobs=args.logprobs,
            num_prompts=args.num_prompts,
            n=args.n,
            best_of=args.best_of,
            request_rate=args.arrival_rate,
            disable_tqdm=args.disable_tqdm,
            poisson_lambda=poisson_lambda,
            burst=args.burst,
            selected_percentiles=[
                float(p) for p in args.metric_percentiles.split(",")
            ],
            ignore_eos=args.ignore_eos,
            max_concurrency=args.max_concurrency,
            max_output_len=args.max_output_len,
            slo_constraint=slo_constraint,
            penalty_factor=args.penalty_factor,
        ))


if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description="Benchmark DeepResearch collective requests.")
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/generate",
        help="API endpoint.",
    )
    parser.add_argument(
        "--trace-path",
        type=str,
        default="benchmarks/dataset/deepresearch_trace_filtered_8192.jsonl",
        help="Path to the DeepResearch trace file.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=10,
        help="Number of collective requests to process from the trace.",
    )
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=1.0,
        help="Arrival rate (req/s) for the poisson distribution.",
    )
    parser.add_argument(
        "--burst",
        type=bool,
        default=False,
        help="Specify to use burst request arrival pattern.",
    )
    parser.add_argument(
        "--slo-constraint",
        type=str,
        default="1000,1000,5000",
        help="SLO constraint for the benchmark (ttft, tbt, ttlt in ms).",
    )
    parser.add_argument(
        "--penalty-factor",
        type=int,
        default=1,
        help="Penalty factor for SLO violations.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        help="Policy of the scheduler.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name or path of the tokenizer, if not using the default tokenizer.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of prompts to generate from 'best-of' prompts.",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and returns the best one.",
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help="Number of logprobs-per-token to compute & return.",
    )
    parser.add_argument(
        "--max-output-len",
        type=int,
        default=1024,
        help="Maximum length of the output text.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        default=False,
        help="Set ignore_eos flag when sending the benchmark request.",
    )
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="50,90,95,99",
        help="Comma-separated list of percentiles for selected metrics.",
    )
    
    args = parser.parse_args()
    main(args)