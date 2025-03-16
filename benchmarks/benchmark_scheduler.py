"""Benchmark the latency of processing a single batch of requests."""
import argparse
import asyncio
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
from tqdm import tqdm
from datasets import load_dataset
from pathlib import Path
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer
try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser

sys.path.append(str(Path(__file__).resolve().parents[1]))
from benchmarks.trace import (client_simulator, send_request, send_collective_request, RequestInput, 
                              RequestOutput, Trace, RequestFormat, BaseDataset, RequestType)

@dataclass
class BenchmarkMetrics:
    duration: float
    request_num: List[int]
    request_completed: List[int]
    request_slo_meet: List[int]
    request_service_gain: List[int]
    request_total_input: int
    request_total_output: int
    request_throughput: List[float]
    request_goodput: List[float]
    time_window_service_gain: Dict[int, List[float]]
    input_token_throughput: float
    output_token_throughput: float
    total_token_throughput: float
    mean_ttft_ms: List[float]
    median_ttft_ms: List[float]
    percentiles_ttft_ms: List[List[Tuple[float, float]]]
    mean_tbt_ms: List[float]
    median_tbt_ms: List[float]
    percentiles_tbt_ms: List[List[Tuple[float, float]]]
    mean_e2el_ms: List[float]
    median_e2el_ms: List[float]
    percentiles_e2el_ms: List[List[Tuple[float, float]]]

def calculate_metrics(
    outputs: List[List[RequestOutput]],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentiles: List[float],
    slo_constraint: Optional[Tuple[float, float, float]] = None,    # (ttft, tbt, ttlt)
    time_window_seconds: int = 60,
) -> BenchmarkMetrics:

    request_num = [0, 0, 0, 0]
    request_completed = [0, 0, 0, 0]
    request_slo_meet = [0, 0, 0, 0]
    request_service_gain = [0, 0, 0, 0]
    request_total_input = 0
    request_total_output = 0
    ttft_data = [[], [], [], []]  # [Latency, Throughput, Collective, Total]
    tbt_data = [[], [], [], []]
    e2el_data = [[], [], [], []]
    
    slo_ttft_meet = [0, 0, 0, 0]
    slo_tbt_meet = [0, 0, 0, 0]         
    slo_ttlt_meet = [0, 0, 0, 0]
    time_window_service_gain = {}
    
    for i in range(len(outputs)):
        for req_idx in range(len(outputs[i])):
            task_type = outputs[i][req_idx].request_type       
            request_num[task_type] += 1
            request_num[3] += 1
        
            if outputs[i][req_idx].success:        
                request_completed[task_type] += 1
                request_completed[3] += 1

                ttft = outputs[i][req_idx].request_ttft
                tbt_list = outputs[i][req_idx].request_tbt
                ttlt = outputs[i][req_idx].request_latency
                
                meets_ttft = (ttft <= slo_constraint[0]) if slo_constraint else True
                meets_tbt = all(tbt <= slo_constraint[1] for tbt in tbt_list) if slo_constraint else True
                meets_ttlt = (ttlt <= slo_constraint[2]) if slo_constraint else True
                
                if slo_constraint:
                    if meets_ttft: slo_ttft_meet[task_type] += 1
                    if meets_tbt: slo_tbt_meet[task_type] += 1
                    if meets_ttlt: slo_ttlt_meet[task_type] += 1
                    
                    if (meets_ttft and meets_tbt and task_type == 0) or \
                        (meets_ttlt and (task_type == 1 or task_type == 2)):
                        request_slo_meet[task_type] += 1
                        request_slo_meet[3] += 1
                        
                request_total_input += len(tokenizer(outputs[i][req_idx].request_input, add_special_tokens=False).input_ids)
                request_total_output += len(tokenizer(outputs[i][req_idx].request_output, add_special_tokens=False).input_ids)
                
                # TODO: finish time or start time
                finish_time = outputs[i][req_idx].request_finish_time
                service_gain = outputs[i][req_idx].request_service_gain
                
                # 计算所属时间窗口
                window_end: int = (finish_time // time_window_seconds) * time_window_seconds
                if window_end not in time_window_service_gain:
                    time_window_service_gain[window_end] = [0.0, 0.0, 0.0, 0.0]
                time_window_service_gain[window_end][task_type] += service_gain
                time_window_service_gain[window_end][3] += service_gain
                request_service_gain[task_type] += service_gain
                request_service_gain[3] += service_gain

                if ttft > 0:
                    ttft_data[task_type].append(ttft)
                    ttft_data[3].append(ttft)  # Total

                tbt_data[task_type].extend(tbt_list)
                tbt_data[3].extend(tbt_list)  # Total

                e2el_data[task_type].append(ttlt)
                e2el_data[3].append(ttlt)  # Total

    if request_completed[3] == 0:
        warnings.warn(
            "All tasks failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2)
        
    def compute_stats(data: List[List[float]], percentiles: List[float]) -> Tuple[List[float], List[float], List[List[Tuple]]]:
        means = []
        medians = []
        percentiles_list = []
        for i in range(4):
            values = data[i] or [0]
            means.append(np.mean(values) * 1000)
            medians.append(np.median(values) * 1000)
            percentiles_list.append([
                (p, np.percentile(values, p) * 1000)
                for p in percentiles
            ])
        return means, medians, percentiles_list
    
    ttft_means, ttft_medians, ttft_percentiles = compute_stats(ttft_data, selected_percentiles)
    tbt_means, tbt_medians, tbt_percentiles = compute_stats(tbt_data, selected_percentiles)
    e2el_means, e2el_medians, e2el_percentiles = compute_stats(e2el_data, selected_percentiles)
    
    metrics = BenchmarkMetrics(
        duration=dur_s,
        request_num=request_num,
        request_completed=request_completed,
        request_slo_meet=request_slo_meet,
        request_service_gain=request_service_gain,
        request_total_input=request_total_input,
        request_total_output=request_total_output,
        time_window_service_gain=time_window_service_gain,
        request_throughput=[completed / dur_s for completed in request_completed],
        request_goodput=[completed / dur_s for completed in request_slo_meet],
        input_token_throughput=request_total_input / dur_s,
        output_token_throughput=request_total_output / dur_s,
        total_token_throughput=(request_total_input + request_total_output) / dur_s,
        mean_ttft_ms=ttft_means,
        median_ttft_ms=ttft_medians,
        percentiles_ttft_ms=ttft_percentiles,
        mean_tbt_ms=tbt_means,
        median_tbt_ms=tbt_medians,
        percentiles_tbt_ms=tbt_percentiles,
        mean_e2el_ms=e2el_means,
        median_e2el_ms=e2el_medians,
        percentiles_e2el_ms=e2el_percentiles,
    )

    return metrics

def print_benchmark_results(metrics: BenchmarkMetrics, time_window_seconds: float) -> None:
    request_types = ["Latency", "Throughput", "Collective", "Total"]
    separator = "{s:{c}^{n}}".format(s='', n=80, c='-')
    header_separator = "{s:{c}^{n}}".format(s='', n=80, c='=')

    # print header
    print(header_separator)
    print("{s:{c}^{n}}".format(s=' Serving Benchmark Results ', n=80, c='='))
    print(header_separator)
    print("{:<40} {:<20} {:<20}".format("Metric", "Value", "Details"))
    print(separator)
    
    # 基础统计
    print("{:<40} {:<20.2f}".format("Duration (seconds)", metrics.duration))
    print("{:<40} {:<20}".format("Total Input Tokens", metrics.request_total_input))
    print("{:<40} {:<20}".format("Total Output Tokens", metrics.request_total_output))
    print("{:<40} {:<20.2f}".format("Output Token Throughput (tokens/s)", metrics.output_token_throughput))
    print("{:<40} {:<20.2f}".format("Total Token Throughput (tokens/s)", metrics.total_token_throughput))
    print(separator)

    # 请求级统计表格
    print("{:<15} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "Type", "Total", "Completed", "SLO Met", "Throughput", "Goodput"))
    print(separator)
    for i in range(4):
        print("{:<15} {:<10} {:<10} {:<10} {:<10.2f} {:<10.2f}".format(
            request_types[i],
            metrics.request_num[i],
            metrics.request_completed[i],
            metrics.request_slo_meet[i],
            metrics.request_throughput[i],
            metrics.request_goodput[i]
        ))
    print(separator)

    def print_latency_stats(metrics: BenchmarkMetrics, metric_name: str):
        names = ["Latency", "Throughput", "Collective", "Total"]
        metric_map = {
            "ttft": ("TTFT", metrics.mean_ttft_ms, metrics.median_ttft_ms, metrics.percentiles_ttft_ms),
            "tbt": ("TBT", metrics.mean_tbt_ms, metrics.median_tbt_ms, metrics.percentiles_tbt_ms),
            "e2el": ("E2EL", metrics.mean_e2el_ms, metrics.median_e2el_ms, metrics.percentiles_e2el_ms),
        }
        title, means, medians, percentiles = metric_map[metric_name]

        print(f"\n{title} Statistics:")
        print("{:<10} {:<10} {:<15} {}".format(
            "Type", "Mean(ms)", "Median(ms)", "Percentiles(ms)"))
        for i in range(4):
            p_str = ", ".join([f"P{int(p)}:{v:.1f}" for p, v in percentiles[i]])
            print("{:<10} {:<10.2f} {:<15.2f} {}".format(
                names[i], means[i], medians[i], p_str))

    print_latency_stats(metrics, "ttft")
    print_latency_stats(metrics, "tbt")
    print_latency_stats(metrics, "e2el")
    print(separator)

    def print_time_window_gain(metrics: BenchmarkMetrics):
        print("\n{:=^80}".format(" Time Window Service Gain "))
        print("{:<10} {:<15} {:<15} {:<15} {:<15}".format(
            "Window(s)", "Latency", "Throughput", "Collective", "Total"))
        print("-" * 80)

        sorted_windows = sorted(metrics.time_window_service_gain.items(), key=lambda x: x[0])

        for window_end, gains in sorted_windows:
            print("{:<10} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f}".format(
                window_end,
                gains[0],  # Latency
                gains[1],  # Throughput
                gains[2],  # Collective
                gains[3],  # Total
            ))
        
        # print total service gain of all time windows
        total_gains = [0.0, 0.0, 0.0, 0.0]
        for _, gains in sorted_windows:
            for i in range(4):
                total_gains[i] += gains[i]
        print("-" * 80)
        print("{:<10} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f}".format(
            "Total",
            total_gains[0],  # Latency
            total_gains[1],  # Throughput
            total_gains[2],  # Collective
            total_gains[3],  # Total
        ))
        print("=" * 80)
    print_time_window_gain(metrics)
    print(header_separator)


async def benchmark(
    api_url: str,
    base_url: str,
    tokenizer: PreTrainedTokenizerBase,
    trace: List[RequestFormat],
    logprobs: Optional[int],
    num_prompts: int,
    n: int,
    best_of: int,
    request_rate: List[float],
    disable_tqdm: bool,
    poisson_lambda: Optional[int],
    burst: Optional[bool],
    selected_percentiles: List[str],
    ignore_eos: bool,
    max_concurrency: Optional[int],
    max_output_len: int,
    slo_constraint: Tuple[float, float, float], # (ttft, tbt, ttlt)
    tot_structure: Tuple[int, int],             # (tot_thoughts, tot_rounds)
):
    trace_len = len(trace)
    requests = (trace * (num_prompts // trace_len + 1))[:num_prompts]
    
    for id, request in enumerate(requests):
        request.collection_id = id

    # Get the first request to validate the correctness
    print("Starting initial single prompt test run...")
    test_request: RequestFormat = requests[0]
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
    
    test_input = RequestInput(
        request=test_request,
        sampling_params=sampling_params,
        client_id=0,
        api_url=api_url,
    )
    
    if test_input.request.request_type == RequestType.Collective:
        test_output: List[RequestOutput] = await send_collective_request(request_info=test_input, 
                                                                         tot_structure=tot_structure)
    else:
        test_output: List[RequestOutput] = await send_request(request_info=test_input)
    if not test_output[0].success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output[0].error}")
    else:
        print("Initial test run completed. Starting main benchmark run...")

    print(f"Traffic request rate: {request_rate}")
    print(f"Maximum request concurrency: {max_concurrency}")
    
    for request in requests:
        if request.request_type == RequestType.Latency:
            request.deadline = slo_constraint[0] + slo_constraint[1] * request.output_len
        elif request.request_type == RequestType.Throughput:
            request.deadline = slo_constraint[2]
        elif request.request_type == RequestType.Collective:
            request.deadline = slo_constraint[2] * tot_structure[1] * 2
        request.deadline *= 1000

    pbar = None if disable_tqdm else tqdm(total=len(requests))

    benchmark_start_time = time.perf_counter()
    # outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)
    if request_rate is None:
        request_per_user = [requests]
    else:
        request_per_user = BaseDataset.divide_by_rate(requests, request_rate)
        
    
    client_tasks = []
    for client_id, input_requests in enumerate(request_per_user):
        client_tasks.append(
            asyncio.create_task(
                client_simulator(
                    input_requests=input_requests,
                    poisson_lambda=poisson_lambda,
                    burst=burst,
                    sampling_params=sampling_params,
                    client_id=client_id,
                    client_deadline=6000,
                    api_url=api_url,
                    tot_structure=tot_structure,
                    pbar=pbar,
                )
            )
    )

    results: List[List[RequestOutput]] = await asyncio.gather(*client_tasks)
    outputs: List[RequestOutput] = [output for result in results 
                            for output in result]

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics = calculate_metrics(
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentiles=selected_percentiles,
        slo_constraint=slo_constraint,
        time_window_seconds=60,
    )
    
    print_benchmark_results(metrics, time_window_seconds=60)


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    trace: List[RequestFormat] = Trace.load_trace(args.trace_path)

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base_url = f"http://{args.host}:{args.port}"

    tokenizer = get_tokenizer(tokenizer_id,
                              trust_remote_code=args.trust_remote_code)
        
    # gootput_config_dict = check_goodput_args(args)
    request_rate = [float(rate) for rate in args.user_request_rate.split(",")]
    if sum(request_rate) != 1.0:
        Warning("The sum of request rates is not equal to 1.0. ")
        request_rate = [rate / sum(request_rate) for rate in request_rate]
        
    args.poisson_lambda = int(1 / args.arrival_rate * 1000)
    
    exp_setting = {
        "date": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "model_id": model_id,
        "tokenizer_id": tokenizer_id,
        "best_of": args.best_of,
        "num_prompts": args.num_prompts,
        "arrival_rate": args.arrival_rate,
        "user_request_rate": args.user_request_rate,
        "poisson_lambda": args.poisson_lambda,
        "burst": args.burst,
        "batch_size": args.batch_size,
        "policy": args.policy,
        "dataset": args.dataset,
        "slo_constraint": args.slo_constraint,
        "max_output_len": args.max_output_len,
        "tot_thoughts": args.tot_thoughts,
        "tot_rounds": args.tot_rounds,
        "chunked": args.chunked,
    }
    
    print("Experiment Setting: ", exp_setting)
    
    asyncio.run(
        benchmark(
            api_url=api_url,
            base_url=base_url,
            tokenizer=tokenizer,
            trace=trace,
            logprobs=args.logprobs,
            num_prompts=args.num_prompts,
            n=args.n,
            best_of=args.best_of,
            request_rate=request_rate,
            disable_tqdm=args.disable_tqdm,
            poisson_lambda=args.poisson_lambda,
            burst=args.burst,
            selected_percentiles=[
                float(p) for p in args.metric_percentiles.split(",")
            ],
            ignore_eos=args.ignore_eos,
            max_concurrency=args.max_concurrency,
            max_output_len=args.max_output_len,
            slo_constraint=tuple(map(float, args.slo_constraint.split(","))),
            tot_structure=(args.tot_thoughts, args.tot_rounds),
        ))


if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description="Benchmark the online serving throughput.")
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
        default="benchmarks/dataset/trace/alpaca.json",
        help="Path to the trace file.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts in the benchmark.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for each request.",
    )
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=2.0,
        help="Arrival rate (req/s) for the poisson distribution.",
    )
    parser.add_argument(
        "--burst",
        type=bool,
        default=False,
        help="Specify to use burst request arrvial pattern.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="lmsys",
        help="Dataset to use for benchmarking.",
    )
    parser.add_argument(
        "--slo-constraint",
        type=str,
        default="1000,1000,2000",
        help="SLO constraint for the benchmark. "
    )
    parser.add_argument(
        "--user-request-rate",
        type=str,
        default="1.0",
        help="Comma-separated list of request rates for different users. ",
    )
    parser.add_argument(
        "--tot-thoughts",
        type=int,
        default=3,
        help="Number of thoughts for the TOT structure.",
    )
    parser.add_argument(
        "--tot-rounds",
        type=int,
        default=2,
        help="Number of rounds for the TOT structure.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up."
    )
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        help="Policy of the scheduler.",
    )
    parser.add_argument(
        "--poisson",
        type=int,
        help="Specify to use poisson distribution for request rate.",
    )
    parser.add_argument(
        "--chunked",
        action="store_true",
        help="Specify to use chunked prefill for server.",
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
        help=
        "Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
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
        help="Generates `best_of` sequences per prompt and "
        "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help=("Number of logprobs-per-token to compute & return as part of "
              "the request. If unspecified, then either (1) if beam search "
              "is disabled, no logprobs are computed & a single dummy "
              "logprob is returned for each token; or (2) if beam search "
              "is enabled 1 logprob per token is computed"),
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
        "--profile",
        action="store_true",
        help="Use Torch Profiler. The endpoint must be launched with "
        "VLLM_TORCH_PROFILER_DIR to enable profiler.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        default=True,
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default='./result',
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default='None',
        help="Specify the filename to save benchmark json results."
        "If not specified, results will be saved in "
        "{backend}-{base_model_id}-{current_dt}.json"
        " format.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        default=False,
        help="Set ignore_eos flag when sending the benchmark request."
        "Warning: ignore_eos is not supported in deepspeed_mii and tgi.")
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="25,50,75,99",
        help="Comma-seperated list of percentiles for selected metrics. "
        "To report 25-th, 50-th, and 75-th percentiles, use \"25,50,75\". "
        "Default value is \"99\". "
        "Use \"--percentile-metrics\" to select metrics.",
    )

    args = parser.parse_args()
    main(args)