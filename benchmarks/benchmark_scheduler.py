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

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer
try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser

sys.path.append(str(Path(__file__).resolve().parents[1]))
from benchmarks.trace import (client_simulator, send_request, RequestInput, 
                              RequestOutput, Trace, RequestFormat, BaseDataset)

prefill_weight = 1
decode_weight = 2

@dataclass
class BenchmarkMetrics:
    task_num: List[int]
    request_num: List[int]
    task_completed: List[int]
    request_completed: List[int]
    task_deadline_meet: List[int]
    request_deadline_meet: List[int]
    task_service_gain: List[int]
    request_service_gain: List[int]
    task_total_input: int
    task_total_output: int
    request_total_input: int
    request_total_output: int
    task_throughput: List[float]
    request_throughput: List[float]
    task_goodput: List[float]
    request_goodput: List[float]
    output_token_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    std_ttft_ms: float
    median_ttft_ms: float
    percentiles_ttft_ms: List[float]
    mean_tbt_ms: float
    std_tbt_ms: float
    median_tbt_ms: float
    percentiles_tbt_ms: List[float]
    mean_request_e2el_ms: float
    std_request_e2el_ms: float
    median_request_e2el_ms: float
    percentiles_request_e2el_ms: List[float]
    mean_task_e2el_ms: float
    std_task_e2el_ms: float
    median_task_e2el_ms: float
    percentiles_task_e2el_ms: List[float]


def calculate_metrics(
    input_requests: List[RequestFormat],
    outputs: List[RequestOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentile_metrics: List[str],
    selected_percentiles: List[float],
    gootput_config_dict: Dict[str, float],
) -> BenchmarkMetrics:

    task_num = [0, 0, 0, 0]
    request_num = [0, 0, 0, 0]
    task_completed = [0, 0, 0, 0]
    request_completed = [0, 0, 0, 0]
    task_deadline_meet = [0, 0, 0, 0]
    request_deadline_meet = [0, 0, 0, 0]
    task_service_gain = [0, 0, 0, 0]
    request_service_gain = [0, 0, 0, 0]
    
    task_total_input = 0
    task_total_output = 0
    request_total_input = 0
    request_total_output = 0
    request_tbt: List[float] = []
    
    task_ttft: List[float] = []
    request_ttft: List[float] = []
    task_e2el: List[float] = []
    request_e2el: List[float] = []
    
    for i in range(len(outputs)):
        task_type = outputs[i].type  
        task_num[task_type] += 1
        task_num[3] += 1
        num_request = outputs[i].collection_num_requests
        request_num[task_type] += num_request
        request_num[3] += num_request
        if outputs[i].success:
            task_completed[task_type] += 1
            task_completed[3] += 1
            
            request_completed[task_type] += outputs[i].collection_num_requests
            request_completed[3] += outputs[i].collection_num_requests
            
            input_len = 0
            output_len = 0
            input_len_list = []
            output_len_list = []
            for req_in, req_out in zip(outputs[i].request_input, outputs[i].request_output):
                req_output_len = len(tokenizer(req_out, add_special_tokens=False).input_ids)
                req_input_len = len(tokenizer(req_in, add_special_tokens=False).input_ids)
                input_len_list.append(req_input_len)
                output_len_list.append(req_output_len)
                input_len += req_input_len
                output_len += req_output_len
                
            request_total_output += output_len
            request_total_input += input_len
            
            task_input_length = input_requests[i].prompt_len
            task_output_length = len(tokenizer(outputs[i].task_output, add_special_tokens=False).input_ids)
            task_total_input += task_input_length
            task_total_output += task_output_length
            
            if outputs[i].finish_before_ddl:
                task_deadline_meet[task_type] += 1
                task_deadline_meet[3] += 1
                
                request_deadline_meet[task_type] += outputs[i].collection_num_requests
                request_deadline_meet[3] += outputs[i].collection_num_requests
                
                cur_task_service_gain = task_input_length * prefill_weight + task_output_length * decode_weight
                task_service_gain[task_type] += cur_task_service_gain
                task_service_gain[3] += cur_task_service_gain
                
                for req_input_len, req_output_len in zip(input_len_list, output_len_list):
                    req_service_gain = req_input_len * prefill_weight + req_output_len * decode_weight
                    request_service_gain[task_type] += req_service_gain
                    request_service_gain[3] += req_service_gain
            
            task_ttft.append(outputs[i].task_ttft)
            task_e2el.append(outputs[i].task_latency)
            request_ttft.extend(outputs[i].request_ttft)
            request_e2el.extend(outputs[i].request_latency)
            
            if output_len > 1:
                for req_latency, req_ttft, req_output_len in \
                    zip(outputs[i].request_latency, outputs[i].request_ttft, output_len_list):
                    tbt = (req_latency - req_ttft) / (req_output_len - 1)
                    request_tbt.append(tbt)
            # Note: if output_len <= 1, we regard tbt as 0 for goodput

    if task_completed[3] == 0:
        warnings.warn(
            "All tasks failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2)
    metrics = BenchmarkMetrics(
        task_num=task_num,
        request_num=request_num,
        task_completed=task_completed,
        request_completed=request_completed,
        task_deadline_meet=task_deadline_meet,
        request_deadline_meet=request_deadline_meet,
        task_service_gain=task_service_gain,
        request_service_gain=request_service_gain,
        task_total_input=task_total_input,
        task_total_output=task_total_output,
        request_total_input=request_total_input,
        request_total_output=request_total_output,
        task_throughput=[completed / dur_s for completed in task_completed],
        request_throughput=[completed / dur_s for completed in request_completed],
        task_goodput=[completed / dur_s for completed in task_deadline_meet],
        request_goodput=[completed / dur_s for completed in request_deadline_meet],
        output_token_throughput=request_total_output / dur_s,
        total_token_throughput=(request_total_input + request_total_output) / dur_s,
        mean_ttft_ms=np.mean(request_ttft or 0) * 1000,
        std_ttft_ms=np.std(request_ttft or 0) * 1000,
        median_ttft_ms=np.median(request_ttft or 0) * 1000,
        percentiles_ttft_ms=[(p, np.percentile(request_ttft or 0, p) * 1000) for p in selected_percentiles],
        mean_tbt_ms=np.mean(request_tbt or 0) * 1000,
        std_tbt_ms=np.std(request_tbt or 0) * 1000,
        median_tbt_ms=np.median(request_tbt or 0) * 1000,
        percentiles_tbt_ms=[(p, np.percentile(request_tbt or 0, p) * 1000) for p in selected_percentiles],
        mean_request_e2el_ms=np.mean(request_e2el or 0) * 1000,
        std_request_e2el_ms=np.std(request_e2el or 0) * 1000,
        median_request_e2el_ms=np.median(request_e2el or 0) * 1000,
        percentiles_request_e2el_ms=[(p, np.percentile(request_e2el or 0, p) * 1000) for p in selected_percentiles],
        mean_task_e2el_ms=np.mean(task_e2el or 0) * 1000,
        std_task_e2el_ms=np.std(task_e2el or 0) * 1000,
        median_task_e2el_ms=np.median(task_e2el or 0) * 1000,
        percentiles_task_e2el_ms=[(p, np.percentile(task_e2el or 0, p) * 1000) for p in selected_percentiles],
    )

    return metrics


async def benchmark(
    api_url: str,
    base_url: str,
    tokenizer: PreTrainedTokenizerBase,
    trace: List[RequestFormat],
    logprobs: Optional[int],
    n: int,
    best_of: int,
    request_rate: List[float],
    disable_tqdm: bool,
    profile: bool,
    selected_percentile_metrics: List[str],
    selected_percentiles: List[str],
    ignore_eos: bool,
    max_concurrency: Optional[int],
    max_output_len: int,
):
    requests = trace

    # Get the first request to validate the correctness
    print("Starting initial single prompt test run...")
    test_request: RequestFormat = requests[0]
    sampling_params = SamplingParams(
        n=n,              
        temperature=0.01,
        top_p=1.0,
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
    
    test_output: RequestOutput = await send_request(request_info=test_input)
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}")
    else:
        print("Initial test run completed. Starting main benchmark run...")

    print(f"Traffic request rate: {request_rate}")
    print(f"Maximum request concurrency: {max_concurrency}")

    # This can be used once the minimum Python version is 3.10 or higher,
    # and it will simplify the code in limited_request_func.
    #    semaphore = (asyncio.Semaphore(max_concurrency)
    #                 if max_concurrency else contextlib.nullcontext())
    semaphore = (asyncio.Semaphore(max_concurrency)
                 if max_concurrency else None)

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
                    sampling_params=sampling_params,
                    client_id=client_id,
                    client_deadline=1000,
                    api_url=api_url,
                    pbar=pbar,
                )
            )
    )

    results: List[List[RequestOutput]] = await asyncio.gather(*client_tasks)
    outputs: List[RequestOutput] = [output for result in results 
                            for output in result]

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics = calculate_metrics(
        input_requests=requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentile_metrics=selected_percentile_metrics,
        selected_percentiles=selected_percentiles,
        gootput_config_dict=None,
    )

    request_type = ["Latency", "Throughput", "Collective", "Total"]
    disable_show_by_request_type = False
    
    def print_benchmark_result(title, metric_name, format_str="{:<40} {:<10}", is_service=False, is_float=False):
        print("{s:{c}^{n}}".format(s=title, n=50, c='-'))
        indices = range(4) if not disable_show_by_request_type else [3]
        for i in indices:
            value = getattr(metrics, metric_name)[i]
            if is_float:
                print(format_str.format(f"{request_type[i]} {title.lower()} (req/s):", value))
            elif is_service:
                print(format_str.format(f"{request_type[i]} {title.lower()} :", value))
            else:
                print(format_str.format(f"{title} {request_type[i]} requests:", f"{value}/{metrics.request_num[i]}"))

    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.request_total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.request_total_output))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", metrics.output_token_throughput))
    print("{:<40} {:<10.2f}".format("Total Token throughput (tok/s):", metrics.total_token_throughput))
    
    print_benchmark_result('Successful Request', 'request_completed')
    print_benchmark_result('Deadline Meet Request', 'request_deadline_meet')
    print_benchmark_result('Service Gain', 'request_service_gain', "{:<40} {:<10.2f}", is_service=True)
    print_benchmark_result('Throughput', 'request_throughput', "{:<40} {:<10.2f}", is_float=True)
    print_benchmark_result('Goodput', 'request_goodput', "{:<40} {:<10.2f}", is_float=True)
    
    

    result = {
        "duration": benchmark_duration,
        "task_num": metrics.task_num,
        "task_completed": metrics.task_completed,
        "request_completed": metrics.request_completed,
        "request_deadline_meet": metrics.request_deadline_meet,
        #"task_service_gain": metrics.task_service_gain,
        "request_service_gain": metrics.request_service_gain,
        "total_input_tokens": metrics.request_total_input,
        "total_output_tokens": metrics.request_total_output,
        "request_throughput": metrics.request_throughput,
        "request_goodput": metrics.request_goodput,
        "output_throughput": metrics.output_token_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "errors": [output.error for output in outputs],
    }

    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c='-'))
        print("{:<40} {:<10.2f}".format(
            f"Mean {metric_name} (ms):",
            getattr(metrics, f"mean_{metric_attribute_name}_ms")))
        print("{:<40} {:<10.2f}".format(
            f"Median {metric_name} (ms):",
            getattr(metrics, f"median_{metric_attribute_name}_ms")))
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms")
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms")
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms")
        for p, value in getattr(metrics,
                                f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):",
                                            value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tbt", "TBT",
                       "Time Between Token")
    process_one_metric("task_e2el", "Task E2EL", "Task End-to-end Latency")
    process_one_metric("request_e2el", "Request", "Request End-to-end Latency")

    print("=" * 50)

    return result


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
    request_rate = [float(rate) for rate in args.request_rate.split(",")]
    if sum(request_rate) != 1.0:
        Warning("The sum of request rates is not equal to 1.0. ")
        request_rate = [rate / sum(request_rate) for rate in request_rate]
    
    benchmark_result = asyncio.run(
        benchmark(
            api_url=api_url,
            base_url=base_url,
            tokenizer=tokenizer,
            trace=trace,
            logprobs=args.logprobs,
            n=args.n,
            best_of=args.best_of,
            request_rate=request_rate,
            disable_tqdm=args.disable_tqdm,
            profile=args.profile,
            selected_percentile_metrics=args.percentile_metrics.split(","),
            selected_percentiles=[
                float(p) for p in args.metric_percentiles.split(",")
            ],
            ignore_eos=args.ignore_eos,
            max_concurrency=args.max_concurrency,
            max_output_len=args.max_output_len,
        ))

    # Save config and results to json
    if args.save_result:
        result_json: Dict[str, Any] = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["model_id"] = model_id
        result_json["tokenizer_id"] = tokenizer_id
        result_json["best_of"] = args.best_of
        result_json["num_prompts"] = len(trace)

        # Metadata
        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=")
                    result_json[kvstring[0].strip()] = kvstring[1].strip()
                else:
                    raise ValueError(
                        "Invalid metadata format. Please use KEY=VALUE format."
                    )

        # Traffic
        result_json["request_rate"] = args.request_rate

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        # Save to file
        base_model_id = model_id.split("/")[-1]
        file_name = f"{base_model_id}-{current_dt}.json"  #noqa
        if args.result_filename:
            file_name = args.result_filename
        if args.result_dir:
            file_name = os.path.join(args.result_dir, file_name)
        with open(file_name, "w", encoding='utf-8') as outfile:
            json.dump(result_json, outfile)


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
        default="example-2.json",
        help="Path to the trace file.",
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
        default=2,
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
    parser.add_argument(
        "--request-rate",
        type=str,
        default="1.0",
        help="Comma-separated list of request rates for different users. ",
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
        "--percentile-metrics",
        type=str,
        default="ttft,tbt,itl,e2el",
        help="Comma-seperated list of selected metrics to report percentils. "
        "This argument specifies the metrics to report percentiles. "
        "Allowed metric names are \"ttft\", \"tbt\", \"itl\", \"e2el\". "
        "Default value is \"ttft,tbt,itl\".")
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="99",
        help="Comma-seperated list of percentiles for selected metrics. "
        "To report 25-th, 50-th, and 75-th percentiles, use \"25,50,75\". "
        "Default value is \"99\". "
        "Use \"--percentile-metrics\" to select metrics.",
    )

    args = parser.parse_args()
    main(args)