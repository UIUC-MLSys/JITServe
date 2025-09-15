"""Benchmark the latency of processing a single batch of requests."""
import argparse
import asyncio
import copy
import json
import numpy as np
import os
import random
import sys
import time
import torch
import warnings

from dataclasses import dataclass, field
from datetime import datetime
from torch.profiler import profile, record_function, ProfilerActivity
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
from benchmarks.trace import (client_simulator, send_request, send_collective_request, RequestInput, TaskOutput,
                              RequestOutput, Trace, BaseDataset)
from vllm.request_info import RequestInfo, RequestType

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
    # time_window_service_gain: Dict[int, List[float]]
    time_window_request_goodput: Dict[int, List[float]]
    time_window_token_goodput: Dict[int, List[float]]
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
    request_details: List[Dict[str, Any]] = field(default_factory=list)   # 新增字段

def calculate_metrics(
    outputs: List[List["RequestOutput"]],
    dur_s: float,
    tokenizer: "PreTrainedTokenizerBase",
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
    # time_window_service_gain = {}
    time_window_request_goodput = {}
    time_window_token_goodput = {}

    request_metrics = []
    
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

                req_slo_constraint = tuple(slo * (outputs[i][req_idx].collection_id % 4 + 1) for slo in slo_constraint) if slo_constraint else None
                
                meets_ttft = (ttft <= req_slo_constraint[0]) if req_slo_constraint else True
                meets_tbt = all(tbt <= req_slo_constraint[1] for tbt in tbt_list) if req_slo_constraint else True
                meets_ttlt = (ttlt <= req_slo_constraint[2]) if req_slo_constraint else True
                
                if req_slo_constraint:
                    if meets_ttft: slo_ttft_meet[task_type] += 1
                    if meets_tbt: slo_tbt_meet[task_type] += 1
                    if meets_ttlt: slo_ttlt_meet[task_type] += 1
                    
                    if (meets_ttft and meets_tbt and task_type == 0) or \
                        (meets_ttlt and task_type == 1) or \
                        (outputs[i][req_idx].finish_before_ddl and task_type == 2):
                        request_slo_meet[task_type] += 1
                        request_slo_meet[3] += 1
                        
                request_input_token_length = len(tokenizer(outputs[i][req_idx].request_input, add_special_tokens=False).input_ids)
                request_output_token_length = len(tokenizer(outputs[i][req_idx].request_output, add_special_tokens=False).input_ids)
                request_total_input += request_input_token_length
                request_total_output += request_output_token_length
                
                # TODO: finish time or start time
                finish_time = outputs[i][req_idx].request_finish_time
                service_gain = outputs[i][req_idx].request_service_gain
                
                # 计算所属时间窗口
                window_end: int = (finish_time // time_window_seconds) * time_window_seconds
                # if window_end not in time_window_service_gain:
                #     time_window_service_gain[window_end] = [0.0, 0.0, 0.0, 0.0]
                # time_window_service_gain[window_end][task_type] += service_gain
                # time_window_service_gain[window_end][3] += service_gain
                request_service_gain[task_type] += service_gain
                request_service_gain[3] += service_gain

                if ttft > 0:
                    ttft_data[task_type].append(ttft)
                    ttft_data[3].append(ttft)  # Total

                tbt_data[task_type].extend(tbt_list)
                tbt_data[3].extend(tbt_list)  # Total

                e2el_data[task_type].append(ttlt)
                e2el_data[3].append(ttlt)  # Total

                avg_tbt = np.mean(tbt_list) if tbt_list else 0.0
                # for latency sensitive requests, ttft will infuence meet_tbt_count
                token_arrive_times = np.cumsum(tbt_list) + ttft
                token_slo_times = np.array([req_slo_constraint[1] * (j + 1) + req_slo_constraint[0] for j in range(len(tbt_list))])
                # compare
                meet_tbt_count = np.sum(token_arrive_times <= token_slo_times)
                total_tbt_count = len(tbt_list)

                if window_end not in time_window_request_goodput:
                    time_window_request_goodput[window_end] = [0.0, 0.0, 0.0, 0.0]
                    time_window_token_goodput[window_end] = [0.0, 0.0, 0.0, 0.0]

                # request goodput：是否满足 SLO 就算 1 个
                if req_slo_constraint:
                    if (meets_ttft and meets_tbt and task_type == 0) or \
                       (meets_ttlt and task_type == 1) or \
                       (outputs[i][req_idx].finish_before_ddl and task_type == 2):
                        time_window_request_goodput[window_end][task_type] += 1 / time_window_seconds
                        time_window_request_goodput[window_end][3] += 1 / time_window_seconds

                # token goodput：满足 SLO 的 token 数量
                if task_type == 0:  # latency-sensitive
                    prefill_tokens = request_input_token_length if meets_ttft else 0
                    decode_tokens = meet_tbt_count
                else:  # throughput / collective
                    if meets_ttlt:
                        prefill_tokens = request_input_token_length
                        decode_tokens = request_output_token_length
                    else:
                        prefill_tokens = 0
                        decode_tokens = 0

                token_goodput = prefill_tokens + 8 * decode_tokens
                time_window_token_goodput[window_end][task_type] += token_goodput / time_window_seconds
                time_window_token_goodput[window_end][3] += token_goodput / time_window_seconds

                request_metrics.append({
                    "collection_id": outputs[i][req_idx].collection_id,
                    "request_type": task_type,
                    "input_len": request_input_token_length,
                    "output_len": request_output_token_length,
                    "ttft": ttft,
                    "avg_tbt": avg_tbt,
                    "ttlt": ttlt,
                    "meet_ttft": meets_ttft,
                    "meet_tbt": meets_tbt,
                    "meet_tbt_count": meet_tbt_count,
                    "total_tbt_count": total_tbt_count,                    
                    "meet_ttlt": meets_ttlt,
                })

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
        # time_window_service_gain=time_window_service_gain,
        time_window_request_goodput=time_window_request_goodput,
        time_window_token_goodput=time_window_token_goodput,
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
        request_details=request_metrics,   # 保存请求详细信息
    )

    return metrics

@dataclass
class TaskMetrics:
    task_num: int
    task_completed: int
    task_sum_output_len: int
    task_time_per_token: List[float]  # 每个token生成所需时间(ms)
    mean_time_per_token: float
    median_time_per_token: float
    percentiles_time_per_token: List[Tuple[float, float]]
    mean_task_latency_s: float
    median_task_latency_s: float
    percentiles_task_latency_s: List[Tuple[float, float]]

def calculate_task_metrics(
    tasks: List[TaskOutput],
    tokenizer: PreTrainedTokenizerBase,
    selected_percentiles: List[float] = [50, 90, 95, 99],
) -> TaskMetrics:
    """
    计算TaskOutput的相关指标
    
    Args:
        tasks: TaskOutput列表
        tokenizer: 用于计算token长度的tokenizer
        selected_percentiles: 需要计算的百分位数
        
    Returns:
        TaskMetrics对象包含所有统计指标
    """
    task_num = len(tasks)
    task_completed = 0
    task_sum_output_len = 0
    time_per_token_list = []  # 存储每个token生成时间(ms)
    task_latency_list = []    # 存储任务延迟(ms)
    
    for task in tasks:
        if task.task_latency <= 0:
            continue
            
        task_completed += 1
        
        # 计算该task所有output的总token长度
        task_output_len = sum(
            len(tokenizer(output, add_special_tokens=False).input_ids)
            for output in task.task_output
        )
        task_sum_output_len += task_output_len
        
        if task_output_len > 0:
            # 计算每个token生成所需时间(ms)
            time_per_token = (task.task_latency * 1000) / task_output_len
            time_per_token_list.append(time_per_token)
        
        # 收集延迟数据(秒)
        task_latency_list.append(task.task_latency)
    
    # 计算统计指标
    def compute_percentiles(data: List[float], percentiles: List[float]) -> List[Tuple[float, float]]:
        if not data:
            return [(p, 0.0) for p in percentiles]
        return [(p, np.percentile(data, p)) for p in percentiles]
    
    # 每token时间统计
    mean_time_per_token = np.mean(time_per_token_list) if time_per_token_list else 0
    median_time_per_token = np.median(time_per_token_list) if time_per_token_list else 0
    time_per_token_percentiles = compute_percentiles(time_per_token_list, selected_percentiles)
    
    # 延迟统计(毫秒)
    mean_latency = np.mean(task_latency_list) if task_latency_list else 0
    median_latency = np.median(task_latency_list) if task_latency_list else 0
    latency_percentiles = compute_percentiles(task_latency_list, selected_percentiles)
    
    return TaskMetrics(
        task_num=task_num,
        task_completed=task_completed,
        task_sum_output_len=task_sum_output_len,
        task_time_per_token=time_per_token_list,
        mean_time_per_token=mean_time_per_token,
        median_time_per_token=median_time_per_token,
        percentiles_time_per_token=time_per_token_percentiles,
        mean_task_latency_s=mean_latency,
        median_task_latency_s=median_latency,
        percentiles_task_latency_s=latency_percentiles,
    )

def print_benchmark_results(metrics: BenchmarkMetrics, task_metrics: TaskMetrics, 
                            time_window_seconds: float, w_prefill: int = 1, w_decode: int = 8) -> None:
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

    # 初始化统计
    type_prefill_token = [0.0, 0.0, 0.0, 0.0]
    type_decode_token = [0.0, 0.0, 0.0, 0.0]
    type_weighted_token = [0.0, 0.0, 0.0, 0.0]
    type_total_weighted = [0.0, 0.0, 0.0, 0.0]

    for req in metrics.request_details:
        rtype = req["request_type"]  # 0=Latency, 1=Throughput, 2=Collective

        # prefill / decode token goodput 按请求类型规则
        if rtype == 0:  # Latency-sensitive
            prefill_tokens = req["input_len"] if req["meet_ttft"] else 0
            decode_tokens = req["meet_tbt_count"]
        else:  # Throughput / Collective
            if req["meet_ttlt"]:
                prefill_tokens = req["input_len"]
                decode_tokens = req["output_len"]
            else:
                prefill_tokens = 0
                decode_tokens = 0

        weighted_tokens = w_prefill * prefill_tokens + w_decode * decode_tokens
        total_weighted_tokens = w_prefill * req["input_len"] + w_decode * req["output_len"]

        # 累加各类别
        type_prefill_token[rtype] += prefill_tokens
        type_decode_token[rtype] += decode_tokens
        type_weighted_token[rtype] += weighted_tokens
        type_total_weighted[rtype] += total_weighted_tokens

        # 累加总计
        type_prefill_token[3] += prefill_tokens
        type_decode_token[3] += decode_tokens
        type_weighted_token[3] += weighted_tokens
        type_total_weighted[3] += total_weighted_tokens

    # 请求级统计表格
    print("{:<15} {:<10} {:<10} {:<10} {:<10} {:<10} {:<15}".format(
        "Type", "Total", "Completed", "SLO Met", "SLO Attain", "Throughput", "Goodput"))
    print(separator)
    for i in range(4):
        slo_attain = (metrics.request_slo_meet[i] / metrics.request_num[i] if metrics.request_num[i] > 0 else 0.0)
        print("{:<15} {:<10} {:<10} {:<10} {:<10.2f} {:<10.2f} {:<15.2f}".format(
            request_types[i],
            metrics.request_num[i],
            metrics.request_completed[i],
            metrics.request_slo_meet[i],
            slo_attain,
            metrics.request_throughput[i],
            metrics.request_goodput[i]
        ))
    print(separator)

    print("{:<15} {:<15} {:<15} {:<15}".format("Type", "Prefill Gpt", "Decode Gpt", "Weighted Gpt"))
    print(separator)
    for i in range(4):
        prefill_gp = type_prefill_token[i] / metrics.duration
        decode_gp = type_decode_token[i] / metrics.duration
        weighted_gp = type_weighted_token[i] / metrics.duration
        print("{:<15} {:<15.2f} {:<15.2f} {:<15.2f}".format(
            request_types[i], prefill_gp, decode_gp, weighted_gp
        ))
    print(separator)

    # 任务级统计
    print("\nTask-Level Statistics:")
    print(separator)
    print("{:<40} {:<20}".format("Total Tasks", task_metrics.task_num))
    print("{:<40} {:<20}".format("Completed Tasks", task_metrics.task_completed))
    print("{:<40} {:<20}".format("Total Task Output Tokens", task_metrics.task_sum_output_len))
    print("{:<40} {:<20.2f}".format("Mean Time per Token (ms/token)", task_metrics.mean_time_per_token))
    print("{:<40} {:<20.2f}".format("Median Time per Token (ms/token)", task_metrics.median_time_per_token))
    print("{:<40} {:<20.2f}".format("Mean Task Latency (s)", task_metrics.mean_task_latency_s))
    print("{:<40} {:<20.2f}".format("Median Task Latency (s)", task_metrics.median_task_latency_s))
    print(separator)

    # 每token时间百分位数
    print("\nTime per Token Percentiles (ms/token):")
    print("{:<10} {:<15}".format("Percentile", "Value"))
    for p, val in task_metrics.percentiles_time_per_token:
        print("{:<10} {:<15.2f}".format(f"P{int(p)}", val))
    print(separator)

    # 任务延迟百分位数
    print("\nTask Latency Percentiles (s):")
    print("{:<10} {:<15}".format("Percentile", "Value"))
    for p, val in task_metrics.percentiles_task_latency_s:
        print("{:<10} {:<15.2f}".format(f"P{int(p)}", val))
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

    def print_time_window_goodput(metrics: BenchmarkMetrics):
        print("\n{:=^80}".format(" Time Window Request Goodput "))
        print("{:<10} {:<15} {:<15} {:<15} {:<15}".format(
            "Window(s)", "Latency", "Throughput", "Collective", "Total"))
        print("-" * 80)
        for window_end, vals in sorted(metrics.time_window_request_goodput.items()):
            print("{:<10} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f}".format(
                window_end, vals[0], vals[1], vals[2], vals[3]
            ))

        print("\n{:=^80}".format(" Time Window Token Goodput "))
        print("{:<10} {:<15} {:<15} {:<15} {:<15}".format(
            "Window(s)", "Latency", "Throughput", "Collective", "Total"))
        print("-" * 80)
        for window_end, vals in sorted(metrics.time_window_token_goodput.items()):
            print("{:<10} {:<15.0f} {:<15.0f} {:<15.0f} {:<15.0f}".format(
                window_end, vals[0], vals[1], vals[2], vals[3]
            ))
        print("=" * 80)
    print_time_window_goodput(metrics)
    print(header_separator)

    print("\nPer-Request Metrics:")
    print("{:<15} {:<10} {:<10} {:<10} {:<10} {:<10} {:<12} {:<12} {:<15} {:<12}".format(
        "CollectionID", "InLen", "OutLen", "TTFT", "AvgTBT", "TTLT",
        "Meet_TTFT", "Meet_TBT", "Meet_TBT_Ratio", "Meet_TTLT"
    ))
    print(separator)
    for req in metrics.request_details:
        if req["request_type"] == 0:
            meet_ttft = str(req["meet_ttft"])
            meet_tbt = str(req["meet_tbt"])
            meet_tbt_ratio = f"{req['meet_tbt_count']}/{req['total_tbt_count']}"
            meet_ttlt = "\\"
        else:  # request_type == 1 or 2
            meet_ttft = "\\"
            meet_tbt = "\\"
            meet_tbt_ratio = "\\"
            meet_ttlt = str(req["meet_ttlt"])

        print("{:<15} {:<10} {:<10} {:<10.2f} {:<10.2f} {:<10.2f} {:<12} {:<12} {:<15} {:<12}".format(
            req["collection_id"],
            req["input_len"],
            req["output_len"],
            req["ttft"],
            req["avg_tbt"],
            req["ttlt"],
            meet_ttft,
            meet_tbt,
            meet_tbt_ratio,
            meet_ttlt,
        ))
    print(separator)

async def benchmark(
    api_url: str,
    base_url: str,
    model: str,
    tokenizer: PreTrainedTokenizerBase,
    trace: List[RequestInfo],
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
    penalty_factor: int,
    tot_structure: Tuple[int, int],             # (tot_thoughts, tot_rounds)
    trace_pattern: Optional[str] = "hetero",  # e.g., "homo", "hetero", "random"
    output_pattern: Optional[str] = None,  # e.g., "same"
    is_stream: bool = True,
):
    trace_len = len(trace)
    total_output_len = sum(
        (min(item.output_len, 256) * 14 if item.request_type == RequestType.COLLECTIVE else min(item.output_len, 1024))
        for item in trace
    )
    total_request_number = sum(
        (14 if item.request_type == RequestType.COLLECTIVE else 1)
        for item in trace
    )
    average_output_len = total_output_len // total_request_number
    # deepcopy
    if trace_pattern == "homo":
        print("Using homo trace pattern for requests.")
        update_trace = [copy.deepcopy(item) for item in (trace * (num_prompts // trace_len + 1))[:num_prompts]]
        total_input_len = sum(update_trace[i].input_len for i in range(len(update_trace)))
        requests = []
        input_len = total_input_len // len(update_trace)
        vocab_size = tokenizer.vocab_size
        num_special_tokens = tokenizer.num_special_tokens_to_add()
        real_input_len = input_len - num_special_tokens
        offsets = np.random.randint(0, vocab_size, size=num_prompts)

        for i, real_trace in enumerate(update_trace):
            inner_seq = (
                (offsets[i] + i + np.arange(real_input_len)) % vocab_size
            ).tolist()
            token_sequence = inner_seq
            prompt = tokenizer.decode(token_sequence)
            # After decoding the prompt we have to encode and decode it again.
            # This is done because in some cases N consecutive tokens
            # give a string tokenized into != N number of tokens.
            # For example for GPT2Tokenizer:
            # [6880, 6881] -> ['Ġcalls', 'here'] ->
            # [1650, 939, 486] -> ['Ġcall', 'sh', 'ere']
            # To avoid uncontrolled change of the prompt length,
            # the encoded sequence is truncated before being decode again.
            re_encoded_sequence = tokenizer.encode(prompt, add_special_tokens=False)[
                :real_input_len
            ]
            prompt = tokenizer.decode(re_encoded_sequence)
            real_trace.prompt = prompt
            real_trace.input_len = len(re_encoded_sequence)
            
            requests.append(real_trace)
    elif trace_pattern == "hetero":
        print("Using hetero trace pattern for requests.")
        update_trace = [copy.deepcopy(item) for item in (trace * (num_prompts // trace_len + 1))[:num_prompts]]
        total_input_len = sum(update_trace[i].input_len for i in range(len(update_trace)))
        requests = []
        input_len = total_input_len // len(update_trace)
        vocab_size = tokenizer.vocab_size
        num_special_tokens = tokenizer.num_special_tokens_to_add()
        offsets = np.random.randint(0, vocab_size, size=num_prompts)

        for i, real_trace in enumerate(update_trace):
            real_input_len = update_trace[i].input_len - num_special_tokens
            inner_seq = (
                (offsets[i] + i + np.arange(real_input_len)) % vocab_size
            ).tolist()
            token_sequence = inner_seq
            prompt = tokenizer.decode(token_sequence)
            # After decoding the prompt we have to encode and decode it again.
            # This is done because in some cases N consecutive tokens
            # give a string tokenized into != N number of tokens.
            # For example for GPT2Tokenizer:
            # [6880, 6881] -> ['Ġcalls', 'here'] ->
            # [1650, 939, 486] -> ['Ġcall', 'sh', 'ere']
            # To avoid uncontrolled change of the prompt length,
            # the encoded sequence is truncated before being decode again.
            re_encoded_sequence = tokenizer.encode(prompt, add_special_tokens=False)[
                :real_input_len
            ]
            prompt = tokenizer.decode(re_encoded_sequence)
            real_trace.prompt = prompt
            real_trace.input_len = len(re_encoded_sequence)
            requests.append(real_trace)
    elif trace_pattern is None:
        print("Using real trace pattern for requests.")
        requests = [copy.deepcopy(item) for item in (trace * (num_prompts // trace_len + 1))[:num_prompts]]

    if output_pattern == "same":
        for id, request in enumerate(requests):
            request.output_len = average_output_len
            request.collection_id = id

    for id, request in enumerate(requests):
        request.collection_id = id

    # Get the first request to validate the correctness
    print("Starting initial single prompt test run...")
    test_request: RequestInfo = requests[0]
    test_request.output_len = 10
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
        slo_constraint=slo_constraint,
        sampling_params=sampling_params,
        client_id=0,
        api_url=api_url,
    )
    
    if test_input.request.request_type == RequestType.COLLECTIVE:
        test_output: List[RequestOutput] = await send_collective_request(request_info=test_input,
                                                                         model_name=model, 
                                                                         tot_structure=tot_structure,
                                                                         penalty_factor=penalty_factor,
                                                                         client_deadline=200)
    else:
        test_output: List[RequestOutput] = await send_request(request_info=test_input, model_name=model, client_deadline=200)
    if not test_output[0].success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output[0].error}")
    else:
        print("Initial test run completed. Starting main benchmark run...")

    print(f"Traffic request rate: {request_rate}")
    print(f"Maximum request concurrency: {max_concurrency}")
    
    for request in requests:
        if request.request_type == RequestType.LATENCY:
            request.deadline = slo_constraint[0] + slo_constraint[1] * request.output_len
        elif request.request_type == RequestType.THROUGHPUT:
            request.deadline = slo_constraint[2]
        elif request.request_type == RequestType.COLLECTIVE:
            request.deadline = slo_constraint[2]
        request.deadline *= 1000

    pbar = None if disable_tqdm else tqdm(total=len(requests))

    benchmark_start_time = time.perf_counter()
    # outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)
    #if request_rate is None:
    #    request_per_user = [requests]
    #else:
    #    request_per_user = BaseDataset.divide_by_rate(requests, request_rate)
    request_per_user = [requests]
        
    
    client_tasks = []
    for client_id, input_requests in enumerate(request_per_user):
        client_tasks.append(
            asyncio.create_task(
                client_simulator(
                    input_requests=input_requests,
                    slo_constraint=slo_constraint,
                    penalty_factor=penalty_factor,
                    model_name=model,
                    poisson_lambda=poisson_lambda,
                    burst=burst,
                    sampling_params=sampling_params,
                    client_id=client_id,
                    client_deadline=5000,
                    api_url=api_url,
                    tot_structure=tot_structure,
                    is_stream=is_stream,
                    pbar=pbar,
                )
            )
        )

    results: List[Tuple[List[RequestOutput], List[TaskOutput]]] = await asyncio.gather(*client_tasks)
    outputs: List[RequestOutput] = [output for result in results 
                            for output in result[0]]
    tasks: List[TaskOutput] = [task_output for result in results
                            for task_output in result[1]]
    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics = calculate_metrics(
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentiles=selected_percentiles,
        slo_constraint=slo_constraint,
        time_window_seconds=60,
    )

    task_metrics = calculate_task_metrics(
        tasks=tasks,
        tokenizer=tokenizer,
        selected_percentiles=selected_percentiles,
    )
    
    print_benchmark_results(metrics, task_metrics, time_window_seconds=60)


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    trace: List[RequestInfo] = Trace.load_trace(args.trace_path)

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
        "penalty_factor": args.penalty_factor,
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
            model=model_id,
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
            penalty_factor=args.penalty_factor,
            tot_structure=(args.tot_thoughts, args.tot_rounds),
            is_stream=args.is_stream,
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
        default="benchmarks/dataset/trace/lmsys.json",
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
        choices=["True", "False"],
        default="False",
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
        "--penalty-factor",
        type=int,
        default=1,
        help="Penalty factor for the benchmark. "
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
        "--is-stream",
        choices=["True", "False"],
        default="True",
        help="Whether to use streaming mode for requests (default: True)",
    )
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
        default="1,25,50,75,95,99",
        help="Comma-seperated list of percentiles for selected metrics. "
        "To report 25-th, 50-th, and 75-th percentiles, use \"25,50,75\". "
        "Default value is \"99\". "
        "Use \"--percentile-metrics\" to select metrics.",
    )

    args = parser.parse_args()
    
    # Convert string arguments to boolean
    args.burst = args.burst == "True"
    args.is_stream = args.is_stream == "True"
    
    main(args)