import io
import re
import os
import pandas as pd
from collections import defaultdict, Counter


# given a directory and parse all relevant files in the directory given metrics

def parse_goodput_timeline(file_path: str, metrics: list[str], is_deepresearch: bool = False) -> dict:
    results = {}
    metric_map = {
        'request_goodput': "Time Window Request Goodput",
        'token_goodput': "Time Window Token Goodput"
    }

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return {}

    for metric_key, search_title in metric_map.items():
        if metric_key not in metrics:
            continue
            
        table_lines = []
        is_capturing = False
        header_found = False

        for line in lines:
            if search_title in line and "====" in line and metric_map[metric_key] in line:
                is_capturing = True
                continue
            
            if is_capturing:
                if "Window(s)" in line:
                    header_found = True
                    table_lines.append(line.strip())
                    continue       
                if header_found and (line.strip().startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))):
                    table_lines.append(line.strip())
                elif header_found and ("====" in line or line.strip() == ""):
                    break
        
        if table_lines:
            table_content = "\n".join(table_lines)
            df = pd.read_csv(io.StringIO(table_content), sep=r'\s+')
            df = df.apply(pd.to_numeric, errors='coerce')
            timestamp = df["Window(s)"].tolist()
            if is_deepresearch:
                total = df["DeepResearch"].tolist()
            else:
                total = df["Total"].tolist()
            results[metric_key] = {t: data for t, data in zip(timestamp, total)}

    return results


def parse_duration(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        duration = float(re.search(r"Duration \(seconds\)\s+([\d.]+)", content).group(1))
        return duration
    except Exception:
        return None


def parse_normal_goodput(file_path, metric):
    if metric not in ['request_goodput', 'token_goodput']:
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if metric == 'request_goodput':
            weighted_gpt = float(re.search(r"Total\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)", content).group(1))
        else:
            weighted_gpt = float(re.search(r"Total\s+[\d.]+\s+[\d.]+\s+([\d.]+)", content).group(1))
        return weighted_gpt
    except Exception:
        return None


def parse_deepresearch_goodput(file_path, metric):
    if metric not in ['request_goodput', 'token_goodput']:
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if metric == 'request_goodput':
            slo_goodput = float(re.search(r"SLO Goodput\s+([\d.]+)", content).group(1))
        else:
            slo_goodput = float(re.search(r"SLO Token Goodput\s+([\d.]+)", content).group(1))
        return slo_goodput
    except Exception:
        return None


def parse_normal_request_num(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        req_num = float(re.search(r"Total\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+", content).group(1))
        return req_num
    except Exception:
        return None


def parse_deepresearch_request_num(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        req_num = float(re.search(r"Total Request\s+([\d.]+)", content).group(1))
        return req_num
    except Exception:
        return None
    

def parse_timeline(normal_path, deepresearch_path, metrics):
    normal_data = parse_goodput_timeline(normal_path, metrics, is_deepresearch=False)
    deep_data = parse_goodput_timeline(deepresearch_path, metrics, is_deepresearch=True)

    result = {}
    for metric in metrics:
        normal_data = normal_data.get(metric)
        deep_data = deep_data.get(metric)

        result[metric] = dict(Counter(normal_data) + Counter(deep_data))

    return result


def parse_goodput(normal_path, deepresearch_path, metric):
    normal_data = parse_normal_goodput(normal_path, metric)
    deep_data = parse_deepresearch_goodput(deepresearch_path, metric)

    normal_duration = parse_duration(normal_path)
    deep_duration = parse_duration(deepresearch_path)

    exp_duration = max(normal_duration, deep_duration)
    exp_goodput = (normal_data * normal_duration + deep_data * deep_duration) / exp_duration

    return exp_goodput


def parse_throughput(normal_path, deepresearch_path, metric):
    normal_data = parse_normal_request_num(normal_path)
    deep_data = parse_deepresearch_request_num(deepresearch_path)

    normal_duration = parse_duration(normal_path)
    deep_duration = parse_duration(deepresearch_path)

    exp_duration = max(normal_duration, deep_duration)
    exp_throughput = (normal_data + deep_data) / exp_duration

    return exp_throughput