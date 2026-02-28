import re
import os
from collections import defaultdict

def parse_normal_log(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        duration = float(re.search(r"Duration \(seconds\)\s+([\d.]+)", content).group(1))
        weighted_gpt = float(re.search(r"Total\s+[\d.]+\s+[\d.]+\s+([\d.]+)", content).group(1))
        return duration, weighted_gpt
    except Exception:
        return None, None

def parse_deepresearch_log(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        duration = float(re.search(r"Duration \(seconds\)\s+([\d.]+)", content).group(1))
        slo_token_goodput = float(re.search(r"SLO Token Goodput\s+([\d.]+)", content).group(1))
        return duration, slo_token_goodput
    except Exception:
        return None, None

def main(directory):
    experiments = defaultdict(dict)
    
    # 1. 扫描文件夹进行配对
    files = os.listdir(directory)
    for f in files:
        if not f.endswith(".log"):
            continue
            
        # 匹配 scheduler_xxx_yy_... 或 deepresearch_xxx_yy_...
        # 这里的正则捕获 xxx 和 yy
        match = re.match(r"(scheduler|deepresearch)_([^_]+)_([\d.]+)_", f)
        if match:
            prefix, strategy, rps = match.groups()
            key = (strategy, rps)
            if prefix == "scheduler":
                experiments[key]['normal'] = os.path.join(directory, f)
            else:
                experiments[key]['deep'] = os.path.join(directory, f)

    results = []
    for (strategy, rps), paths in experiments.items():
        if 'normal' in paths and 'deep' in paths:
            d_norm, g_norm = parse_normal_log(paths['normal'])
            d_deep, g_deep = parse_deepresearch_log(paths['deep'])
            
            if d_norm is not None and d_deep is not None:
                # 核心计算公式
                total_work = (g_norm * d_norm) + (g_deep * d_deep)
                max_duration = max(d_norm, d_deep)
                overall_goodput = total_work / max_duration
                
                results.append({
                    'Strategy': strategy,
                    'RPS': rps,
                    'Goodput': overall_goodput
                })

    results.sort(key=lambda x: (x['Strategy'], float(x['RPS'])))
    
    print(f"{'-'*45}")
    print(f"{'Strategy':<15} | {'RPS':<8} | {'Combined Goodput':<15}")
    print(f"{'-'*45}")
    for res in results:
        print(f"{res['Strategy']:<15} | {res['RPS']:<8} | {res['Goodput']:<15.2f}")
    print(f"{'-'*45}")

if __name__ == "__main__":
    target_dir = "/home/exouser/Concord/batch_result/test/"
    main(target_dir)