import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from ..utils import parse_goodput

my_rcparams = {
        # general
        "figure.figsize": (6, 4),
        "figure.dpi": 600,
        "figure.constrained_layout.use": True,
        # font
        # "font.family": "Arial",
        "font.size": 16,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "legend.fontsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        # lines
        "lines.linewidth": 0.8,
        "lines.markersize": 8,
        "lines.markeredgewidth": 1.5,
        # axes
        "axes.linewidth": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.top": False,
        "ytick.right": False,
        # grid
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.8,
        # legend
        "legend.frameon": False,
        "legend.loc": "best",
        # savefig
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    }

def plot(directory):
    files = os.listdir(directory)
    experiments = {}
    rps_set = set()
    for f in files:
        if not f.endswith(".log"):
            continue

        match = re.match(r"(scheduler|deepresearch)_([^_]+)_([\d.]+)_", f)
        if match:
            prefix, strategy, rps = match.groups()
            rps_set.add(float(rps))
            key = (strategy, float(rps))
            experiments.setdefault(key, {})
            if prefix == "scheduler":
                experiments[key]['normal'] = os.path.join(directory, f)
            else:
                experiments[key]['deep'] = os.path.join(directory, f)

    rps_list = sorted(list(rps_set))
    jitserve_goodput = {}
    # get jitserve goodput
    for key, path in experiments.items():
        if key[0] == "jitserve" and 'normal' in path and 'deep' in path:
            jitserve_goodput[key[1]] = parse_goodput(path['normal'], path['deep'], 'token_goodput')
    jitserve_goodput = [jitserve_goodput[rps] for rps in rps_list]

    slosserve_goodput = {}
    for key, path in experiments.items():
        if key[0] == "slosserve" and 'normal' in path and 'deep' in path:
            slosserve_goodput[key[1]] = parse_goodput(path['normal'], path['deep'], 'token_goodput')
    slosserve_goodput = [slosserve_goodput[rps] for rps in rps_list]

    data = {
        'RPS': rps_list,
        'JITServe': jitserve_goodput,
        'SLOServe': slosserve_goodput
    }

    df = pd.DataFrame(data)
    sns.set_theme(style="whitegrid", rc=my_rcparams)
    plt.figure(figsize=(10, 6))
    
    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.plot(df['RPS'], df['JITServe'], marker='o', linewidth=2.5, label='JITServe', color='#8c6bb1')
    plt.plot(df['RPS'], df['SLOServe'], marker='s', linewidth=2.5, label='SLOs-Serve', color='#E6B47C')
    
    plt.xlabel('RPS (Requests Per Second)')
    plt.ylabel('Token Goodput (tokens/s)')
    
    plt.xticks(df['RPS'])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig('figure/ablation/goodput_comparison.pdf', dpi=300)


if __name__ == "__main__":
    plot("batch_result/ablation-slo/")