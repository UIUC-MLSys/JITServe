import os
import re
import matplotlib.pyplot as plt
import numpy as np

from utils import parse_goodput

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
            key = (strategy, rps)
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

    precise_goodput = {}
    for key, path in experiments.items():
        if key[0] == "oracle" and 'normal' in path and 'deep' in path:
            precise_goodput[key[1]] = parse_goodput(path['normal'], path['deep'], 'token_goodput')
    precise_goodput = [precise_goodput[rps] for rps in rps_list]

    # jitserve_goodput = [7350, 7533, 7737, 7637, 7]
    # precise_goodput = [7542, 7808, 7782, 7890, 7843, 8300]

    x = np.arange(len(rps_list))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 6))

    bars1 = ax.bar(x - width/2, jitserve_goodput, width, color='#E6B47C',  label='JITS')
    bars2 = ax.bar(x + width/2, precise_goodput, width, color='#9467BD',label='JITS*')

    ax.set_ylabel('Token Goodput (token/s)', fontsize=25)
    ax.set_xlabel('RPS', fontsize=25)
    ax.set_xticks(x)
    ax.set_xticklabels(rps_list, fontsize=20)
    ax.tick_params(axis='y', labelsize=18)

    # y_min = min(min(jitserve_goodput), min(precise_goodput)) - 50
    # y_max = max(max(jitserve_goodput), max(precise_goodput)) + 50
    ax.set_ylim(6000, 8500)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))

    ax.legend(fontsize=19, frameon=False, ncol=2, loc='upper left')
    ax.grid(axis='y', linestyle='-', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('figure/e2e/oracle_bar.pdf', dpi=300)

if __name__ == "__main__":
    plot("batch_result/e2e-oracle/")