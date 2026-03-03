import os
import re
import matplotlib.pyplot as plt
import numpy as np

from ..utils import parse_throughput

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
            jitserve_goodput[key[1]] = parse_throughput(path['normal'], path['deep'])
    jitserve_goodput = [jitserve_goodput[rps] for rps in rps_list]

    sarthiserve_goodput = {}
    for key, path in experiments.items():
        if key[0] == "fcfs" and 'normal' in path and 'deep' in path:
            sarthiserve_goodput[key[1]] = parse_throughput(path['normal'], path['deep'])
    sarthiserve_goodput = [sarthiserve_goodput[rps] for rps in rps_list]

    x = np.arange(len(rps_list))
    width = 0.30

    fig, ax = plt.subplots(figsize=(6, 6))

    rects1 = ax.bar(x - width/2, jitserve_goodput, width, label='JITServe', color='#E6B47C')
    rects2 = ax.bar(x + width/2, sarthiserve_goodput, width, label='Sarathi-Serve', color='#9467BD')

    ax.set_ylabel('Throughput (request/s)', fontsize=25)
    ax.set_xlabel('RPS', fontsize=25)
    ax.set_xticks(x)
    ax.set_xticklabels(rps_list, fontsize=25)
    ax.set_ylim(2, 4)
    ax.legend(fontsize=19, ncol=1, frameon=False)
    ax.grid(axis='y', linestyle='-', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='-', alpha=0.4)
    ax.tick_params(axis='y', labelsize=20)

    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))

    plt.tight_layout()
    plt.savefig('figure/e2e/e2e_throughput.pdf')


if __name__ == "__main__":
    plot("batch_result/e2e-throughput/")
