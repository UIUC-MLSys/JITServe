import matplotlib.pyplot as plt
import numpy as np

data = {
    "4": {
        "fcfs": [370316.3157675008, 898042.7516533738],
        "slo": [980946.3483489817, 1582438.791395333]
    },
    "8": {
        "fcfs": [86179.55406947294, 296541.4741975983],
        "slo": [28344.29838997312, 901424.5299689233]
    },
    "16": {
        "fcfs": [18479.41093001282, 51943.54827583943],
        "slo": [867.4021284969058, 459572.4568400282]
    },
    "32": {
        "fcfs": [71.04384648846462, 12772.364545257313],
        "slo": [72.47660952270962, 29171.91009877599]
    },
}

batch_sizes = list(data.keys())
categories = ['p50', 'p99']
colors = {'fcfs': (0.1, 0.2, 0.5), 'slo': (0.5, 0.2, 0.1)}

fig, ax = plt.subplots(figsize=(12, 8))

x = np.arange(len(batch_sizes))  # the label locations
width = 0.12  # the width of the bars
gap = 0.2  # the gap between p50 and p99 bars

for i, batch_size in enumerate(batch_sizes):
    fcfs_values = [ttft / 1000 for ttft in data[batch_size]['fcfs']]
    slo_values = [ttft / 1000 for ttft in data[batch_size]['slo']]
    
    # Plot p50 bars
    ax.bar(x[i] - width - gap/2, fcfs_values[0], width, label='FCFS p50' if i == 0 else "", color="#72b063")
    ax.bar(x[i] - gap/2, slo_values[0], width, label='SLO p50' if i == 0 else "", color="#b8dbb3", hatch='//')
    
    # Plot p99 bars
    ax.bar(x[i] + gap/2, fcfs_values[1], width, label='FCFS p99' if i == 0 else "", color="#719aac")
    ax.bar(x[i] + width + gap/2, slo_values[1], width, label='SLO p99' if i == 0 else "", color="#94c6cd", hatch='//')
    
    # Add total sum text for p50 and p99
    ax.text(x[i] - width - gap/2, fcfs_values[0], f'{fcfs_values[0]:.2f}', ha='center', va='bottom')
    ax.text(x[i] - gap/2, slo_values[0], f'{slo_values[0]:.2f}', ha='center', va='bottom')
    ax.text(x[i] + gap/2, fcfs_values[1], f'{fcfs_values[1]:.2f}', ha='center', va='bottom')
    ax.text(x[i] + width + gap/2, slo_values[1], f'{slo_values[1]:.2f}', ha='center', va='bottom')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Batch Size')
ax.set_ylabel('Time To First Token (TTFT) (s)')
ax.set_title('TTFT by Batch Size and Policy')
ax.set_xticks(x)
ax.set_xticklabels([f'{bs}' for bs in batch_sizes])
ax.legend()

# Optionally, use a logarithmic scale for the y-axis
ax.set_yscale('log')

fig.tight_layout()

plt.savefig('result_ttft.png')