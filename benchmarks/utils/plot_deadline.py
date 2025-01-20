import matplotlib.pyplot as plt
import numpy as np

data = {
    "4": {
        "fcfs": [63, 10, 0],
        "slo": [118, 35, 0]
    },
    "8": {
        "fcfs": [147, 32, 0],
        "slo": [232, 58, 0]
    },
    "16": {
        "fcfs": [408, 105, 84],
        "slo": [518, 145, 56]
    },
    "32": {
        "fcfs": [743, 208, 140],
        "slo": [731, 209, 140]
    },
}

batch_sizes = list(data.keys())
categories = ['latency', 'throughput', 'collective']
colors = ['#7e99f4', '#7ab656', '#cc7c71']

fig, ax = plt.subplots(figsize=(12, 8))

x = np.arange(len(batch_sizes))  # the label locations
width = 0.35  # the width of the bars

for i, batch_size in enumerate(batch_sizes):
    fcfs_values = data[batch_size]['fcfs']
    slo_values = data[batch_size]['slo']
    
    # Plot FCFS bars
    bottom = 0
    for j, value in enumerate(fcfs_values):
        ax.bar(x[i] - width/2, value, width, bottom=bottom, label=f'FCFS {categories[j]}' if i == 0 else "", color=colors[j])
        bottom += value
    
    # Add total sum text for FCFS
    total_fcfs = sum(fcfs_values)
    ax.text(x[i] - width/2, bottom, str(total_fcfs), ha='center', va='bottom')
    
    # Plot SLO bars
    bottom = 0
    for j, value in enumerate(slo_values):
        ax.bar(x[i] + width/2, value, width, bottom=bottom, label=f'SLO {categories[j]}' if i == 0 else "", color=colors[j], hatch='//')
        bottom += value
        
        # Add total sum text for SLO
    total_slo = sum(slo_values)
    ax.text(x[i] + width/2, bottom, str(total_slo), ha='center', va='bottom')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Batch Size')
ax.set_ylabel('Number of Requests')
ax.set_title('Deadline Meet by Batch Size and Policy')
ax.set_xticks(x)
ax.set_xticklabels([f'{bs}\nfcfs/slo' for bs in batch_sizes])
ax.legend()

fig.tight_layout()

plt.savefig('result_deadline_meet.png')