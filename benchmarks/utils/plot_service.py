import matplotlib.pyplot as plt
import numpy as np

data = {
    "4": {
        "fcfs": 133819.45430040563,
        "slo": 190155.4211839112
    },
    "8": {
        "fcfs": 291464.95358410315,
        "slo": 441769.8312552163
    },
    "16": {
        "fcfs":624470.8493909441,
        "slo": 801306.9395596284
    },
    "32": {
        "fcfs": 891124.57729933,
        "slo": 922191.677207857
    },
}

batch_sizes = list(data.keys())
fcfs_values = [data[batch_size]["fcfs"] for batch_size in batch_sizes]
slo_values = [data[batch_size]["slo"] for batch_size in batch_sizes]

x = np.arange(len(batch_sizes))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x - width/2, fcfs_values, width, label='FCFS', color='#7e99f4')
rects2 = ax.bar(x + width/2, slo_values, width, label='SLO', color='#cc7c71')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Batch Size')
ax.set_ylabel('Service Gain (weight)')
ax.set_title('Service Gain by Batch Size and Policy')
ax.set_xticks(x)
ax.set_xticklabels(batch_sizes)
ax.legend()

# Optionally, use a logarithmic scale for the y-axis
ax.set_yscale('log')

# Function to add labels on top of the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(int(height)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.savefig("result_service_gain.png")

