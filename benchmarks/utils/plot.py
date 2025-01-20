
import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Your data (assuming you have loaded them into these lists)
data = {
    "request_completed": [1544, 436, 266, 2246],
    "request_service_gain": [778241.469462919, 330860.756688422, 375464.06145425764, 1484566.2876056016],
    "output_throughput": [1192.9697415480723],
    "mean_ttft_ms": [5740.733635749832],
    "median_ttft_ms": [2922.01430501882],
    "p99_ttft_ms": [25108.431074200198],
    "mean_tbt_ms": [47.72715097928353],
    "median_tbt_ms": [46.88106098910794],
    "p99_tbt_ms": [75.3567950159777],
    "mean_request_e2el_ms": [20255.21531623858],
    "median_request_e2el_ms": [18248.65688299178],
    "p99_request_e2el_ms": [62239.29289001482],
}


def load_data_from_folder(folder_path, attributes):
    data = {attr: [] for attr in attributes}
    policies = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            policy_name = filename.split('-')[-3]
            if policy_name not in ['sjf', 'fcfs', 'srtf', 'slo', 'vtc']:
                continue
            policies.append(policy_name)
            with open(os.path.join(folder_path, filename), 'r') as file:
                result = json.load(file)
                for attr in attributes:
                    data[attr].append(result.get(attr, [0]))

    return data, policies

def plot_data(data, policies, attributes, output_folder):
    num_policies = len(policies)
    x = np.arange(num_policies)  # the label locations
    width = 0.2  # the width of the bars

    for attr in attributes:
        if attr == 'request_deadline_meet' or attr == 'request_service_gain':
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, policy in enumerate(policies):
            ax.bar(x + i * width, data[attr][i], width, label=policy)

        ax.set_xlabel('Policy')
        ax.set_ylabel(attr)
        ax.set_title(attr)
        ax.set_xticks(x + width * (num_policies - 1) / 2)
        ax.set_xticklabels(policies)
        ax.legend()

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{attr}.png"))
        plt.close()
        
def plot_request_service_gain_meet(data, policies, output_folder):
    request_types = ['latency-sensitive', 'throughput-intensive', 'collective', 'total']
    num_policies = len(policies)
    num_request_types = len(request_types)
    x = np.arange(num_request_types)  # the label locations
    width = 0.15  # the width of the bars

    # Find the index of the 'fcfs' policy
    fcfs_index = policies.index('fcfs')

    fig, ax = plt.subplots(figsize=(12, 8))
    for i, policy in enumerate(policies):
        if policy == 'fcfs':
            continue  # Skip fcfs as it is the base
        values = [(data['request_service_gain'][i][j] / data['request_service_gain'][fcfs_index][j]) for j in range(num_request_types)]
        ax.bar(x + (i - 1) * width, values, width, label=policy)  # (i - 1) to skip fcfs position

    ax.set_xlabel('Request Type')
    ax.set_ylabel('Request Service Gain (Policy / FCFS)')
    ax.set_title('Request Service Gain Ratio by Request Type and Policy')
    ax.set_xticks(x + width * (num_policies - 2) / 2)  # (num_policies - 2) to account for skipping fcfs
    ax.set_xticklabels(request_types)
    ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "request_service_gain_ratio.png"))
    plt.close()
    
def plot_request_deadline_meet(data, policies, output_folder):
    request_types = ['latency-sensitive', 'throughput-intensive', 'collective', 'total']
    num_policies = len(policies)
    num_request_types = len(request_types)
    x = np.arange(num_request_types)  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 8))
    for i, policy in enumerate(policies):
        print(policy, data['request_deadline_meet'])
        values = [data['request_deadline_meet'][i][j] for j in range(num_request_types)]
        ax.bar(x + i * width, values, width, label=policy)

    ax.set_xlabel('Request Type')
    ax.set_ylabel('Request Deadline Meet')
    ax.set_title('Request Deadline Meet by Request Type and Policy')
    ax.set_xticks(x + width * (num_policies - 1) / 2)
    ax.set_xticklabels(request_types)
    ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "request_deadline_meet.png"))
    plt.close()
    
# Example usage
folder_path = '../result/poisson-400'
output_folder = '../result/poisson-400'
attributes = [
    "request_deadline_meet", 
    "request_service_gain", "output_throughput",
    "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
    "mean_tbt_ms", "median_tbt_ms", "p99_tbt_ms",
    "mean_request_e2el_ms", "median_request_e2el_ms", "p99_request_e2el_ms"
]

data, policies = load_data_from_folder(folder_path, attributes)
plot_request_deadline_meet(data, policies, output_folder)
plot_request_service_gain_meet(data, policies, output_folder)
plot_data(data, policies, attributes, output_folder)