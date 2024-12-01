import json
import numpy as np
import matplotlib.pyplot as plt

def plot_request_frequency(deliver_times, duration_time):
    num_bins = 100
    bin_edges = np.linspace(0, duration_time, num_bins + 1)
    hist, _ = np.histogram(deliver_times, bins=bin_edges)

    plt.figure(figsize=(10, 6))
    plt.hist(deliver_times, bins=bin_edges, edgecolor='black')
    plt.title('Request Frequency Distribution')
    plt.xlabel('Duration (ms)')
    plt.ylabel('Number of Tasks')
    plt.grid(True)
    plt.savefig('request_frequency_distribution.png')

def convert_numpy_int64_to_int(data):
    if isinstance(data, list):
        return [convert_numpy_int64_to_int(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_numpy_int64_to_int(value) for key, value in data.items()}
    elif isinstance(data, np.int64):
        return int(data)
    else:
        return data

def scale_deliver_times(data, new_duration, duration_time):
    scale_factor = new_duration / duration_time
    for item in data:
        item['deliver_time'] = int(item['deliver_time'] * scale_factor)
    return data

def set_poisson_deliver_times(data, lam):
    poisson_intervals = np.random.poisson(lam, len(data))
    cumulative_time = 0
    for i, item in enumerate(data):
        cumulative_time += poisson_intervals[i]
        item['deliver_time'] = cumulative_time
        item['deadline'] = max(int(item['deadline'] * np.random.normal(1, 0.5) * 3), 0)
    return data

def set_request_rate_deliver_times(data, rate):
    interval = 1000 / rate
    cumulative_time = 0
    for item in data:
        cumulative_time += interval
        item['deliver_time'] = cumulative_time
    return data

def get_request_rate(deliver_times):
    return 1000 / (deliver_times[-1] / len(deliver_times))
    
if __name__ == "__main__":
    new_duration = None  #(ms)
    lambda_param = 1200
    request_rate = 10       #(req/s)
    
    input_file = '../example-long-1.json'
    output_file = '../scaled_poisson-1200_example-3.json'
    # plot_path = '../BurstGPT_request_frequency_distribution.png'
    
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
    duration_time = data[-1]['deliver_time']
    if new_duration is not None:
        scaled_data = scale_deliver_times(data, new_duration, duration_time)
    elif lambda_param is not None:
        poisson_data = set_poisson_deliver_times(data, lambda_param)
        scaled_data = convert_numpy_int64_to_int(poisson_data)
    elif request_rate is not None:
        scaled_data = set_request_rate_deliver_times(data, request_rate)
        scaled_data = convert_numpy_int64_to_int(scaled_data)
    
    new_request_rate = get_request_rate([item['deliver_time'] for item in scaled_data])
    new_duration = scaled_data[-1]['deliver_time']
    print(f'New request rate: {new_request_rate} req/s')
    # plot_request_frequency([item['deliver_time'] for item in scaled_data], new_duration)
    
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(scaled_data, file, ensure_ascii=False, indent=4)
        