import asyncio
import numpy as np
import json
import requests
import sys
import time
from pathlib import Path
from typing import List
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import SamplingParams

sys.path.append(str(Path(__file__).resolve().parents[2]))
from benchmarks.trace import (client_simulator, send_request, send_collective_request, RequestInput, 
                              RequestOutput, Trace, RequestFormat, RequestType)

# token/s for latency-sensitive requests
DELIVERY_SPEED = 10
# for other requests we use:
NORMAL_MEAN = 5
NORMAL_VAR = 0.5
# lambda for poisson distribution delivery time
POISSON_LAMBDA = 400
# number of requests
NEW_REQUESTS_NUMBER = 2000
# request ratio
REQUEST_RATIO = [0.772, 0.218, 0.01]
# TODO: use BurstGPT trace
IS_BURST = False

# Example usage
API_URL = "http://localhost:8000/generate"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

test_trace_path = None
test_trace_save_path = None
trace_path = '../dataset/trace/p200-d10-n4000.json'
trace_save_path = None

def adjust_request_type(trace: List[RequestFormat], adapt_ratio):
    assert sum(adapt_ratio) == 1
    num_trace = len(trace)
    num_types = len(adapt_ratio)
    
    # Calculate the number of each type
    type_counts = [int(num_trace * ratio) for ratio in adapt_ratio]
    type_counts[-1] = num_trace - sum(type_counts[:-1])
    
    # Create an array with the specified number of each type
    types = []
    for i, count in enumerate(type_counts):
        types.extend([i] * count)
    np.random.shuffle(types)
    
    for i, result in enumerate(trace):
        if types[i] == 0:
            trace[i].request_type = RequestType.Latency
        elif types[i] == 1:
            trace[i].request_type = RequestType.Throughput
        else:
            trace[i].request_type = RequestType.Collective
        
    return trace
        

def adjust_deadline(results):
    for result in results:
        if result["request_type"] == 0:
            result["deadline"] = max(int(result["output_len"] / DELIVERY_SPEED * 1000), 1000)
        else:
            result["deadline"] = max(int(result["deadline"] * np.random.normal(NORMAL_MEAN, NORMAL_VAR)), 1000)
            
    return results

def adjust_deliver_time(results):
    cumulative_time = 0
    poisson_intervals = np.random.poisson(POISSON_LAMBDA, len(results))
    for i, result in enumerate(results):
        cumulative_time += poisson_intervals[i]
        result["deliver_time"] = int(cumulative_time)
        
    return results
        
def adjust_data_length(results, new_len):
    if len(results) < new_len:
        leftover_len = new_len - len(results)
        new_results = []
        while leftover_len >= len(results):
            new_results.extend(results)
            leftover_len -= len(results)
        new_results.extend(results[:leftover_len])
        results = new_results
    else:
        results = results[:new_len]
    return results
    
def save_trace(results, save_path):
    # Save the trace
    with open(save_path, 'w', encoding='utf-8') as f:
        new_results = []
        for result in results:
            new_results.append({
                "prompt": result["prompt"],
                "output": result["output"],
                "prompt_len": result["prompt_len"],
                "output_len": result["output_len"],
                "collection_id": result["collection_id"],
                "request_type": result["request_type"],
                "deliver_time": result["deliver_time"],
                "deadline": result["deadline"],
                "priority": result["priority"]
            })
        json.dump(new_results, f, indent=4, ensure_ascii=False)

async def call_vllm_api(trace: List[RequestFormat]):
    results = []
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    sampling_params = SamplingParams(
        n=1,              
        temperature=0.01,
        top_p=1.0,
        max_tokens=1024,
        logprobs=None,
        best_of=2,
        ignore_eos=False,
    )
    
    for i, request in enumerate(tqdm(trace, desc="Processing requests")):  
        try:
            test_input = RequestInput(
                request=request,
                sampling_params=sampling_params,
                client_id=0,
                api_url=API_URL,
            )
            prompt = test_input.request.prompt
            start_time = time.time()

            # Make the API call
            if request.request_type == RequestType.Collective:
                response: RequestOutput = await send_collective_request(test_input, client_deadline=1000)
            else:
                response: RequestOutput = await send_request(test_input, client_deadline=1000)

            # Measure end time
            end_time = time.time()

            # Calculate latency
            latency = int((end_time - start_time) * 1000)

            # Store the result
            results.append({
                "prompt": prompt,
                "output": response.request_output,
                "prompt_len": request.prompt_len,
                "output_len": 0,
                "collection_id": int(request.collection_id),
                "request_type": request.request_type.value,
                "deliver_time": int(request.deliver_time),
                "deadline": int(latency),
                "priority": 0
            })
        except Exception as e:
            print(f"Error index: {i}")
            print(e)

    for i in range(len(results)):
        for output in results[i]["output"]:
            results[i]["output_len"] += len(tokenizer(output).input_ids)
    return results

async def main(trace, trace_save_path=None):
    with open(trace_path, 'r', encoding='utf-8') as f:
        trace_list = json.load(f)
    #trace = []
    #for req in trace_list:
    #    trace.append(RequestFormat.from_dict(req))
    
    #trace = adjust_request_type(trace, REQUEST_RATIO)
    #print("Finish adjusting request type")

    #results = await call_vllm_api(trace)
    #print("Finish calling VLLM API")
    results = trace_list
    #results = adjust_deadline(results)
    #print("Finish Adjusting deadlines")
    results = adjust_deliver_time(results)
    print("Finish Adjusting deliver times")
    results = adjust_data_length(results, NEW_REQUESTS_NUMBER)
    print("Finish Adjusting data length")
    
    if trace_save_path is None:
        model_name = MODEL_NAME.split("/")[-1]
        # trace_save_path = f"{model_name}-p{POISSON_LAMBDA}-d{DELIVERY_SPEED}-n{NEW_REQUESTS_NUMBER}.json"
        trace_save_path = f"../dataset/trace/p{POISSON_LAMBDA}-d{DELIVERY_SPEED}-n{NEW_REQUESTS_NUMBER}.json"
    print(f"Saving trace to {trace_save_path}")
    save_trace(results, trace_save_path)

if __name__ == "__main__":

    asyncio.run(main(trace_path, trace_save_path))