import requests
import time
import json
import sys
import asyncio
from pathlib import Path
from vllm import SamplingParams
from typing import List
from transformers import AutoTokenizer
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parents[1]))
from benchmarks.trace import (client_simulator, send_request, send_collective_request, RequestInput, 
                              RequestOutput, Trace, RequestFormat, RequestType)


async def call_vllm_api(trace: List[RequestFormat], model_name, api_url):
    results = []
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
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
                api_url=api_url,
            )
            prompt = test_input.request.prompt
            start_time = time.time()

            # Make the API call
            if request.request_type == RequestType.Collective:
                response = await send_collective_request(test_input, client_deadline=1000)
            else:
                response = await send_request(test_input, client_deadline=1000)

            # Measure end time
            end_time = time.time()

            # Calculate latency
            latency = int((end_time - start_time) * 1000)

            # Extract generated text and token length from the response
            response_data = response.generated_text

            # Store the result
            results.append({
                "prompt": prompt,
                "output": response.generated_text,
                "prompt_len": request.prompt_len,
                "output_len": tokenizer.encode(response.generated_text, return_tensors='pt').shape[1],
                "collection_id": request.collection_id,
                "request_type": request.request_type.value,
                "deliver_time": request.deliver_time,
                "deadline": latency,
                "priority": 0
            })
        except Exception as e:
            print(f"Error index: {i}")

    return results

# Example usage
trace_path = 'example.json'
trace_save_path = 'example'
model_name = "meta-llama/Llama-3.1-8B-Instruct"
api_url = "http://localhost:8000/generate"

trace = Trace.load_trace(trace_path)

results = asyncio.run(call_vllm_api(trace, model_name, api_url))

# Save the trace
for times in [0.5, 1, 2, 4]:
    s_trace_path = f'{trace_save_path}-{times}.json'
    with open(s_trace_path, 'w', encoding='utf-8') as f:
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
                "deadline": int(result["deadline"] * times),
                "priority": result["priority"]
            })
        json.dump(new_results, f, indent=4, ensure_ascii=False)