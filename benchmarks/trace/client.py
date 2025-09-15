import aiohttp
import asyncio
import json
import numpy as np
import random
import pandas as pd
import sys
import time
import traceback
from copy import deepcopy
from tqdm import tqdm
from typing import List, Tuple, AsyncGenerator, Optional
from vllm import SamplingParams
from vllm.request_info import RequestInfo, RequestType

class RequestInput:
    def __init__(
        self, 
        request: RequestInfo,
        slo_constraint: Tuple[float, float, float],
        sampling_params: SamplingParams,
        client_id: int,
        api_url: str,
        num_stages: int = 1,
        ):
        self.request = request
        self.slo_constraint = slo_constraint
        self.sampling_params = sampling_params
        self.client_id = client_id
        self.api_url = api_url
        self.num_stages = num_stages

class RequestOutput:
    def __init__(self):
        self.finish_before_ddl = False
        self.request_type = -1
        self.request_input = ''
        self.request_output = ''
        self.request_ttft = 0
        self.request_tbt = []
        self.request_service_gain = 0
        self.request_latency = 0
        self.request_finish_time = 0
        self.collection_id = -1
        # Task level metrics
        self.task_latency = 0
        # Debug porpeties
        self.success = False
        self.error = ""

class TaskOutput:
    # Task level metric
    def __init__(self):
        self.task_latency = 0
        self.task_output = []

def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def llama_prompt_format(prompt: str) -> str:
    return f"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

    {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

def qwen_prompt_format(prompt: str) -> str:
    return f"""
    <|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n
    <|im_start|>user
    {prompt}<|im_end|>\n\n<|im_start|>assistant\n
    """

def get_prompt_format_func(model_name: str) -> callable:
    '''
    Get the prompt format function based on the model name.
    '''
    if 'llama' in model_name.lower():
        return llama_prompt_format
    elif 'qwen' in model_name.lower():
        return qwen_prompt_format
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

def parse_output(output: str, input_length: int) -> str:
    '''
    Parse the output from the model.
    '''
    return output[input_length:]


async def get_request(
    input_requests: List[RequestInfo],
    poisson_lambda: float = 1000.0,
    burst: bool = False,
) -> AsyncGenerator[RequestInfo, None]:
    request_num = len(input_requests)
    input_requests = iter(input_requests)
    
    # if burst using the burst pattern
    if burst:
        print("Using BurstGPT pattern.")
        df = pd.read_csv('/home/jovyan/workspace/Concord/benchmarks/trace/BurstGPT_1.csv')
        timestamps = df['Timestamp'].tolist()
        baseline_timestamp = timestamps[99]
        timestamps = [ts - baseline_timestamp for ts in timestamps[100:100 + request_num]]
        original_req_rate = request_num * 1000 / timestamps[request_num - 1]
        target_req_rate = 1 / poisson_lambda * 1000
        timestamps = [int(timestamp * original_req_rate / target_req_rate) for timestamp in timestamps]
        last_timestamp = 0

        for request, new_timestamp in zip(input_requests, timestamps):
            interval = (new_timestamp - last_timestamp) / 1000
            last_timestamp = new_timestamp
            await asyncio.sleep(interval)
            yield request
    else:
        for request in input_requests:
            interval = np.random.poisson(poisson_lambda) / 1000
            await asyncio.sleep(interval)        
            yield request


async def send_collective_request(
    request_info: RequestInput, 
    tot_structure: Tuple[int, int],
    penalty_factor: int,
    model_name: str,
    client_deadline: float = 20,
    pbar: Optional[tqdm] = None,
    is_stream: bool = True,
    ) -> List[RequestOutput]:
    '''
    Send collective requests to the model and return the responses.
    '''
    api_url = request_info.api_url
    timeout = aiohttp.ClientTimeout(total=client_deadline)
    prompt_format = get_prompt_format_func(model_name)
    output_list: List[RequestOutput] = []
    
    # ToT parameters
    num_thoughts, tot_round = tot_structure
    n = num_thoughts
    best_of = num_thoughts
    total_num_request = (n + 1) + (n * best_of + 1) * (tot_round - 1)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        payload = {
            "request_info": request_info.request.to_dict(),
            "slo_constraint": request_info.slo_constraint,
            "sampling_params": request_info.sampling_params.to_dict(),
            "client_id": request_info.client_id,
            "stream": is_stream,
            "num_stages": 4,
            "target_output_length": request_info.request.output_len,  # Send output length to server
        }
        st = time.perf_counter()
        
        async def generate_thoughts(session, api_url, thought, payload) -> RequestOutput:
            new_payload = deepcopy(payload)
            
            input_prompt = prompt_format(thought)
            input_length = len(input_prompt)
            new_payload['request_info']['prompt'] = input_prompt
            new_payload["sampling_params"]["n"] = 1
            new_payload["sampling_params"]["best_of"] = 1
            new_payload["sampling_params"]["max_tokens"] = 256
            # Use the output_len from request as target for stopping
            new_payload["target_output_length"] = request_info.request.output_len
            
            output = RequestOutput()
            output.request_input = input_prompt
            output.collection_id = request_info.request.collection_id
            output.request_type = request_info.request.request_type.value

            st = time.perf_counter()
            try:
                async with session.post(url=api_url, json=new_payload) as response:
                    if response.status == 200:
                        choice = ''
                        buffer = ""
                        async for chunk_bytes in response.content.iter_any():
                            chunk_bytes = chunk_bytes.strip()
                            if not chunk_bytes:
                                continue
                            chunk = chunk_bytes.decode("utf-8")
                            buffer += chunk
                        
                        if is_stream:
                            while "\0" in buffer:
                                part, buffer = buffer.split("\0", 1)
                                data = json.loads(part)
                                choice = data["text"][0]
                                output.request_tbt = data["tbt"]
                                output.request_service_gain = data["service_gain"]
                                if output.request_ttft == 0.0 and data["ttft"] is not None:
                                    output.request_ttft = data["ttft"]
                        else:
                            data = json.loads(buffer)
                            choice = data["text"][0]
                            output.request_tbt = data["tbt"]
                            output.request_service_gain = data["service_gain"]
                            if output.request_ttft == 0.0 and data["ttft"] is not None:
                                output.request_ttft = data["ttft"]
                                
                        output.request_output = parse_output(choice, input_length)
                        output.request_latency = time.perf_counter() - st
                    else:
                        output.success = False
                        output.error = f"Failed response, status: {response.status}"
                        return output
            except Exception:
                exc_info = sys.exc_info()
                output.success = False
                output.error = "".join(traceback.format_exception(*exc_info))
                return output
            
            output.success = True
            output.request_finish_time = time.perf_counter()
            return output
        
        async def value_thoughts(session, api_url, choices, payload) -> RequestOutput:
            # construct the promote for the value model
            value_prompt = """
            Please evaluate the following thoughts on a scale from 1 to 10, where 10 represents the highest value (best) and 1 represents the lowest value. Use the following criteria for scoring:
            - Clarity of expression and coherence
            - Depth of insight or originality
            - Relevance to the context or topic
            - Impact or significance of the thought

            Assign scores accordingly without providing explanations.
            """
            for i, choice in enumerate(choices):
                value_prompt += f"Choice {i+1}: {choice}\n"     
            value_prompt = prompt_format(value_prompt)
            
            # Adpat the payload for the value process
            new_payload = deepcopy(payload)
            new_payload["request_info"]["prompt"] = value_prompt
            new_payload["sampling_params"]["n"] = 1
            new_payload["sampling_params"]["best_of"] = 1
            new_payload["sampling_params"]["max_tokens"] = 256
            # Use the output_len from request as target for stopping
            new_payload["target_output_length"] = request_info.request.output_len
            input_length = len(new_payload['request_info']['prompt'])
            
            output = RequestOutput()
            output.request_input = value_prompt
            output.collection_id = request_info.request.collection_id
            output.request_type = request_info.request.request_type.value
            try:
                async with session.post(url=api_url, json=new_payload) as response:
                    if response.status == 200:
                        value = ''
                        buffer = ''
                        async for chunk_bytes in response.content.iter_any():
                            chunk_bytes = chunk_bytes.strip()
                            if not chunk_bytes:
                                continue
                            chunk = chunk_bytes.decode("utf-8")
                            buffer += chunk

                        if is_stream:
                            while "\0" in buffer:
                                part, buffer = buffer.split("\0", 1)
                                data = json.loads(part)
                                value = data["text"][0]
                                output.request_tbt = data["tbt"]
                                output.request_service_gain = data["service_gain"]
                                if output.request_ttft == 0.0 and data["ttft"] is not None:
                                    output.request_ttft = data["ttft"]
                        else:
                            data = json.loads(buffer)
                            value = data["text"][0]
                            output.request_tbt = data["tbt"]
                            output.request_service_gain = data["service_gain"]
                            if output.request_ttft == 0.0 and data["ttft"] is not None:
                                output.request_ttft = data["ttft"]

                        output.request_output = parse_output(value, input_length)
                        output.request_latency = time.perf_counter() - st 
                    else:
                        output.success = False
                        output.error = f"Failed response, status: {response.status}"
                        return output
            except Exception:
                exc_info = sys.exc_info()
                output.success = False
                output.error = "".join(traceback.format_exception(*exc_info))
                return output     
            
            output.success = True
            output.request_finish_time = time.perf_counter()
            return output
            
        try:
            # using ToT reuqest pattern for collective requests
            thoughts = [request_info.request.prompt]
            for round in range(tot_round):
                # generate thoughts
                tasks = []
                choices = []
                for thought in thoughts:
                    if round > 0:
                        thought += "\nPlease review and revise the response."
                    for _ in range(num_thoughts):
                        tasks.append(generate_thoughts(session, api_url, thought, payload))
                
                results: List[RequestOutput] = await asyncio.gather(*tasks)  
                for generate_output in results:
                    if not generate_output.success:
                        print("Generate thoughts failed")
                        print(generate_output.error)
                        return []
                    else:
                        choices.append(generate_output.request_output)
                        output_list.append(generate_output)
                # value thoughts and select the top-k
                value_output: RequestOutput = await value_thoughts(session, api_url, choices, payload)
                if not value_output.success:
                    print("Value thoughts failed")
                    print(value_output.error)
                    return []
                output_list.append(value_output)
                
                # sort the choices based on the value
                value = [random.random() for _ in range(len(choices))]
                choices = [x for _, x in sorted(zip(value, choices), reverse=True)]
                thoughts = choices[:num_thoughts]
            
            ed = time.perf_counter()
            task_latency = ed - st
            total_deadline = request_info.request.deadline / 1000 * 4
            if task_latency > total_deadline:
                slo_violation_penalty = (total_deadline) / task_latency
                slo_violation_penalty = max(1e-6, min(1.0, slo_violation_penalty))**penalty_factor
                for output in output_list:
                    output.finish_before_ddl = False
                    output.request_service_gain *= slo_violation_penalty
                    output.task_latency = task_latency
            else:
                for output in output_list:
                    output.finish_before_ddl = True
                    output.task_latency = task_latency
        except Exception:
            exc_info = sys.exc_info()
            error = "".join(traceback.format_exception(*exc_info))
            print(error)
            return []
    
    if pbar is not None:
        pbar.update(1)
    return output_list


async def send_request(
    request_info: RequestInput, 
    model_name: str,
    client_deadline: float = 20,
    pbar: Optional[tqdm] = None,
    is_stream: bool = True,
    ) -> List[RequestOutput]:
    '''
    Send requests to the model and return the responses.
    '''
    api_url = request_info.api_url
    prompt_format = get_prompt_format_func(model_name)
    request_prompt = prompt_format(request_info.request.prompt)
    request_info.request.prompt = request_prompt
    input_length = len(request_prompt)
    
    output = RequestOutput()
    output.request_type = request_info.request.request_type.value
    output.collection_id = request_info.request.collection_id
    output.request_input = request_info.request.prompt
    
    timeout = aiohttp.ClientTimeout(total=client_deadline)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        payload = {
            "request_info": request_info.request.to_dict(),
            "slo_constraint": request_info.slo_constraint,
            "sampling_params": request_info.sampling_params.to_dict(),
            "client_id": request_info.client_id,
            "stream": is_stream,
            "target_output_length": request_info.request.output_len,  # Send output length to server
        }
        st = time.perf_counter()
        
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    buffer = ""
                    async for chunk_bytes in response.content.iter_any():
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        
                        chunk = chunk_bytes.decode("utf-8")
                        buffer += chunk

                    generated_text = ""
                    if is_stream:
                        while "\0" in buffer:
                            part, buffer = buffer.split("\0", 1)
                            data = json.loads(part)
                            output.request_tbt = data["tbt"]
                            output.request_service_gain = data["service_gain"]
                            generated_text = data["text"][0]
                            if output.request_ttft == 0.0 and data["ttft"] is not None:
                                output.request_ttft = data["ttft"]
                    else:
                        data = json.loads(buffer)
                        output.request_tbt = data["tbt"]
                        output.request_service_gain = data["service_gain"]
                        generated_text = data["text"][0]
                        if output.request_ttft == 0.0 and data["ttft"] is not None:
                            output.request_ttft = data["ttft"]

                    output.request_output = parse_output(generated_text, input_length)
                    output.request_latency = time.perf_counter() - st
                    if output.request_latency <= request_info.request.deadline / 1000:
                        output.finish_before_ddl = True
                    else:
                        output.finish_before_ddl = False
                else:
                    output.success = False
                    output.error = response.reason or ""
                    return [output]
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
            return [output]

    if pbar is not None:
        pbar.update(1)
                    
    output.success = True
    output.request_finish_time = time.perf_counter()
    return [output]


async def client_simulator(
    input_requests: List[RequestInfo], 
    slo_constraint: Tuple[float, float, float],
    penalty_factor: int,
    poisson_lambda: float,
    burst: bool,
    sampling_params: SamplingParams,     
    client_id: int,
    client_deadline: int,
    api_url: str,
    tot_structure: Tuple[int, int],
    model_name: str,
    is_stream: bool = True,
    pbar: Optional[tqdm] = None,
    ) -> Tuple[List[RequestOutput], List[TaskOutput]]:
    '''
    Execute the client to send requests to the model.
    '''
    start_time = time.perf_counter()
    tasks = []
    tasks: List[asyncio.Task] = []
    if len(input_requests) == 0:
        return []
    async for request in get_request(input_requests, poisson_lambda, burst):      
        request_info = RequestInput(request, slo_constraint, sampling_params, client_id, api_url, tot_structure[1]*2)

        if request.request_type == RequestType.COLLECTIVE:
            tasks.append(asyncio.create_task(send_collective_request(request_info, tot_structure, penalty_factor, model_name, client_deadline, pbar, is_stream)))
        else:
            tasks.append(asyncio.create_task(send_request(request_info, model_name, client_deadline, pbar, is_stream)))
        
    outputs: List[List[RequestOutput]] = await asyncio.gather(*tasks)
    finish_time = []
    tasks: List[TaskOutput] = []
    for output_list in outputs:
        if len(output_list) == 0:
            continue
        output_type = output_list[0].request_type
        output_success = output_list[0].success
        if output_success is False:
            continue
        if output_type == 2:
            task_output = TaskOutput()
            task_output.task_latency = output_list[0].task_latency
            for output in output_list:
                task_output.task_output.append(output.request_output)
            tasks.append(task_output)
        for output in output_list:
            finish_time.append(output.request_finish_time)
            output.request_finish_time -= start_time

    finish_time = sorted(finish_time)
    # start from a new line
    print()

    # collect throughput for every n requests
    n = 100
    for i in range(n, len(finish_time) + 1, n):
        t = finish_time[i - 1] - finish_time[i - n]
        if t > 0:
            throughput = n / t
            print(f"Throughput for requests {i-n+1} to {i}: {throughput:.2f} req/s")
    
    return outputs, tasks