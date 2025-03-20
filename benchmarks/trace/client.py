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
from .trace import RequestFormat, RequestType

class RequestInput:
    def __init__(
        self, 
        request: RequestFormat,
        slo_constraint: Tuple[float, float, float],
        sampling_params: SamplingParams,
        client_id: int,
        api_url: str
        ):
        self.request = request
        self.slo_constraint = slo_constraint
        self.sampling_params = sampling_params
        self.client_id = client_id
        self.api_url = api_url
    

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
        # Debug porpeties
        self.success = False
        self.error = ""


def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def prompt_format(prompt: str) -> str:
    return f"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

    {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """


def parse_output(output: str, input_length: int) -> str:
    '''
    Parse the output from the model.
    '''
    return output[input_length:]


async def get_request(
    input_requests: List[RequestFormat],
    poisson_lambda: float = 1000.0,
    burst: bool = False,
) -> AsyncGenerator[RequestFormat, None]:
    request_num = len(input_requests)
    input_requests = iter(input_requests)
    
    # if burst using the burst pattern
    if burst:
        df = pd.read_csv('trace/BurstGPT_1.csv')
        timestamps = df['Timestamp'].tolist()
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
    client_deadline: float = 20,
    pbar: Optional[tqdm] = None,
    is_stream: bool = True,
    ) -> List[RequestOutput]:
    '''
    Send collective requests to the model and return the responses.
    '''
    api_url = request_info.api_url
    timeout = aiohttp.ClientTimeout(total=client_deadline)
    output_list: List[RequestOutput] = []
    
    # ToT parameters
    num_thoughts, tot_round = tot_structure
    n = num_thoughts
    best_of = num_thoughts
    total_num_request = (n + 1) + (n * best_of + 1) * (tot_round - 1)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        payload = {
            "request_info": request_info.request.to_dict(),
            "sampling_params": request_info.sampling_params.to_dict(),
            "client_id": request_info.client_id,
            "stream": is_stream,
        }
        st = time.perf_counter()
        
        async def generate_thoughts(session, api_url, thought, payload) -> RequestOutput:
            new_payload = deepcopy(payload)
            
            input_prompt = prompt_format(thought)
            input_length = len(input_prompt)
            new_payload['request_info']['prompt'] = input_prompt
            new_payload["sampling_params"]["n"] = 1
            new_payload["sampling_params"]["best_of"] = 1
            
            output = RequestOutput()
            output.request_input = input_prompt
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
            input_length = len(new_payload['request_info']['prompt'])
            
            output = RequestOutput()
            output.request_input = value_prompt
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
            if task_latency > request_info.request.deadline / 1000:
                slo_violation_penalty = min(1, ((request_info.request.deadline / 1000) / task_latency)**2)
                for output in output_list:
                    output.finish_before_ddl = False
                    output.request_service_gain *= slo_violation_penalty
            else:
                for output in output_list:
                    output.finish_before_ddl = True
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
    client_deadline: float = 20,
    pbar: Optional[tqdm] = None,
    is_stream: bool = True,
    ) -> List[RequestOutput]:
    '''
    Send requests to the model and return the responses.
    '''
    api_url = request_info.api_url
    request_prompt = prompt_format(request_info.request.prompt)
    request_info.request.prompt = request_prompt
    input_length = len(request_prompt)
    
    output = RequestOutput()
    output.request_type = request_info.request.request_type.value
    output.request_input = request_info.request.prompt
    
    timeout = aiohttp.ClientTimeout(total=client_deadline)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        payload = {
            "request_info": request_info.request.to_dict(),
            "sampling_params": request_info.sampling_params.to_dict(),
            "client_id": request_info.client_id,
            "stream": is_stream
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
    input_requests: List[RequestFormat], 
    slo_constraint: Tuple[float, float, float],
    poisson_lambda: float,
    burst: bool,
    sampling_params: SamplingParams,     
    client_id: int,
    client_deadline: int,
    api_url: str,
    tot_structure: Tuple[int, int],
    pbar: Optional[tqdm] = None,
    ) -> List[RequestOutput]:
    '''
    Execute the client to send requests to the model.
    '''
    start_time = time.perf_counter()
    tasks = []
    tasks: List[asyncio.Task] = []
    if len(input_requests) == 0:
        return []
    async for request in get_request(input_requests, poisson_lambda, burst):      
        request_info = RequestInput(request, slo_constraint, sampling_params, client_id, api_url)
        deadline = client_deadline

        if request.request_type == RequestType.Collective:
            tasks.append(asyncio.create_task(send_collective_request(request_info, tot_structure, deadline, pbar)))
        else:
            tasks.append(asyncio.create_task(send_request(request_info, deadline, pbar)))
        
    outputs: List[List[RequestOutput]] = await asyncio.gather(*tasks)
    for output_list in outputs:
        for output in output_list:
            output.request_finish_time -= start_time
    
    return outputs