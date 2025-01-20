import aiohttp
import asyncio
import json
import random
import sys
import time
import traceback
from copy import deepcopy
from tqdm import tqdm
from typing import List, Tuple, AsyncGenerator, Optional
from vllm import SamplingParams
from .trace import  RequestFormat, RequestType

# Time per token
DELIVERY_SPEED = 5 

class RequestInput:
    def __init__(
        self, 
        request: RequestFormat,
        sampling_params: SamplingParams,
        client_id: int,
        api_url: str
        ):
        self.request = request
        self.sampling_params = sampling_params
        self.client_id = client_id
        self.api_url = api_url
    

class RequestOutput:
    def __init__(self):
        self.type = -1
        self.task_input = ''
        self.task_output = ''
        self.request_input = []
        self.request_output = []
        self.success = False
        self.finish_before_ddl = False
        self.error = ""
        self.task_ttft = 0
        self.task_latency = 0
        self.request_ttft = []
        self.request_tbt = []
        self.request_service_gain = 0
        self.request_latency = []
        self.collection_num_requests = 1
        # for throughput
        self.success_num_requests = 0


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
) -> AsyncGenerator[RequestFormat, None]:
    last_request_deliver_time = input_requests[0].deliver_time
    input_requests = iter(input_requests)

    for request in input_requests:
        interval = (request.deliver_time - last_request_deliver_time) / 1000
        last_request_deliver_time = request.deliver_time
        await asyncio.sleep(interval)        
        yield request


async def send_collective_request(
    request_info: RequestInput, 
    client_deadline: float = 20,
    pbar: Optional[tqdm] = None,
    is_stream: bool = True,
    ) -> RequestOutput:
    '''
    Send collective requests to the model and return the responses.
    '''
    api_url = request_info.api_url
    timeout = aiohttp.ClientTimeout(total=client_deadline)
    output = RequestOutput()
    output.type = request_info.request.request_type.value
    output.task_input = request_info.request.prompt
    
    # ToT parameters
    num_thoughts = 3
    tot_round = 2
    n = num_thoughts
    best_of = num_thoughts
    success_num_request = (n + 1) + (n * best_of + 1) * (tot_round - 1)
    output.collection_num_requests = success_num_request
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        payload = {
            "request_info": request_info.request.to_dict(),
            "sampling_params": request_info.sampling_params.to_dict(),
            "client_id": request_info.client_id,
            "stream": is_stream,
        }
        ttft = 0.0
        num_requests = 0
        st = time.perf_counter()
        
        async def generate_thoughts(session, api_url, thought, payload) -> Tuple[bool, str, str, float, List[float],
                                                                              float, float]:
            new_payload = deepcopy(payload)
            
            input_prompt = prompt_format(thought)
            input_length = len(input_prompt)
            new_payload['request_info']['prompt'] = input_prompt
            new_payload["sampling_params"]["n"] = 1
            new_payload["sampling_params"]["best_of"] = 1
            # new_payload["request_info"]["deadline"] = new_payload["request_info"]["deadline"] / 8 * 3
            
            ttft = 0.0
            tbt = []
            service_gain = 0
            latency = 0.0
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

                        latency = time.perf_counter() - st
                        
                        if is_stream:
                            while "\0" in buffer:
                                part, buffer = buffer.split("\0", 1)
                                data = json.loads(part)
                                choice = data["text"][0]
                                tbt = data["tbt"]
                                service_gain = data["service_gain"]

                                if ttft == 0.0 and data["ttft"] is not None:
                                    ttft = data["ttft"]
                        else:
                            data = json.loads(buffer)
                            choice = data["text"][0]
                            tbt = data["tbt"]
                            service_gain = data["service_gain"]
                            if ttft == 0.0 and data["ttft"] is not None:
                                ttft = data["ttft"]
                                
                        choice = parse_output(choice, input_length)
                    else:
                        return (False, '', '', ttft, tbt, service_gain, 0)
            except Exception:
                exc_info = sys.exc_info()
                error = "".join(traceback.format_exception(*exc_info))
                return (False, '', '', ttft, tbt, service_gain, latency)

            return (True, input_prompt, choice, ttft, tbt, service_gain, latency)
        
        async def value_thoughts(session, api_url, choices, payload) -> Tuple[bool, str, str, float, List[float],
                                                                              float, float]:
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
            # new_payload["request_info"]["deadline"] /= 8
            
            st = time.perf_counter()
            ttft = 0.0
            tbt = []
            service_gain = 0
            latency = 0.0
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
                                tbt = data["tbt"]
                                service_gain = data["service_gain"]

                                if ttft == 0.0 and data["ttft"] is not None:
                                    ttft = data["ttft"]
                        else:
                            data = json.loads(buffer)
                            value = data["text"][0]
                            if ttft == 0.0 and data["ttft"] is not None:
                                ttft = data["ttft"]
                            tbt = data["tbt"]
                            service_gain = data["service_gain"]

                        latency = time.perf_counter() - st    
                        # To ensure that the simulation results remain unaffected by the behavior of the value model,
                        # which may sometimes fail to generate values (e.g., unexpected response formats),
                        # we use randomly generated values as a fallback mechanism.
                        value = parse_output(value, input_length)
                        # value = [random.random() for _ in range(len(choices))]
                    else:
                        return (False, '', [], ttft, tbt, service_gain, latency)
            except Exception:
                exc_info = sys.exc_info()
                error = "".join(traceback.format_exception(*exc_info)) 
                return (False, '', [], 0, [], 0, 0)     

            return (True, value_prompt, value, ttft, tbt, service_gain, latency)
            
        try:
            # using ToT reuqest pattern for collective requests
            thoughts = [request_info.request.prompt]
            output.success = True
            for round in range(tot_round):
                # generate thoughts
                tasks = []
                choices = []
                for thought in thoughts:
                    if round > 0:
                        thought += "\nPlease review and revise the response."
                    for _ in range(num_thoughts):
                        tasks.append(generate_thoughts(session, api_url, thought, payload))
                
                results = await asyncio.gather(*tasks)  
                for generate_success, generate_input, generate_choice, \
                    generate_ttft, generate_tbt, generate_service, generate_latency in results:
                    if not generate_success:
                        output.error = "Generate thoughts failed"
                        output.success = False
                        break
                    else:
                        choices.append(generate_choice) 
                        if ttft == 0.0:
                            ttft = generate_ttft
                            output.task_ttft = ttft
                        output.request_ttft.append(generate_ttft)
                        output.request_latency.append(generate_latency) 
                        output.request_tbt.extend(generate_tbt)
                        output.request_service_gain += generate_service
                        output.request_input.append(generate_input)
                        output.request_output.append(generate_choice)
                        num_requests += 1
                if not output.success:
                    break
                # value thoughts and select the top-k
                value_success, value_input, value, \
                    value_ttft, value_tbt, value_servie, value_latency = \
                        await value_thoughts(session, api_url, choices, payload)
                if not value_success:
                    output.error = "Value thoughts failed"
                    output.success = False
                    break
                output.request_ttft.append(value_ttft)
                output.request_latency.append(value_latency)
                output.request_tbt.extend(value_tbt)
                output.request_service_gain += value_servie
                output.request_input.append(value_input)
                output.request_output.append(value)
                num_requests += 1
                
                # sort the choices based on the value
                value = [random.random() for _ in range(len(choices))]
                choices = [x for _, x in sorted(zip(value, choices), reverse=True)]
                thoughts = choices[:num_thoughts]
            
            output.task_latency = time.perf_counter() - st
            output.task_output = thoughts[0]
            output.success_num_requests = num_requests
            if num_requests == success_num_request:
                output.success = True
                if output.task_latency <= request_info.request.deadline / 1000:
                    output.finish_before_ddl = True
                else:
                    output.request_service_gain *= (request_info.request.deadline / 1000) / output.task_latency
                    output.finish_before_ddl = False
            else:
                output.success = False
        except Exception:
            output.success = False
            
            # To ensure that the simulation results remain unaffected by Excpetion
            # And possible success API calls can still be used for the profiling
            output.task_latency = time.perf_counter() - st
            output.success_num_requests = num_requests
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
    
    if pbar is not None:
        pbar.update(1)
    return output


async def send_request(
    request_info: RequestInput, 
    client_deadline: float = 20,
    pbar: Optional[tqdm] = None,
    is_stream: bool = True,
    ) -> RequestOutput:
    '''
    Send requests to the model and return the responses.
    '''
    api_url = request_info.api_url
    # timeout = aiohttp.ClientTimeout(total=client_deadline, sock_read=300)
    output = RequestOutput()
    output.type = request_info.request.request_type.value
    output.task_input = request_info.request.prompt  
    request_prompt = prompt_format(request_info.request.prompt)
    output.request_input.append(request_prompt)
    request_info.request.prompt = request_prompt
    input_length = len(request_prompt)
    
    timeout = aiohttp.ClientTimeout(total=client_deadline)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        payload = {
            "request_info": request_info.request.to_dict(),
            "sampling_params": request_info.sampling_params.to_dict(),
            "client_id": request_info.client_id,
            "stream": is_stream
        }
        # output.prompt_len = request_info.prompt_len
        
        ttft = 0.0
        tbt = []
        service_gain = 0
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
                                        
                    latency = time.perf_counter() - st
                    output.task_latency = latency
                    output.request_latency.append(latency)
                    output.success = True
                    
                    generated_text = ""
                    if is_stream:
                        while "\0" in buffer:
                            part, buffer = buffer.split("\0", 1)
                            data = json.loads(part)
                            # First token
                            if ttft == 0.0 and data["ttft"] is not None:
                                ttft = data["ttft"]
                                output.task_ttft = ttft
                                output.request_ttft.append(ttft)
                            output.request_tbt = data["tbt"]
                            output.request_service_gain = data["service_gain"]
                            generated_text = data["text"][0]
                    else:
                        data = json.loads(buffer)
                        if ttft == 0.0 and data["ttft"] is not None:
                            ttft = data["ttft"]
                            output.task_ttft = ttft
                            output.request_ttft.append(ttft)
                        output.request_tbt = data["tbt"]
                        output.request_service_gain = data["service_gain"]
                        generated_text = data["text"][0]
                        
                    output_text = parse_output(generated_text, input_length)
                    output.task_output = output_text
                    output.request_output.append(output_text)
                    
                    if output.task_latency <= request_info.request.deadline / 1000:
                        output.finish_before_ddl = True
                    else:
                        output.request_service_gain *= (request_info.request.deadline / 1000) / output.task_latency
                        output.finish_before_ddl = False
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar is not None:
        pbar.update(1)
        
    return output


async def client_simulator(
    input_requests: List[RequestFormat], 
    sampling_params: SamplingParams,     
    client_id: int,
    client_deadline: int,
    api_url: str,
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
    async for request in get_request(input_requests):      
        request_info = RequestInput(request, sampling_params, client_id, api_url)
        deadline = client_deadline

        if request.request_type == RequestType.Collective:
            tasks.append(asyncio.create_task(send_collective_request(request_info, deadline, pbar)))
        else:
            tasks.append(asyncio.create_task(send_request(request_info, deadline, pbar)))
        
    latency = time.perf_counter() - start_time
    outputs: List[RequestOutput] = await asyncio.gather(*tasks)
    
    return outputs