import aiohttp
import asyncio
import json
import random
import sys
import time
import traceback
from tqdm import tqdm
from typing import List, Tuple, AsyncGenerator, Optional
from vllm import SamplingParams
from .trace import  RequestFormat, RequestType


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
        self.latency = 0.0
        self.generated_text = ""
        self.success = False
        self.error = ""
        self.itl = []
        self.ttft = 0.0
        self.collection_num_requests = 1


def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


async def get_request(
    input_requests: List[RequestFormat],
) -> AsyncGenerator[RequestFormat, None]:
    input_requests = iter(input_requests)
    last_request_deliver_time = 0
    
    for request in input_requests:
        yield request

        interval = (request.deliver_time - last_request_deliver_time) / 1000
        last_request_deliver_time = request.deliver_time
        await asyncio.sleep(interval)
        

async def send_collective_request(
    request_info: RequestInput, 
    client_deadline: int = 10,
    pbar: Optional[tqdm] = None
    ) -> RequestOutput:
    '''
    Send collective requests to the model and return the responses.
    '''
    api_url = request_info.api_url
    timeout = aiohttp.ClientTimeout(total=client_deadline)
    output = RequestOutput()
    
    # ToT parameters
    num_thoughts = 3
    tot_round = 2
    request_info.sampling_params.n = num_thoughts
    request_info.sampling_params.best_of = num_thoughts
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        payload = {
            "request_info": request_info.request.to_dict(),
            "sampling_params": request_info.sampling_params.to_dict(),
            "client_id": request_info.client_id,
            "stream": True
        }
        generated_text = ""
        ttft = 0.0
        num_requests = 0
        st = time.perf_counter()
        most_recent_timestamp = st
        
        async def generate_thoughts(session, api_url, payload) -> Tuple[bool, List[str]]:
            ttft = 0.0
            try:
                async with session.post(url=api_url, json=payload) as response:
                    if response.status == 200:
                        async for chunk_bytes in response.content:
                            chunk_bytes = chunk_bytes.strip()
                            if not chunk_bytes:
                                continue
                            chunk = remove_prefix(chunk_bytes.decode("utf-8"), "ret: ")
                            data = json.loads(chunk[:-1])
                            choices = [x for x in data["text"]]
                            timestamp = time.perf_counter()
                            # First token
                            if ttft == 0.0:
                                ttft = time.perf_counter() - st
                            # Decoding phase
                            # else:
                            #     output.itl.append(timestamp -
                            #                       most_recent_timestamp)
                            most_recent_timestamp = timestamp
                    else:
                        return (False, [], ttft)
            except Exception:
                exc_info = sys.exc_info()
                error = "".join(traceback.format_exception(*exc_info))
                return (False, [], ttft)
            
            return (True, choices, ttft)
        
        async def value_thoughts(session, api_url, choices, payload) -> Tuple[bool, List[float]]:
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
            
            new_payload = payload.copy()
            new_payload["request_info"]["prompt"] = value_prompt
            new_payload["sampling_params"]["n"] = 1
            try:
                async with session.post(url=api_url, json=new_payload) as response:
                    if response.status == 200:
                        async for chunk_bytes in response.content:
                            chunk_bytes = chunk_bytes.strip()
                            if not chunk_bytes:
                                continue
                            # TODO chunk parsing
                            chunk = remove_prefix(chunk_bytes.decode("utf-8"), "ret: ")
                            data = json.loads(chunk[:-1])
                            value = data["text"][0]
                            
                            # To ensure that the simulation results remain unaffected by the behavior of the value model,
                            # which may sometimes fail to generate values (e.g., unexpected response formats),
                            # we use randomly generated values as a fallback mechanism.
                            value = [random.random() for _ in range(len(choices))]
                    else:
                        return (False, [])
            except Exception:
                exc_info = sys.exc_info()
                error = "".join(traceback.format_exception(*exc_info))  
                return (False, [])     
              
            return (True, value)
            
        
        try:
            # using ToT reuqest pattern for collective requests
            thoughts = [request_info.request.prompt]
            output.success = True
            for round in range(tot_round):
                # generate thoughts
                tasks = []
                for thought in thoughts:
                    new_payload = payload.copy()
                    if round > 0:
                        thought += "\nPlease review and revise the response."
                    new_payload["request_info"]["prompt"] = thought
                    tasks.append(generate_thoughts(session, api_url, new_payload))
                
                results = await asyncio.gather(*tasks)
                
                choices = []
                for generate_success, generate_choices, generate_ttft in results:
                    if ttft < generate_ttft:
                        ttft = generate_ttft
                    if not generate_success:
                        output.error = "Generate thoughts failed"
                        output.success = False
                        break
                    else:
                        choices.extend(generate_choices)                
                if not output.success:
                    break
                        
                for choice in choices:
                    generated_text += choice
                    num_requests += 1
                # value thoughts and select the top-k
                value_success, value = await value_thoughts(session, api_url, choices, payload)
                if not value_success:
                    output.error = "Value thoughts failed"
                    output.success = False
                    break
                num_requests += 1
                
                # sort the choices based on the value
                choices = [x for _, x in sorted(zip(value, choices), reverse=True)]
                thoughts = choices[:num_thoughts]
            
            generated_text = thoughts[0]
            output.generated_text = generated_text
            output.success = True
            output.collection_num_requests = num_requests
            output.latency = time.perf_counter() - st
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
    
    if pbar is not None:
        pbar.update(1)
    return output


async def send_request(
    request_info: RequestInput, 
    client_deadline: int = 10,
    pbar: Optional[tqdm] = None
    ) -> RequestOutput:
    '''
    Send requests to the model and return the responses.
    '''
    api_url = request_info.api_url
    timeout = aiohttp.ClientTimeout(total=client_deadline)
    output = RequestOutput()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        payload = {
            "request_info": request_info.request.to_dict(),
            "sampling_params": request_info.sampling_params.to_dict(),
            "client_id": request_info.client_id,
            "stream": False
        }
        # output.prompt_len = request_info.prompt_len
        print(f"request_info.client_id: {request_info.client_id}")

        generated_text = ""
        ttft = 0.0
        latency = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        
                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "ret: ")
                        # TODO chunk parsing
                        # remove EOS token
                        data = json.loads(chunk)
                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0.0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft
                        # Decoding phase
                        else:
                            output.itl.append(timestamp -
                                              most_recent_timestamp)
                        most_recent_timestamp = timestamp
                        generated_text += data["text"][0]
                        
                    output.generated_text = generated_text
                    output.success = True
                    output.latency = time.perf_counter() - st
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
    async for request in get_request(input_requests):
        # request_format = RequestFormat.from_dict(request)
        # TODO            
        request_info = RequestInput(request, sampling_params, client_id, api_url)
        print(f"client_id: {client_id}")
        if request.request_type == RequestType.Collective:
            tasks.append(asyncio.create_task(send_collective_request(request_info, client_deadline, pbar)))
        else:
            tasks.append(asyncio.create_task(send_request(request_info, client_deadline, pbar)))
        
    latency = time.perf_counter() - start_time
    outputs: List[RequestOutput] = await asyncio.gather(*tasks)
    
    return outputs