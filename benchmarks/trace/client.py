import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
from vllm import LLM, SamplingParams
from vllm.scheduler import RequestInfo
from .trace import Trace, BaseDataset, Request, RequestType


def sync_generate(llm, prompt, sampling_params):
    return llm.generate(prompt, sampling_params)

async def send_requests(llm: LLM, request: Request, 
                        sampling_params: SamplingParams) -> Tuple[List, float, int]:
    '''
    Send requests to the model and return the responses.
    '''
    collection_num = 0
    start_time = time.time()
    loop = asyncio.get_running_loop()
    if request.request_type == RequestType.Collective:
        # TODO(Wei): should use ToT pattern
        collection_num = 1
        request_info = RequestInfo(request.request_type, request.deadline, 
                                   request.collection_id)
        response = loop.run_in_executor(None, sync_generate, llm, request.prompt, sampling_params)
    else:
        collection_num = 1
        request_info = RequestInfo(request.request_type, request.deadline, 
                                   request.collection_id)
        response = loop.run_in_executor(None, sync_generate, llm, request.prompt, sampling_params)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return (response, elapsed_time, collection_num)


async def client_simulator(llm: LLM, trace: List[dict], 
                           sampling_params: SamplingParams) -> Tuple[List, float]:
    '''
    Execute the client to send requests to the model.
    '''
    start_time = time.time()
    tasks = []
    
    for data in trace:
        request = Request.from_dict(data)
        await asyncio.sleep(request.deliver_time / 1000)

        task = asyncio.create_task(send_requests(llm, request, sampling_params))
        tasks.append(task)
    
    responses = await asyncio.gather(*tasks)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return (responses, elapsed_time)