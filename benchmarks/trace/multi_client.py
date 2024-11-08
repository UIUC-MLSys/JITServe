import asyncio
import json
import time
from typing import List, Tuple, Optional
import requests
from .trace import Trace, BaseDataset, Request, RequestType


def post_http_request(prompt: str,
                      api_url: str,
                      n: int = 1,
                      stream: bool = False) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "prompt": prompt,
        "n": n,
        "use_beam_search": False,
        "temperature": 0.0,
        "top_k": 1,
        "max_tokens": 1024,
        "stream": stream,
    }
    response = requests.post(api_url, headers=headers, json=pload, stream=stream)
    return response


def get_response(response: requests.Response) -> List[str]:
    data = json.loads(response.content)
    output = data["text"]
    return output


async def send_requests(api_url: str, request: Request,
                        n: int = 1, stream: bool = False) -> Tuple[List[str], float, int]:
    '''
    Sends requests to the model API and returns the responses.
    '''
    start_time = time.time()
    response = post_http_request(request.prompt, api_url, n, stream)
    if stream:
        output = [chunk for chunk in response.iter_lines(decode_unicode=True) if chunk]
    else:
        output = get_response(response)
    elapsed_time = time.time() - start_time
    collection_num = 1  # Customize collection number logic if needed
    return (output, elapsed_time, collection_num)


async def client_simulator(api_url: str, trace: List[dict],
                           n: int = 1, stream: bool = False) -> Tuple[List[Tuple[List[str], float, int]], float]:
    '''
    Simulates a client sending multiple requests to the model API.
    '''
    start_time = time.time()
    tasks = []

    for data in trace:
        request = Request.from_dict(data)
        await asyncio.sleep(request.deliver_time / 1000)  # Delays sending request
        task = asyncio.create_task(send_requests(api_url, request, n, stream))
        tasks.append(task)

    responses = await asyncio.gather(*tasks)
    elapsed_time = time.time() - start_time

    return (responses, elapsed_time)
