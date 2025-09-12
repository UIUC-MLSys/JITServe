"""DeepResearch client for handling collective requests with multiple stages."""
import aiohttp
import asyncio
import json
import numpy as np
import pandas as pd
import time
import sys
import traceback
from copy import deepcopy
from typing import List, Tuple, AsyncGenerator, Optional, Dict, Any
from tqdm.asyncio import tqdm as async_tqdm
from vllm import SamplingParams

from .trace_deepresearch import (
    RequestType, DeepResearchCollectiveRequest,
    DeepResearchStage, DeepResearchRequest
)
from vllm.request_info import RequestInfo


class RequestInput:
    def __init__(
        self, 
        request: RequestInfo,
        slo_constraint: Tuple[float, float, float],
        sampling_params: SamplingParams,
        client_id: int,
        api_url: str,
        num_stages: int = 0,
        requests_per_stage: List[int] = None,
        accumulate_stage_ratio: float = 0.0
    ):
        self.request = request
        self.slo_constraint = slo_constraint
        self.sampling_params = sampling_params
        self.client_id = client_id
        self.api_url = api_url
        self.num_stages = num_stages
        self.requests_per_stage = requests_per_stage or []
        self.accumulate_stage_ratio = accumulate_stage_ratio


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
        # Stage-specific metrics
        self.collection_id = -1
        self.stage_id = -1
        self.request_id = -1
        self.state = ''
        # Task level metrics
        self.task_latency = 0
        # Debug properties
        self.success = False
        self.error = ""
        self.stop_reason = ""


class StageOutput:
    """Output for a complete stage with multiple requests."""
    def __init__(self):
        self.stage_id = -1
        self.stage_latency = 0
        self.stage_outputs: List[RequestOutput] = []
        self.success = False


class CollectiveOutput:
    """Output for a complete collective request with multiple stages."""
    def __init__(self):
        self.collection_id = -1
        self.total_latency = 0
        self.stage_outputs: List[StageOutput] = []
        self.success = False


def get_prompt_format_func(model_name: str) -> callable:
    """Get the prompt format function based on the model name."""
    # For DeepResearch, we use the prompts as-is since they're already formatted
    return lambda x: x


async def send_single_request(
    request_info: RequestInput,
    model_name: str,
    client_deadline: float = 200,
    is_stream: bool = True,
) -> RequestOutput:
    """Send a single request to the model."""
    api_url = request_info.api_url
    prompt_format = get_prompt_format_func(model_name)
    request_prompt = prompt_format(request_info.request.prompt)
    
    output = RequestOutput()
    output.request_type = request_info.request.request_type.value
    output.request_input = request_prompt
    output.collection_id = request_info.request.collection_id
    output.stage_id = request_info.request.stage_id
    output.request_id = request_info.request.request_id
    output.state = request_info.request.state
    
    timeout = aiohttp.ClientTimeout(total=client_deadline)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Include target_output_length based on the trace data
        payload = {
            "request_info": request_info.request.to_dict(),
            "slo_constraint": request_info.slo_constraint,
            "sampling_params": request_info.sampling_params.to_dict(),
            "client_id": request_info.client_id,
            "stream": is_stream,
            "target_output_length": request_info.request.output_len,
            "num_stages": request_info.num_stages,
            "requests_per_stage": request_info.requests_per_stage,
            "accumulate_stage_ratio": request_info.accumulate_stage_ratio,
        }
        
        # Update the prompt in the payload
        payload["request_info"]["prompt"] = request_prompt
        
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
                            output.request_tbt = data.get("tbt", [])
                            output.request_service_gain = data.get("service_gain", 0)
                            output.stop_reason = data.get("finish_reason", "")
                            generated_text = data.get("text", [""])[0]
                            if output.request_ttft == 0.0 and data.get("ttft") is not None:
                                output.request_ttft = data["ttft"]
                    else:
                        data = json.loads(buffer)
                        output.request_tbt = data.get("tbt", [])
                        output.request_service_gain = data.get("service_gain", 0)
                        output.stop_reason = data.get("finish_reason", "")
                        generated_text = data.get("text", [""])[0]
                        if output.request_ttft == 0.0 and data.get("ttft") is not None:
                            output.request_ttft = data["ttft"]
                    
                    output.request_output = generated_text[len(request_prompt):]
                    
                    output.request_latency = time.perf_counter() - st
                    output.success = True
                else:
                    output.success = False
                    output.error = f"Failed response, status: {response.status}"
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
    
    output.request_finish_time = time.perf_counter()
    return output


async def send_stage_requests(
    stage_requests: List[RequestInfo],
    slo_constraint: Tuple[float, float, float],
    sampling_params: SamplingParams,
    client_id: int,
    api_url: str,
    model_name: str,
    client_deadline: float = 200,
    is_stream: bool = True,
    num_stages: int = 0,
    requests_per_stage: List[int] = None,
    collective_request: 'DeepResearchCollectiveRequest' = None,
) -> StageOutput:
    """Send all requests in a stage concurrently."""
    stage_output = StageOutput()
    stage_output.stage_id = stage_requests[0].stage_id if stage_requests else -1
    
    st = time.perf_counter()
    
    # Calculate accumulate_stage_ratio for this stage
    accumulate_stage_ratio = 0.0
    if collective_request is not None and stage_requests:
        current_stage_id = stage_requests[0].stage_id
        accumulate_stage_ratio = collective_request.calculate_accumulate_stage_ratio(current_stage_id)
    
    # Create tasks for all requests in the stage
    tasks = []
    for request in stage_requests:
        request_info = RequestInput(
            request, slo_constraint, sampling_params, client_id, api_url, 
            num_stages, requests_per_stage or [], accumulate_stage_ratio
        )
        tasks.append(send_single_request(request_info, model_name, client_deadline, is_stream))
    
    # Execute all requests in the stage concurrently
    request_outputs = await asyncio.gather(*tasks)
    
    stage_output.stage_outputs = request_outputs
    stage_output.stage_latency = time.perf_counter() - st
    stage_output.success = all(output.success for output in request_outputs)
    
    return stage_output


async def send_deepresearch_collective_request(
    collective_request: DeepResearchCollectiveRequest,
    slo_constraint: Tuple[float, float, float],
    sampling_params: SamplingParams,
    client_id: int,
    api_url: str,
    model_name: str,
    penalty_factor: int = 1,
    client_deadline: float = 200,
    is_stream: bool = True,
    pbar: Optional[async_tqdm] = None,
) -> CollectiveOutput:
    """Send a DeepResearch collective request with multiple stages."""
    collective_output = CollectiveOutput()
    collective_output.collection_id = collective_request.collection_id
    
    st = time.perf_counter()
    
    # Get stage structure information
    num_stages, requests_per_stage = collective_request.get_stage_structure()
    
    # Convert to RequestInfo objects
    all_requests = collective_request.to_request_infos()
    
    # Group requests by stage
    stages_dict: Dict[int, List[RequestInfo]] = {}
    for request in all_requests:
        if request.stage_id not in stages_dict:
            stages_dict[request.stage_id] = []
        stages_dict[request.stage_id].append(request)
    
    # Process stages sequentially (as they may depend on each other)
    for stage_id in sorted(stages_dict.keys()):
        stage_requests = stages_dict[stage_id]
        
        # Send all requests in this stage
        stage_output = await send_stage_requests(
            stage_requests,
            slo_constraint,
            sampling_params,
            client_id,
            api_url,
            model_name,
            client_deadline,
            is_stream,
            num_stages,
            requests_per_stage,
            collective_request
        )
        
        collective_output.stage_outputs.append(stage_output)
        
        # If a stage fails, stop processing
        if not stage_output.success:
            collective_output.success = False
            break
    else:
        collective_output.success = True
    
    collective_output.total_latency = time.perf_counter() - st
    
    # Apply penalty if deadline is exceeded
    total_deadline = slo_constraint[2] * collective_request.stage_num
    if collective_output.total_latency > total_deadline:
        # penalty = min(1, (total_deadline / collective_output.total_latency) ** penalty_factor)
        penalty = 1 if total_deadline >= collective_output.total_latency else 0
        for stage_output in collective_output.stage_outputs:
            for request_output in stage_output.stage_outputs:
                request_output.request_service_gain *= penalty
                request_output.finish_before_ddl = False
    else:
        for stage_output in collective_output.stage_outputs:
            for request_output in stage_output.stage_outputs:
                request_output.finish_before_ddl = True
    
    if pbar is not None:
        pbar.update(1)
    
    return collective_output


async def get_request_generator(
    collective_requests: List[DeepResearchCollectiveRequest],
    poisson_lambda: float = 1000.0,
    burst: bool = False,
) -> AsyncGenerator[DeepResearchCollectiveRequest, None]:
    """Generate collective requests with specified arrival pattern."""
    request_num = len(collective_requests)
    
    # if burst using the burst pattern
    if burst:
        print("Using BurstGPT pattern for DeepResearch collective requests.")
        df = pd.read_csv('/home/jovyan/workspace/Concord/benchmarks/trace/BurstGPT_1.csv')
        timestamps = df['Timestamp'].tolist()
        baseline_timestamp = timestamps[99]
        timestamps = [ts - baseline_timestamp for ts in timestamps[100:100 + request_num]]
        original_req_rate = request_num * 1000 / timestamps[request_num - 1]
        target_req_rate = 1 / poisson_lambda * 1000
        timestamps = [int(timestamp * original_req_rate / target_req_rate) for timestamp in timestamps]
        last_timestamp = 0

        for collective_request, new_timestamp in zip(collective_requests, timestamps):
            interval = (new_timestamp - last_timestamp) / 1000
            last_timestamp = new_timestamp
            await asyncio.sleep(interval)
            yield collective_request
    else:
        for collective_request in collective_requests:
            interval = np.random.poisson(poisson_lambda) / 1000
            await asyncio.sleep(interval)
            yield collective_request


async def deepresearch_client_simulator(
    collective_requests: List[DeepResearchCollectiveRequest],
    slo_constraint: Tuple[float, float, float],
    penalty_factor: int,
    poisson_lambda: float,
    burst: bool,
    sampling_params: SamplingParams,
    client_id: int,
    client_deadline: float,
    api_url: str,
    model_name: str,
    is_stream: bool = True,
    pbar: Optional[async_tqdm] = None,
) -> List[CollectiveOutput]:
    """Execute the DeepResearch client to send collective requests."""
    start_time = time.perf_counter()
    tasks = []
    
    async for collective_request in get_request_generator(
        collective_requests, poisson_lambda, burst
    ):
        task = asyncio.create_task(
            send_deepresearch_collective_request(
                collective_request,
                slo_constraint,
                sampling_params,
                client_id,
                api_url,
                model_name,
                penalty_factor,
                client_deadline,
                is_stream,
                pbar
            )
        )
        tasks.append(task)
    
    collective_outputs = await asyncio.gather(*tasks)
    
    # Adjust finish times relative to start
    for collective_output in collective_outputs:
        for stage_output in collective_output.stage_outputs:
            for request_output in stage_output.stage_outputs:
                request_output.request_finish_time -= start_time
    
    return collective_outputs