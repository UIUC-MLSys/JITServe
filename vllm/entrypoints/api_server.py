"""
NOTE: This API server is used only for demonstrating usage of AsyncEngine
and simple performance benchmarks. It is not intended for production use.
For production use, we recommend using our OpenAI compatible server.
We are also not going to accept PRs modifying this file, please
change `vllm/entrypoints/openai/api_server.py` instead.
"""
import asyncio
import json
import numpy as np
import pickle
import socket
import ssl
from argparse import Namespace
import threading
import time
from typing import Any, AsyncGenerator, Optional, Tuple, List, Dict, Set

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.launcher import serve_http
from vllm.graph.similarity import (Graph, ToTStructure, DeepResearchStructure, 
                                      predict_stage_ratio, predict_deepresearch_stage_ratio,
                                      is_tot_request, is_deepresearch_request,
                                      Default_DeepResearch_Stage, Default_DeepResearch_Requests_Pattern)
from vllm.logger import init_logger
from vllm.prediction.prediction import load_model, async_predict
from vllm.request_info import RequestInfo, RequestType
from vllm.sampling_params import SamplingParams, RequestOutputKind
from vllm.usage.usage_lib import UsageContext
from vllm.utils import (FlexibleArgumentParser, iterate_with_cancellation,
                        random_uuid)
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger("vllm.entrypoints.api_server")

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None

# Prediction model
use_prediction = False
use_default_length = False
prediction_model = None
prediction_tokenizer = None

# Graph matching
use_graph_matching = False
graph_structure_type = "tot"  # "tot" or "deepresearch"
graph_matching_mode = "online"  # "none", "static", "online", "precise"
use_total_deadline = False
collection_graph_set: Set[Graph] = set()
collection_graph_unfinished_dict: Dict[int, ToTStructure] = dict()
deepresearch_structure: Optional[DeepResearchStructure] = None

def send_request_to_prediction(connection: socket.socket, request_info: RequestInfo, prompt):
    try:
        data = {'prompt': [prompt]}
        data_to_send = pickle.dumps(data)
        connection.sendall(data_to_send)

        data = connection.recv(4096)
        if data:
            received_data = pickle.loads(data)
            return received_data['output_len'][0]
        else:
            return 1024
    except socket.error as e:
        # logger.info(f"Error while sending/receiving data: {e}")
        return 1024

def send_and_update_request_info(request_info: RequestInfo, prompt: str):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', 65433))

        result_len = send_request_to_prediction(s, request_info, prompt)

        if result_len:
            request_info.output_len = result_len

def calculate_stage_ratio(request_info: RequestInfo, prompt: str, collection_id: int, stage_id: int = 0) -> float:
    """Calculate stage ratio based on graph matching mode and structure type."""
    global graph_matching_mode, graph_structure_type, use_total_deadline
    global collection_graph_set, collection_graph_unfinished_dict, deepresearch_structure
    
    # Default ratios based on structure type
    if graph_structure_type == "tot":
        from vllm.graph.similarity import Defalut_ToT_Requests_Pattern, Defalut_ToT_Stage
        if use_total_deadline:
            return 1.0  # Use total deadline
        default_ratio = Defalut_ToT_Requests_Pattern[min(stage_id, len(Defalut_ToT_Requests_Pattern)-1)] / sum(Defalut_ToT_Requests_Pattern)
    else:  # deepresearch
        if use_total_deadline:
            return 1.0  # Use total deadline
        default_ratio = 1.0 / Default_DeepResearch_Stage
    
    if graph_matching_mode == "none":
        return default_ratio
    elif graph_matching_mode == "precise":
        # Oracle knowledge - return optimal ratio (for now, use default)
        return default_ratio
    elif graph_matching_mode == "static":
        # First stage matching only
        if stage_id == 0:
            # Use graph matching for first stage
            if graph_structure_type == "tot":
                if collection_id not in collection_graph_unfinished_dict:
                    collection_graph_unfinished_dict[collection_id] = ToTStructure()
                tot_structure = collection_graph_unfinished_dict[collection_id]
                if tot_structure.is_finished:
                    tot_structure.reset()
                unfinished_graph = tot_structure.convert_to_unfinished_graph(
                    len(prompt), request_info.output_len)
                return predict_stage_ratio(unfinished_graph, collection_graph_set)
            else:  # deepresearch
                if deepresearch_structure is not None:
                    unfinished_graph = deepresearch_structure.convert_to_unfinished_graph(
                        collection_id, len(prompt), request_info.output_len)
                    if unfinished_graph is not None:
                        return predict_deepresearch_stage_ratio(
                            unfinished_graph, collection_graph_set, stage_id)
        return default_ratio
    elif graph_matching_mode == "online":
        # Dynamic graph matching (current implementation)
        if graph_structure_type == "tot":
            if collection_id not in collection_graph_unfinished_dict:
                collection_graph_unfinished_dict[collection_id] = ToTStructure()
            tot_structure = collection_graph_unfinished_dict[collection_id]
            if tot_structure.is_finished:
                tot_structure.reset()
            unfinished_graph = tot_structure.convert_to_unfinished_graph(
                len(prompt), request_info.output_len)
            return predict_stage_ratio(unfinished_graph, collection_graph_set)
        else:  # deepresearch
            if deepresearch_structure is not None:
                unfinished_graph = deepresearch_structure.convert_to_unfinished_graph(
                    collection_id, len(prompt), request_info.output_len)
                if unfinished_graph is not None:
                    return predict_deepresearch_stage_ratio(
                        unfinished_graph, collection_graph_set, stage_id)
    
    return default_ratio

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    request_info = request_dict.pop("request_info", {})
    stream = request_dict.pop("stream", False)
    client_id = request_dict.get("client_id", 0)
    
    # Extract target_output_length if provided by client
    target_output_length = request_dict.pop("target_output_length", None)
    
    sampling_params_dict = request_dict.pop("sampling_params")
    # Add target_output_length to sampling params if provided
    if target_output_length is not None:
        sampling_params_dict["target_output_length"] = target_output_length
    
    sampling_params = SamplingParams(**sampling_params_dict)
    
    prompt = request_info.get("prompt", "")
    request_info["client_id"] = client_id
    request_info["slo_constraint"] = tuple(request_dict["slo_constraint"])
    request_info = RequestInfo.from_json(request_info)
    request_id = random_uuid()
    
    if use_prediction:
        # using network socket to send request to prediction model
        request_info.output_len = 1024
        if not use_default_length:
            threading.Thread(target=send_and_update_request_info, args=(request_info, prompt), daemon=True).start()
    
    # Handle graph matching and deadline adjustment for collective requests
    if request_info.request_type == RequestType.COLLECTIVE:
        collection_id = request_info.collection_id
        stage_id = request_info.stage_id if hasattr(request_info, 'stage_id') else 0
        
        # Initialize structures based on type
        if graph_structure_type == "tot":
            # Initialize ToT structure if needed
            if collection_id not in collection_graph_unfinished_dict:
                collection_graph_unfinished_dict[collection_id] = ToTStructure()
            tot_structure = collection_graph_unfinished_dict[collection_id]
            if tot_structure.is_finished:
                tot_structure.reset()
                
        elif graph_structure_type == "deepresearch" and deepresearch_structure is not None:
            # Initialize DeepResearch collective if needed
            num_stages = getattr(request_info, 'num_stages', Default_DeepResearch_Stage)
            requests_per_stage = getattr(request_info, 'requests_per_stage', Default_DeepResearch_Requests_Pattern)
            
            if not deepresearch_structure.is_collective_finished(collection_id):
                deepresearch_structure.initialize_collective(
                    collection_id, num_stages, requests_per_stage)
        
        # Calculate stage ratio based on matching mode and structure
        if use_graph_matching or graph_matching_mode != "none":
            stage_ratio = calculate_stage_ratio(request_info, prompt, collection_id, stage_id)
            request_info.deadline *= stage_ratio

    assert engine is not None
    results_generator = engine.generate(prompt, request_info, 
                                        sampling_params, request_id, client_id=request_info.client_id)
    results_generator = iterate_with_cancellation(
        results_generator, is_cancelled=request.is_disconnected)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        request_input_length = 0
        request_output_length = 0
        async for request_output in results_generator:
            prompt = request_output.prompt
            request_input_length += len(prompt)
            assert prompt is not None
            text_outputs = [
                prompt + output.text for output in request_output.outputs
            ]
            request_output_length += sum([len(output.text) for output in request_output.outputs])
            ret = {"text": text_outputs, "ttft": request_output.ttft,
                   "tbt": request_output.tbt, "service_gain": request_output.service_gain}
            yield (json.dumps(ret) + "\0").encode("utf-8")
            
        # Track collection completion for graph learning (regardless of matching mode)
        if request_info.request_type == RequestType.COLLECTIVE:
            if graph_structure_type == "tot":
                if collection_id in collection_graph_unfinished_dict:
                    tot_structure = collection_graph_unfinished_dict[collection_id]
                    request_finished = getattr(request_output, 'finished', False)
                    collection_finish = tot_structure.add_length(
                        request_input_length, request_output_length, request_finished)
                    
                    if collection_finish:
                        # Only convert if we have valid data
                        if tot_structure.output_input_ratio and tot_structure.stage_finish_time:
                            collection_graph_set.add(tot_structure.convert_to_graph())
                        tot_structure.is_finished = True
                        
            elif graph_structure_type == "deepresearch" and deepresearch_structure is not None:
                stage_id = request_info.stage_id if hasattr(request_info, 'stage_id') else 0
                # Ensure collective is initialized before adding completion
                if collection_id in deepresearch_structure.collective_requests:
                    request_finished = getattr(request_output, 'finished', False)
                    collection_finish = deepresearch_structure.add_request_completion(
                        collection_id, stage_id, request_input_length, 
                        request_output_length, request_finished)
                else:
                    collection_finish = False
                
                if collection_finish:
                    graph = deepresearch_structure.convert_to_graph(collection_id)
                    if graph is not None:
                        collection_graph_set.add(graph)

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    try:
        async for request_output in results_generator:
            final_output = request_output
    except asyncio.CancelledError:
        return Response(status_code=499)

    assert final_output is not None
    prompt = final_output.prompt
    assert prompt is not None
    
    request_input_length = len(prompt)
    request_output_length = sum([len(output.text) for output in final_output.outputs])
    
    # Track collection completion for graph learning (regardless of matching mode)
    if request_info.request_type == RequestType.COLLECTIVE:
        if graph_structure_type == "tot":
            if collection_id in collection_graph_unfinished_dict:
                tot_structure = collection_graph_unfinished_dict[collection_id]
                request_finished = getattr(final_output, 'finished', False)
                collection_finish = tot_structure.add_length(
                    request_input_length, request_output_length, request_finished)
                
                if collection_finish:
                    # Only convert if we have valid data
                    if tot_structure.output_input_ratio and tot_structure.stage_finish_time:
                        collection_graph_set.add(tot_structure.convert_to_graph())
                    tot_structure.is_finished = True
                    
        elif graph_structure_type == "deepresearch" and deepresearch_structure is not None:
            stage_id = request_info.stage_id if hasattr(request_info, 'stage_id') else 0
            # Ensure collective is initialized before adding completion
            if collection_id in deepresearch_structure.collective_requests:
                request_finished = getattr(final_output, 'finished', False)
                collection_finish = deepresearch_structure.add_request_completion(
                    collection_id, stage_id, request_input_length, 
                    request_output_length, request_finished)
            else:
                collection_finish = False
            
            if collection_finish:
                graph = deepresearch_structure.convert_to_graph(collection_id)
                if graph is not None:
                    collection_graph_set.add(graph)

    text_outputs = [prompt + output.text for output in final_output.outputs]
    ret = {"text": text_outputs, "time": final_output.ttft,
           "tbt": final_output.tbt, "service_gain": final_output.service_gain}
    return JSONResponse(ret)


def build_app(args: Namespace) -> FastAPI:
    global app

    app.root_path = args.root_path
    return app


async def init_app(
    args: Namespace,
    llm_engine: Optional[AsyncLLMEngine] = None,
) -> FastAPI:
    app = build_app(args)

    global engine
    global use_prediction
    global use_graph_matching
    global use_default_length
    global graph_structure_type
    global graph_matching_mode
    global use_total_deadline
    global deepresearch_structure

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_args.max_model_len = 8192 #32768

    if not args.disable_prediction:
        if engine_args.scheduling_policy in ['sjf', 'concord']:
            use_prediction = True
            logger.info(f"Prediction model is enabled")
            
    if not args.disable_graph_matching:
        if engine_args.scheduling_policy in ['concord', 'srtf']:
            use_graph_matching = True
            logger.info(f"Graph matching is enabled")

    if args.use_default_length:
        use_default_length = True
    
    # Set graph structure type and matching mode from arguments
    graph_structure_type = args.graph_structure_type
    graph_matching_mode = args.graph_matching_mode
    use_total_deadline = args.use_total_deadline
    logger.info(f"Graph structure type set to: {graph_structure_type}")
    logger.info(f"Graph matching mode set to: {graph_matching_mode}")
    logger.info(f"Use total deadline: {use_total_deadline}")
    
    # Override graph matching based on mode
    if graph_matching_mode == "none":
        use_graph_matching = False
        logger.info("Graph matching disabled by mode")
    
    # Initialize DeepResearch structure if needed
    if graph_structure_type == "deepresearch":
        deepresearch_structure = DeepResearchStructure()
        logger.info("DeepResearch structure initialized")
        
    engine = (llm_engine
              if llm_engine is not None else AsyncLLMEngine.from_engine_args(
                  engine_args, usage_context=UsageContext.API_SERVER))

    return app


async def init_prediction_model(
    args: Namespace,
) -> None:
    model_path = args.prediction_model_path
    tokenizer_path = args.prediction_tokenizer_path

    global prediction_model
    global prediction_tokenizer

    prediction_model, prediction_tokenizer = load_model(model_path, tokenizer_path)
    
    assert prediction_model is not None
    assert prediction_tokenizer is not None

    return


async def run_server(args: Namespace,
                     llm_engine: Optional[AsyncLLMEngine] = None,
                     **uvicorn_kwargs: Any) -> None:
    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    if args.disable_prediction is False:
        await init_prediction_model(args)
    app = await init_app(args, llm_engine)
    assert engine is not None

    shutdown_task = await serve_http(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
        **uvicorn_kwargs,
    )

    await shutdown_task


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--ssl-ca-certs",
                        type=str,
                        default=None,
                        help="The CA certificates file")
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)"
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser.add_argument(
        "--disable-prediction",
        action='store_true',
        help="Enable prediction model")
    parser.add_argument(
        "--disable-graph-matching",
        action='store_true',
        help="Enable graph matching")
    parser.add_argument(
        "--use-default-length",
        action='store_true',
        help="Using default length in exp wo prediction model")
    parser.add_argument(
        "--prediction-model-path",
        type=str,
        default='/home/jovyan/workspace/qrf_model/0_qrf_lmsys_chat_llama3_8b.pkl',
        help="Path to the prediction model")
    parser.add_argument(
        "--prediction-tokenizer-path",
        type=str,
        default='/home/jovyan/workspace/qrf_vectorizer/0_qrf_lmsys_chat_llama3_8b.pkl',
        help="Path to the prediction tokenizer")
    parser.add_argument(
        "--graph-structure-type",
        type=str,
        choices=["tot", "deepresearch"],
        default="tot",
        help="Type of graph structure to use: 'tot' for Tree-of-Thoughts or 'deepresearch' for DeepResearch collective requests")
    parser.add_argument(
        "--graph-matching-mode",
        type=str,
        choices=["none", "static", "online", "precise"],
        default="online",
        help="Graph matching mode: 'none' (no matching), 'static' (first stage), 'online' (dynamic), 'precise' (oracle)")
    parser.add_argument(
        "--use-total-deadline",
        action='store_true',
        help="Use total deadline instead of default structure ratio")
    parser.add_argument("--log-level", type=str, default="debug")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    asyncio.run(run_server(args))
