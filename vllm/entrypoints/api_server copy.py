"""
NOTE: This API server is used only for demonstrating usage of AsyncEngine
and simple performance benchmarks. It is not intended for production use.
For production use, we recommend using our OpenAI compatible server.
We are also not going to accept PRs modifying this file, please
change `vllm/entrypoints/openai/api_server.py` instead.
"""
import asyncio
import json
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
from vllm.graph.similarity import Graph, ToTStructure, predict_stage_ratio
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
prediction_model = None
prediction_tokenizer = None

# Graph matching
use_graph_matching = False
graph_matching_lock = threading.Lock()
collection_graph_set: Set[Graph] = set()
collection_graph_unfinished_dict: Dict[int, ToTStructure] = dict()

def send_request_to_prediction(connection: socket.socket, request_info: RequestInfo, prompt):
    try:
        data = {'prompt': [prompt]}
        data_to_send = pickle.dumps(data)
        connection.sendall(data_to_send)
        logger.info(f"Sent {request_info.collection_id} prompt to prediction.py")

        data = connection.recv(4096)
        if data:
            received_data = pickle.loads(data)
            logger.info(f"Received response from prediction.py")
            return received_data['output_len'][0]
        else:
            logger.info("No response received from yy.py.")
            return 1024
    except socket.error as e:
        logger.info(f"Error while sending/receiving data: {e}")
        return 1024

def send_and_update_request_info(request_info: RequestInfo, prompt: str):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', 65433))

        result_len = send_request_to_prediction(s, request_info, prompt)

        if result_len:
            request_info.output_len = result_len
            logger.info(f"Updated request_info.output_len: {request_info.output_len}")

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
    
    sampling_params = SamplingParams(**request_dict.pop("sampling_params"))
    
    prompt = request_info.get("prompt", "")
    request_info["client_id"] = client_id
    request_info = RequestInfo.from_json(request_info)
    request_info.real_output_len = request_info.output_len
    request_info.output_len = 1024
    request_id = random_uuid()
    
    if use_prediction:
        request_info.output_len = 1024
        threading.Thread(target=send_and_update_request_info, args=(request_info, prompt), daemon=True).start()
    
    if use_graph_matching and request_info.request_type == RequestType.COLLECTIVE:
        collection_id = request_info.collection_id
        if collection_id not in collection_graph_unfinished_dict:
            collection_graph_unfinished_dict[collection_id] = ToTStructure()
        tot_structure = collection_graph_unfinished_dict[collection_id]
        if tot_structure.is_finished:
            tot_structure.reset()
        
        unfinished_graph = tot_structure.convert_to_unfinished_graph(
            len(prompt), request_info.output_len)
        stage_ratio = predict_stage_ratio(unfinished_graph, collection_graph_set)
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
            ret = {"text": text_outputs, "time": request_output.ttft}
            yield (json.dumps(ret) + "\0").encode("utf-8")
            
        if use_graph_matching and request_info.request_type == RequestType.COLLECTIVE:
            tot_structure = collection_graph_unfinished_dict[collection_id]
            collection_finish = tot_structure.add_length(request_input_length, request_output_length)
            
            if collection_finish:
                collection_graph_set.add(tot_structure.convert_to_graph())
                tot_structure.is_finished = True

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
    
    if use_graph_matching and request_info.request_type == RequestType.COLLECTIVE:
        tot_structure = collection_graph_unfinished_dict[collection_id]
        collection_finish = tot_structure.add_length(request_input_length, request_output_length)
        
        if collection_finish:
            collection_graph_set.add(tot_structure.convert_to_graph())
            tot_structure.is_finished = True

    text_outputs = [prompt + output.text for output in final_output.outputs]
    ret = {"text": text_outputs, "time": final_output.ttft}
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

    engine_args = AsyncEngineArgs.from_cli_args(args)

    if not args.disable_prediction:
        if engine_args.scheduling_policy in ['sjf', 'slo']:
            use_prediction = True
            logger.info(f"Prediction model is enabled")
            
    if not args.disable_graph_matching:
        if engine_args.scheduling_policy in ['slo', 'srtf']:
            use_graph_matching = True
            logger.info(f"Graph matching is enabled")
        
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
        "--prediction-model-path",
        type=str,
        default='/home/exouser/qrf_model/0_qrf_lmsys_chat_llama3_8b.pkl',
        help="Path to the prediction model")
    parser.add_argument(
        "--prediction-tokenizer-path",
        type=str,
        default='/home/exouser/qrf_vectorizer/0_qrf_lmsys_chat_llama3_8b.pkl',
        help="Path to the prediction tokenizer")
    parser.add_argument("--log-level", type=str, default="debug")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    asyncio.run(run_server(args))
