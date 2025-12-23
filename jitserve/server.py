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
from typing import Any, AsyncGenerator, Optional, Tuple, List, Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.launcher import serve_http
from vllm.logger import init_logger
from jitserve.request_analyzer.prediction import load_model
from jitserve.request_analyzer.deepresearch_trace_reader import read_deepresearch_traces
from jitserve.request_analyzer.graph_context import GraphMatchContext
from jitserve.request_analyzer.similarity import Default_DeepResearch_Stage
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import (FlexibleArgumentParser, iterate_with_cancellation,
                        random_uuid)

# Import tokenizer utilities
from vllm.transformers_utils.tokenizer import get_tokenizer

from jitserve.request_info import RequestInfo, RequestType

logger = init_logger("vllm.entrypoints.api_server")

TIMEOUT_KEEP_ALIVE = 5  # seconds.
DEFAULT_PREDICTION_LENGTH = 1024
app = FastAPI()
engine = None

# Prediction model
use_prediction = False
use_default_length = False

# Deepresearch base graphs for graph matching
deepresearch_base_graphs = []
prediction_model = None
prediction_tokenizer = None

# Generation tokenizer
generation_tokenizer = None

# Graph matching
use_graph_matching = False
graph_structure_type = "tot"  # "tot" or "deepresearch"
graph_matching_mode = "online"  # "none", "static", "online", "precise"
use_total_deadline = False
use_all_node = False  # All-node vs super-node approach
stage_ratio_method = "execution_time"  # "execution_time" or "output_length"
graph_ctx: Optional[GraphMatchContext] = None

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


def _build_sampling_params(request_dict: Dict[str, Any]) -> SamplingParams:
    target_output_length = request_dict.pop("target_output_length", None)
    sampling_params_dict = request_dict.pop("sampling_params")
    if target_output_length is not None:
        sampling_params_dict["target_output_length"] = target_output_length
    return SamplingParams(**sampling_params_dict)


def _build_request_info(request_dict: Dict[str, Any],
                        client_id: int) -> Tuple[RequestInfo, str]:
    request_info = request_dict.pop("request_info", {})
    prompt = request_info.get("prompt", "")
    request_info["client_id"] = client_id
    request_info["slo_constraint"] = tuple(request_dict["slo_constraint"])
    return RequestInfo.from_json(request_info), prompt


def _apply_request_deadline(request_info: RequestInfo,
                            request_dict: Dict[str, Any]) -> None:
    if request_info.request_type == RequestType.THROUGHPUT:
        request_info.deadline = request_info.slo_constraint[2]
    elif request_info.request_type == RequestType.COLLECTIVE:
        num_stages = request_dict.get("num_stages", Default_DeepResearch_Stage)
        request_info.deadline = request_info.slo_constraint[2] * num_stages


def _maybe_start_prediction(request_info: RequestInfo, prompt: str) -> None:
    if not use_prediction:
        return
    request_info.output_len = DEFAULT_PREDICTION_LENGTH
    if not use_default_length:
        threading.Thread(target=send_and_update_request_info,
                         args=(request_info, prompt),
                         daemon=True).start()


def _handle_collective_request(request_info: RequestInfo,
                               request_dict: Dict[str, Any],
                               prompt: str) -> Tuple[bool, int]:
    if graph_ctx is None:
        return False, request_info.collection_id

    collection_id = request_info.collection_id
    stage_id = request_info.stage_id if hasattr(request_info, 'stage_id') else 0
    accumulate_stage_ratio = request_dict.pop("accumulate_stage_ratio", 0.0)
    num_stages = request_dict.pop("num_stages", Default_DeepResearch_Stage)
    requests_per_stage = request_dict.pop("requests_per_stage", None)
    served_time = request_dict.pop("served_time", 0.0)

    is_deepresearch = requests_per_stage is not None
    graph_ctx.init_collection(collection_id, is_deepresearch, num_stages,
                              requests_per_stage)

    if use_graph_matching or graph_ctx.graph_matching_mode != "none":
        stage_ratio = graph_ctx.calculate_stage_ratio(
            num_stages,
            requests_per_stage,
            prompt,
            collection_id,
            is_deepresearch=is_deepresearch,
            stage_id=stage_id,
            accumulate_stage_ratio=accumulate_stage_ratio,
        )
        request_info.deadline *= stage_ratio
        request_info.deadline -= served_time

    return is_deepresearch, collection_id

def _track_collection_completion(collection_id: int, is_deepresearch: bool,
                                 request_input_length: int,
                                 request_output_length: int) -> None:
    if graph_ctx is None:
        return
    graph_ctx.track_collection_completion(collection_id, is_deepresearch,
                                          request_input_length,
                                          request_output_length)

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
    stream = request_dict.pop("stream", False)
    client_id = request_dict.get("client_id", 0)

    logger.debug(f"stream: {stream}")

    sampling_params = _build_sampling_params(request_dict)
    request_info, prompt = _build_request_info(request_dict, client_id)
    _apply_request_deadline(request_info, request_dict)
    _maybe_start_prediction(request_info, prompt)

    request_id = random_uuid()
    is_deepresearch = False
    collection_id = request_info.collection_id
    if request_info.request_type == RequestType.COLLECTIVE:
        is_deepresearch, collection_id = _handle_collective_request(
            request_info, request_dict, prompt)

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
            request_input_length = len(request_output.prompt_token_ids)
            assert prompt is not None
            text_outputs = [
                prompt + output.text for output in request_output.outputs
            ]
            request_output_length = sum([len(output.token_ids) for output in request_output.outputs])
            
            ret = {"text": text_outputs, "ttft": request_output.ttft,
                   "tbt": request_output.tbt, "service_gain": request_output.service_gain}
            yield (json.dumps(ret) + "\0").encode("utf-8")
            
        # Track collection completion for graph learning (regardless of matching mode)
        if request_info.request_type == RequestType.COLLECTIVE:
            _track_collection_completion(collection_id, is_deepresearch,
                                         request_input_length,
                                         request_output_length)

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

    request_input_length = len(final_output.prompt_token_ids)
    request_output_length = sum([len(output.token_ids) for output in final_output.outputs])
    logger.info(f"Request input length: {request_input_length}, Request output length: {request_output_length}")

    # Track collection completion for graph learning (regardless of matching mode)
    if request_info.request_type == RequestType.COLLECTIVE:
        _track_collection_completion(collection_id, is_deepresearch,
                                     request_input_length,
                                     request_output_length)

    text_outputs = [prompt + output.text for output in final_output.outputs]
    
    # Get finish_reason from the first output (assuming single output)
    finish_reason = final_output.outputs[0].finish_reason if final_output.outputs else None
    
    ret = {"text": text_outputs, "time": final_output.ttft,
           "tbt": final_output.tbt, "service_gain": final_output.service_gain,
           "finish_reason": finish_reason}
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
    global use_all_node
    global stage_ratio_method
    global generation_tokenizer
    global graph_ctx

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_args.max_model_len = 8192

    if not args.disable_prediction:
        if engine_args.scheduling_policy in ['ltr', 'jitserve', 'slo']:
            use_prediction = True
            logger.info(f"Prediction model is enabled")
            
    if not args.disable_graph_matching:
        if engine_args.scheduling_policy in ['jitserve', 'slo', 'srtf']:
            use_graph_matching = True
            logger.info(f"Graph matching is enabled")

    if args.use_default_length:
        use_default_length = True
    
    # Set graph structure type and matching mode from arguments
    graph_structure_type = args.graph_structure_type
    graph_matching_mode = args.graph_matching_mode
    use_total_deadline = args.use_total_deadline
    use_all_node = args.use_all_node
    stage_ratio_method = args.stage_ratio_method
    logger.info(f"Graph structure type set to: {graph_structure_type}")
    logger.info(f"Graph matching mode set to: {graph_matching_mode}")
    logger.info(f"Use total deadline: {use_total_deadline}")
    logger.info(f"Use all node: {use_all_node}")
    logger.info(f"Stage ratio method: {stage_ratio_method}")
    
    # Override graph matching based on mode
    if graph_matching_mode == "none":
        use_graph_matching = False
        logger.info("Graph matching disabled by mode")
    
    # Initialize generation tokenizer
    generation_tokenizer = get_tokenizer(args.model)
    logger.info(f"Generation tokenizer initialized for model: {args.model}")

    graph_ctx = GraphMatchContext(
        generation_tokenizer=generation_tokenizer,
        use_all_node=use_all_node,
        stage_ratio_method=stage_ratio_method,
        graph_structure_type=graph_structure_type,
        graph_matching_mode=graph_matching_mode,
        use_total_deadline=use_total_deadline,
    )
    
    # Load deepresearch training data and construct graphs
    logger.info("Loading deepresearch training data...")
    deepresearch_training_path = "benchmarks/dataset/trace/deepresearch_training.json"
    try:
        deepresearch_graphs = read_deepresearch_traces(deepresearch_training_path, is_all_node=use_all_node)
        logger.info(f"Loaded {len(deepresearch_graphs)} deepresearch training graphs.")
        global deepresearch_base_graphs
        deepresearch_base_graphs = deepresearch_graphs
        if graph_ctx is not None:
            graph_ctx.add_deepresearch_base_graphs(deepresearch_graphs)
        
    except FileNotFoundError:
        logger.warning(f"Deepresearch training file not found: {deepresearch_training_path}")
        deepresearch_base_graphs = []
    except Exception as e:
        logger.error(f"Error loading deepresearch training data: {e}")
        deepresearch_base_graphs = []
        
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
        default='assets/qrf/qrf_model/0_qrf_lmsys_chat_llama3_8b.pkl',
        help="Path to the prediction model")
    parser.add_argument(
        "--prediction-tokenizer-path",
        type=str,
        default='assets/qrf/qrf_vectorizer/0_qrf_lmsys_chat_llama3_8b.pkl',
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
    parser.add_argument(
        "--use-all-node",
        action='store_true',
        help="Use all-node approach (each request becomes a node) instead of super-node approach (one node per stage)")
    parser.add_argument(
        "--stage-ratio-method",
        type=str,
        choices=["execution_time", "output_length"],
        default="execution_time",
        help="Method for calculating stage ratios: 'execution_time' uses actual timing, 'output_length' uses max output length per stage")
    parser.add_argument("--log-level", type=str, default="debug")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    asyncio.run(run_server(args))
