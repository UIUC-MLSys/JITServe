"""Multi-replica front end (paper §4.3, "Supporting Multiple Models").

Clients talk to this router exactly as they would to a single
`jitserve.server`. Each request is placed on one replica and its response is
streamed back.

  --routing power-of-k   JITServe: sample K replicas, ask each for the
                         request's replica-specific priority (/score) and
                         place it where that priority is highest.
  --routing round-robin  baseline placement.
  --routing least-loaded baseline placement (fewest in-flight requests).

All subrequests of a compound request (same collection_id) go to the replica
that received its first subrequest, so per-collection state stays local.

    python -m jitserve.router --replicas http://localhost:8001,http://localhost:8002 \
        --routing power-of-k --k 2 --model meta-llama/Llama-3.1-8B-Instruct
"""
import argparse
import asyncio
import itertools
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import aiohttp
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from jitserve.request_analyzer.prediction import DEFAULT_PORT, PredictionClient
from jitserve.request_info import RequestType

app = FastAPI()
args: argparse.Namespace
replicas: List[str] = []
session: Optional[aiohttp.ClientSession] = None
in_flight: Dict[str, int] = {}
placed: Dict[str, int] = {}
collection_replica: Dict[Any, str] = {}
round_robin = None
tokenizer = None
prediction_client: Optional[PredictionClient] = None
executor = ThreadPoolExecutor(max_workers=8)


async def _predict_len(prompt: str) -> Optional[int]:
    if prediction_client is None:
        return None
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, prediction_client.predict,
                                        [prompt])
    return result[0] if result else None


async def _score(replica: str, payload: Dict[str, Any]) -> Dict[str, float]:
    try:
        async with session.post(f"{replica}/score", json=payload) as resp:
            return await resp.json()
    except aiohttp.ClientError:
        return {"priority": float("-inf"), "queue_len": float("inf")}


async def _choose_replica(body: Dict[str, Any]) -> str:
    request_info = body.get("request_info", {})
    request_type = RequestType(request_info.get("request_type", 0))
    collection_id = request_info.get("collection_id")
    # ToT and DeepResearch clients number collections independently (and the
    # DeepResearch client always sends client_id 0), so tell them apart the
    # same way the server does: only DeepResearch sends requests_per_stage.
    is_deepresearch = body.get("requests_per_stage") is not None
    sticky_key = (is_deepresearch, body.get("client_id"), collection_id)
    if request_type == RequestType.COLLECTIVE and sticky_key in collection_replica:
        return collection_replica[sticky_key]

    if args.routing == "round-robin":
        replica = next(round_robin)
    elif args.routing == "least-loaded":
        replica = min(replicas, key=lambda r: in_flight[r])
    else:
        prompt = request_info.get("prompt", "")
        predicted = await _predict_len(prompt)
        if predicted is not None:
            body["predicted_output_len"] = predicted
        slo = body["slo_constraint"]
        deadline = None
        if request_type == RequestType.THROUGHPUT:
            deadline = slo[2]
        elif request_type == RequestType.COLLECTIVE:
            deadline = slo[2] * body.get("num_stages", 1)
        payload = {
            "prompt_len": len(tokenizer(prompt).input_ids) if tokenizer else len(prompt) // 4,
            "predicted_output_len": predicted or args.default_output_len,
            "request_type": int(request_type),
            "slo_constraint": slo,
            "deadline": deadline,
        }
        sampled = random.sample(replicas, min(args.k, len(replicas)))
        scores = await asyncio.gather(*(_score(r, payload) for r in sampled))
        replica = max(zip(sampled, scores),
                      key=lambda x: (x[1]["priority"], -x[1]["queue_len"]))[0]

    if request_type == RequestType.COLLECTIVE:
        collection_replica[sticky_key] = replica
    return replica


async def _log_load() -> None:
    """Print placement counts and in-flight requests per replica, for checking
    load balance offline."""
    start = asyncio.get_running_loop().time()
    while True:
        await asyncio.sleep(args.log_interval)
        t = asyncio.get_running_loop().time() - start
        print(f"[router] t={t:.0f}s placed={dict(placed)} in_flight={dict(in_flight)}",
              flush=True)


@app.get("/health")
async def health() -> Response:
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    body = await request.json()
    replica = await _choose_replica(body)
    placed[replica] += 1
    in_flight[replica] += 1
    try:
        resp = await session.post(f"{replica}/generate", json=body)
    except aiohttp.ClientError as e:
        in_flight[replica] -= 1
        return JSONResponse({"error": str(e)}, status_code=502)
    if resp.status != 200:
        content = await resp.read()
        resp.release()
        in_flight[replica] -= 1
        return Response(content=content, status_code=resp.status)

    async def relay():
        try:
            async for chunk in resp.content.iter_any():
                yield chunk
        finally:
            resp.release()
            in_flight[replica] -= 1

    return StreamingResponse(relay())


@app.on_event("startup")
async def startup() -> None:
    global session
    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
    if args.log_interval > 0:
        asyncio.create_task(_log_load())


@app.on_event("shutdown")
async def shutdown() -> None:
    await session.close()


def main() -> None:
    global args, replicas, round_robin, tokenizer, prediction_client
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--replicas", type=str, required=True,
                        help="Comma-separated replica base URLs")
    parser.add_argument("--routing",
                        choices=["power-of-k", "round-robin", "least-loaded"],
                        default="power-of-k")
    parser.add_argument("--k", type=int, default=2,
                        help="Replicas sampled per request (K=M covers all)")
    parser.add_argument("--model", type=str, default=None,
                        help="Tokenizer for prompt lengths in power-of-k mode")
    parser.add_argument("--prediction-port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--default-output-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=float, default=30.0,
                        help="Seconds between per-replica load lines (0 disables)")
    args = parser.parse_args()

    random.seed(args.seed)
    replicas = [r.rstrip("/") for r in args.replicas.split(",") if r]
    in_flight.update({r: 0 for r in replicas})
    placed.update({r: 0 for r in replicas})
    round_robin = itertools.cycle(replicas)
    if args.routing == "power-of-k":
        prediction_client = PredictionClient(port=args.prediction_port)
        if args.model:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
