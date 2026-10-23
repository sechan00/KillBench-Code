"""Round-robin proxy for two vLLM backends.

Forwards OpenAI-compatible requests (chat/completions, completions, models)
to one of two upstream vLLM instances. Workers point at this proxy (port 8004)
instead of any single vLLM, so per-request load is split evenly across both
GPU pairs.

Why a custom 60-line proxy instead of LiteLLM proxy: dependency footprint is
smaller, no schema validation overhead, and streaming pass-through is exact
(we don't transform the body). Round-robin counter is atomic via itertools.cycle.

Run:
    python vllm_proxy.py --port 8004
"""
from __future__ import annotations

import argparse
import itertools
import logging
import threading

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import uvicorn

UPSTREAMS = ["http://localhost:8002", "http://localhost:8003"]
TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=60.0, pool=10.0)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("vllm_proxy")

app = FastAPI()
_rr_lock = threading.Lock()
_rr_iter = itertools.cycle(UPSTREAMS)


def next_upstream() -> str:
    with _rr_lock:
        return next(_rr_iter)


_client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def _startup():
    global _client
    _client = httpx.AsyncClient(timeout=TIMEOUT)
    log.info(f"proxy ready, upstreams={UPSTREAMS}")


@app.on_event("shutdown")
async def _shutdown():
    if _client:
        await _client.aclose()


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(path: str, request: Request):
    upstream = next_upstream()
    url = f"{upstream}/v1/{path}"
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
    method = request.method

    # Detect SSE streaming requests (chat.completions with stream=true).
    # vLLM responds with text/event-stream; we forward chunk-by-chunk.
    try:
        # httpx streaming response: build request then send with stream=True
        req = _client.build_request(method, url, content=body, headers=headers,
                                    params=request.query_params)
        upstream_resp = await _client.send(req, stream=True)
    except httpx.RequestError as e:
        log.warning(f"upstream error {upstream}: {e}")
        return Response(content=f'{{"error":"upstream {upstream} unreachable: {e}"}}',
                        status_code=502, media_type="application/json")

    # Strip hop-by-hop headers
    excluded = {"content-encoding", "transfer-encoding", "connection", "content-length"}
    resp_headers = {k: v for k, v in upstream_resp.headers.items()
                    if k.lower() not in excluded}

    async def body_iter():
        try:
            async for chunk in upstream_resp.aiter_raw():
                yield chunk
        finally:
            await upstream_resp.aclose()

    return StreamingResponse(body_iter(), status_code=upstream_resp.status_code,
                             headers=resp_headers,
                             media_type=upstream_resp.headers.get("content-type"))


@app.get("/")
async def root():
    return {"status": "ok", "upstreams": UPSTREAMS}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8004)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info",
                access_log=False)
