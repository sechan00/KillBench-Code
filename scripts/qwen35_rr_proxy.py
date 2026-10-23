"""Round-robin proxy that fronts 4 llama-cpp-python instances of
Qwen3.5-35B-uncensored (ports 8003-8006) as a single OpenAI-compatible
endpoint on port 8010.

eval.py only knows one OPENAI_API_BASE; this proxy distributes inbound
chat-completions requests across the four backends so n-jobs=8 actually
parallelizes (llama_cpp.server serializes per-instance requests).

Usage:
    python scripts/qwen35_rr_proxy.py
        [--port 8010] [--upstreams 8003,8004,8005,8006]
"""
import argparse
import asyncio
import itertools
import logging
import sys

import aiohttp
from aiohttp import web

log = logging.getLogger("qwen35_rr_proxy")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--upstreams", default="8003,8004,8005,8006")
    ap.add_argument("--host", default="127.0.0.1")
    return ap.parse_args()


def make_app(upstream_ports: list[int]) -> web.Application:
    rr = itertools.cycle(upstream_ports)
    lock = asyncio.Lock()
    session: aiohttp.ClientSession | None = None

    async def on_startup(app):
        nonlocal session
        # Long timeout — generation can take minutes for 16k-context tasks.
        timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
        session = aiohttp.ClientSession(timeout=timeout)

    async def on_cleanup(app):
        await session.close()

    async def handler(request: web.Request) -> web.StreamResponse:
        async with lock:
            port = next(rr)
        upstream = f"http://127.0.0.1:{port}{request.rel_url}"
        body = await request.read()
        # Strip hop-by-hop and Host headers so aiohttp sets them correctly.
        fwd_headers = {k: v for k, v in request.headers.items()
                       if k.lower() not in {"host", "content-length",
                                             "authorization"}}
        # Force the backend API key (eval.py loads .env with override=True,
        # which replaces our local-uncensored key with the real OpenAI key
        # in the SDK's outbound Authorization header — re-stamp it here).
        fwd_headers["Authorization"] = "Bearer local-uncensored"
        try:
            async with session.request(
                request.method, upstream, data=body, headers=fwd_headers
            ) as r:
                content = await r.read()
                # Echo through status, content-type, and pass-through body.
                resp = web.Response(
                    status=r.status,
                    body=content,
                    content_type=r.headers.get("content-type", "application/json"),
                )
                return resp
        except Exception as e:
            log.warning(f"upstream {upstream} error: {e}")
            return web.json_response(
                {"error": {"message": f"upstream {port} error: {e}",
                          "type": "proxy_error"}},
                status=502,
            )

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_route("*", "/{tail:.*}", handler)
    return app


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()
    ports = [int(p) for p in args.upstreams.split(",")]
    log.info(f"round-robin: {args.host}:{args.port} → {ports}")
    web.run_app(make_app(ports), host=args.host, port=args.port,
                access_log=None)


if __name__ == "__main__":
    sys.exit(main())
