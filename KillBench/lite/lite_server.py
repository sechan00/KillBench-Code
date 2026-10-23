"""Background HTTP server that exposes a Lite scenario website directory
(e.g. `AutoGuard_ver2/data/website/PII/`) over localhost so a Playwright
browser can navigate it.

Why a real HTTP server (not file://): Playwright's accessibility tree
extraction misbehaves on `file://` URLs (no proper origin, some link
behaviors break). A trivial `http.server` serving the static HTML directory
is the path of least resistance and lets the Lite HTML's absolute links
(`<a href="/about">`) resolve correctly.

Lifecycle: one server per scenario, started lazily on first request,
shut down at process exit. Threading-based — non-blocking, no event loop.
"""

from __future__ import annotations

import atexit
import logging
import os
import socket
import threading
from functools import lru_cache
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Dict, Optional

log = logging.getLogger("killbench_lite.server")

# Scenario name (lowercase, used as URL slug + dict key) → directory under
# AutoGuard_ver2/data/website/ that holds index.html and friends.
# lite/ now nests under KillBench/, so repo root is three levels up.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_WEBSITE_BY_SCENARIO: Dict[str, Path] = {
    "pii":  _REPO_ROOT / "data" / "website" / "PII",
    "news": _REPO_ROOT / "data" / "website" / "news",
    "hack": _REPO_ROOT / "data" / "website" / "webhack",
}

# scenario → (host, port). Populated lazily on first start_server call.
_RUNNING: Dict[str, tuple] = {}
_RUNNING_LOCK = threading.Lock()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_handler(directory: Path):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        # Lite HTML has links like /about (no .html). Map them to about.html
        # on disk before SimpleHTTPRequestHandler does the file lookup.
        def translate_path(self, path: str) -> str:
            # Strip query and fragment
            p = path.split("?", 1)[0].split("#", 1)[0]
            if p == "" or p == "/":
                p = "/index.html"
            elif not p.endswith("/") and "." not in os.path.basename(p):
                # /about → /about.html (if it exists)
                candidate = (directory / p.lstrip("/")).with_suffix(".html")
                if candidate.exists():
                    p = "/" + candidate.name
            return super().translate_path(p)

        def log_message(self, fmt, *args):  # quiet
            log.debug("http %s - " + fmt, self.address_string(), *args)

    return Handler


def start_server(scenario: str) -> str:
    """Start (or reuse) the HTTP server for `scenario`. Returns the base URL.

    Worker-subprocess path: if `LITE_URL_<SCENARIO>` is set in the env
    (parent process exported it before spawning workers), do NOT start a
    local server — return the parent's URL so workers and parent agree
    on which HTTP endpoint the [URL]-substituted prompts point at.

    Idempotent: a second call with the same scenario returns the cached URL.
    """
    scenario = scenario.lower()
    if scenario not in _WEBSITE_BY_SCENARIO:
        raise ValueError(
            f"Unknown scenario {scenario!r}; expected one of "
            f"{sorted(_WEBSITE_BY_SCENARIO)}"
        )
    env_url = os.environ.get(f"LITE_URL_{scenario.upper()}")
    if env_url:
        # Worker: reuse parent's server URL. Parent is the listener.
        return env_url
    with _RUNNING_LOCK:
        if scenario in _RUNNING:
            host, port = _RUNNING[scenario]
            return f"http://{host}:{port}"

        directory = _WEBSITE_BY_SCENARIO[scenario]
        if not directory.is_dir():
            raise FileNotFoundError(f"Website directory missing: {directory}")

        port = _free_port()
        host = "127.0.0.1"
        httpd = HTTPServer((host, port), _make_handler(directory))
        t = threading.Thread(
            target=httpd.serve_forever,
            name=f"lite-http-{scenario}",
            daemon=True,
        )
        t.start()
        _RUNNING[scenario] = (host, port)
        atexit.register(lambda: httpd.shutdown())
        log.info(f"lite_server: serving {scenario!r} from {directory} at http://{host}:{port}")
        return f"http://{host}:{port}"


@lru_cache(maxsize=8)
def start_url_for(scenario: str) -> str:
    """Convenience: full URL to the scenario's index page."""
    base = start_server(scenario)
    return f"{base}/index.html"
