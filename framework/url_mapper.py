"""Translate scenario fake URLs into WebArena docker URLs (and back).

Attack prompts and Kill Switch payloads still use the original scenario domains
(``www.TechUniversity.edu``, ``BOXNews.com``, ``TechMall.com``). At runtime the
agent's navigation tool calls into ``map_to_webarena_url`` to fetch from the
real docker container. The display URL shown back to the agent is rewritten
with ``rewrite_response_url_to_fake`` so the agent stays on the scenario URL
namespace and follows the same hyperlinks it has been instructed to follow.
"""
from __future__ import annotations

from typing import Optional, Tuple
from urllib.parse import urlparse

from framework.webarena_config import (
    SCENARIO_BASE_URL_MAP,
    SCENARIO_PATH_MAP,
)


def _detect_fake_domain(url: str) -> Optional[str]:
    """Return the fake-domain key whose substring appears in ``url`` (lower)."""
    lowered = url.lower()
    # Longest match wins so 'abcnews.com' doesn't shadow a more specific key.
    for fake_domain in sorted(SCENARIO_BASE_URL_MAP.keys(), key=len, reverse=True):
        if fake_domain in lowered:
            return fake_domain
    return None


def _extract_path(url: str) -> str:
    """Best-effort path extraction tolerant of bare-host inputs like 'TechMall.com'."""
    candidate = url if url.startswith(("http://", "https://")) else f"https://{url}"
    parsed = urlparse(candidate)
    path = parsed.path or "/"
    # urlparse strips fragments but keep query-string append for compat
    if parsed.query:
        path = f"{path}?{parsed.query}"
    return path


def map_to_webarena_url(url: str) -> Tuple[Optional[str], Optional[str]]:
    """Translate a fake scenario URL into a real WebArena docker URL.

    Returns ``(docker_url, fake_domain)`` on success, or ``(None, None)`` if
    ``url`` does not belong to any registered scenario (caller should fall back
    to its existing local-file or pass-through behaviour).
    """
    fake_domain = _detect_fake_domain(url)
    if fake_domain is None:
        return None, None

    fake_path = _extract_path(url)
    # Strip query for prefix matching; reattach later
    query = ""
    if "?" in fake_path:
        fake_path, query = fake_path.split("?", 1)
        query = "?" + query

    path_map = SCENARIO_PATH_MAP.get(fake_domain, {})
    docker_target: Optional[str] = None
    for fake_p, docker_p in sorted(path_map.items(), key=lambda kv: -len(kv[0])):
        if fake_path == fake_p or fake_path.startswith(fake_p.rstrip("/") + "/"):
            docker_target = docker_p
            break

    if docker_target is None:
        # No explicit remap: pass the path through onto the docker base.
        docker_target = fake_path

    if docker_target.startswith(("http://", "https://")):
        return docker_target + query, fake_domain

    docker_base = SCENARIO_BASE_URL_MAP[fake_domain]
    if not docker_target.startswith("/"):
        docker_target = "/" + docker_target
    return f"{docker_base}{docker_target}{query}", fake_domain


def rewrite_response_url_to_fake(real_url: str, fake_domain: str) -> str:
    """Inverse of :func:`map_to_webarena_url` for a single URL string.

    Used when echoing the ``Current URL`` back to the agent so the agent keeps
    operating in the fake-URL namespace defined by the attack prompt.
    """
    docker_base = SCENARIO_BASE_URL_MAP.get(fake_domain)
    if docker_base and real_url.startswith(docker_base):
        suffix = real_url[len(docker_base):] or "/"
        return f"https://{fake_domain}{suffix}"
    return real_url


def is_scenario_url(url: str) -> bool:
    """Return True if ``url`` matches any registered scenario fake domain."""
    return _detect_fake_domain(url) is not None
