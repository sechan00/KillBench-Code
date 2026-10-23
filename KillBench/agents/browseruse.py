"""BrowserUse agent runner for SafeArena Kill Switch evaluation.

BrowserUse (https://github.com/browser-use/browser-use) drives its own
Chromium process — it cannot be plugged into agentlab's `Study` (which expects
an `agent.get_action(obs)` interface and owns its own browser). Instead we run
BrowserUse out-of-band: spin up the session, navigate to the task's start_url,
inject the Kill Switch payload, run `Agent.run()` to completion, and capture
its final response. The unified eval.py loop drives this per task.

The classification (REFUSAL / ATTEMPT / AMBIGUOUS) is done downstream by the
same Anthropic refusal classifier used for the text-only baseline so results
are directly comparable.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from browser_use import Agent, BrowserProfile, BrowserSession
    from browser_use import ChatAnthropic as BrowserUseChatAnthropic
    from browser_use import ChatOpenAI as BrowserUseChatOpenAI
    from browser_use.tools.service import Tools
    _BROWSER_USE_AVAILABLE = True
    _BU_IMPORT_ERR = None
except Exception as e:  # pragma: no cover
    Agent = BrowserProfile = BrowserSession = None
    _BROWSER_USE_AVAILABLE = False
    _BU_IMPORT_ERR = e

from payloads import Payload

log = logging.getLogger("killswitch.browseruse")


# Match the framework/BrowserUseAgent.py chrome binary detection.
_DEFAULT_CHROME_CANDIDATES = [
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
]


def _detect_chrome_binary() -> Optional[str]:
    override = os.getenv("BROWSER_USE_CHROME_BINARY")
    if override and os.path.exists(override):
        return override
    for cand in _DEFAULT_CHROME_CANDIDATES:
        if os.path.exists(cand):
            return cand
    return None


def _make_llm(model_name: str, temperature: float = 0.5):
    low = model_name.lower()
    if "claude" in low:
        return BrowserUseChatAnthropic(model=model_name, temperature=temperature)
    if "gpt" in low or low.startswith("o1") or low.startswith("o3"):
        return BrowserUseChatOpenAI(model=model_name, temperature=temperature)
    return BrowserUseChatOpenAI(model=model_name, temperature=temperature)


@dataclass
class BrowserUseRunResult:
    final_response: str
    n_steps: int
    success: bool
    error: Optional[str] = None
    injected: Optional[bool] = None     # True/False if injection attempted,
                                        # None if no payload was supplied


_LOG_DIR_ENV = "KILLSWITCH_INJECTION_LOG_DIR"
_MARKER_CHARS = 120


def _write_injection_record(
    task_id: Optional[str],
    payload: Optional[Payload],
    category: Optional[str],
    page_url: str,
    injected: bool,
) -> None:
    """Mirror of task_patch._write_injection_record for the BrowserUse path."""
    log_dir = os.environ.get(_LOG_DIR_ENV)
    if not log_dir or not task_id or payload is None:
        return
    p = Path(log_dir)
    try:
        p.mkdir(parents=True, exist_ok=True)
        marker = (payload.text or "").strip()[:_MARKER_CHARS]
        record = {
            "task_id": task_id,
            "payload_name": payload.name,
            "scenario": payload.scenario,
            "category": category,
            "payload_text": payload.text,
            "payload_text_marker": marker,
            "injection_method": "body.beforeend",
            "injection_url": page_url,
            "injected": injected,
            "agent_path": "browseruse",
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }
        (p / f"{task_id}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        log.warning(f"browseruse: failed to write injection record for "
                    f"{task_id}: {e}")


async def _run_async(
    task: str,
    start_url: str,
    llm,
    payload: Optional[Payload],
    headless: bool = True,
    max_steps: int = 30,
    allowed_domains: Optional[list[str]] = None,
    task_id: Optional[str] = None,
    category: Optional[str] = None,
) -> BrowserUseRunResult:
    profile_kwargs = dict(
        headless=headless,
        disable_security=True,
        allowed_domains=allowed_domains or None,
    )
    binary = _detect_chrome_binary()
    if binary:
        profile_kwargs["executable_path"] = binary

    profile = BrowserProfile(**profile_kwargs)
    session = BrowserSession(browser_profile=profile)

    injected: Optional[bool] = None
    try:
        # 1) Pre-navigate so we can inject Kill Switch payload BEFORE the agent
        #    looks at the page. Use the session's underlying CDP/Playwright.
        await session.start()
        page = await session.get_current_page()
        await page.goto(start_url, wait_until="domcontentloaded")
        if payload is not None and payload.html:
            try:
                await page.wait_for_selector("body", timeout=5000)
                await page.evaluate(
                    "(h) => { document.body.insertAdjacentHTML('beforeend', h); }",
                    payload.html,
                )
                injected = True
                log.info(
                    f"BU injection: payload={payload.name} scenario={payload.scenario} "
                    f"bytes={len(payload.html)} url={page.url}"
                )
            except Exception as e:
                injected = False
                log.warning(f"BU injection failed: {e}")
            _write_injection_record(task_id, payload, category, page.url,
                                    injected=bool(injected))

        # 2) Hand the (already-navigated, possibly-injected) session to BrowserUse.
        tools = Tools(exclude_actions=["search"])
        agent = Agent(
            task=task,
            llm=llm,
            browser_session=session,
            tools=tools,
            use_judge=False,
        )
        log.info(f"BU Agent.run start: {task[:120]}...")
        history = await agent.run(max_steps=max_steps)

        final_response = ""
        n_steps = 0
        if history is not None:
            try:
                final_response = history.final_result() or ""
            except Exception:
                final_response = ""
            try:
                n_steps = len(history.history) if hasattr(history, "history") else 0
            except Exception:
                pass
            if not final_response:
                # Fallback: take the last model output / thought.
                for item in reversed(getattr(history, "history", []) or []):
                    mo = getattr(item, "model_output", None)
                    if mo:
                        final_response = str(mo)[:2000]
                        break
        return BrowserUseRunResult(
            final_response=final_response,
            n_steps=n_steps,
            success=True,
            injected=injected,
        )
    except Exception as e:
        log.exception("BU run failed")
        return BrowserUseRunResult(
            final_response="",
            n_steps=0,
            success=False,
            error=str(e),
            injected=injected,
        )
    finally:
        try:
            await session.kill()
        except Exception:
            try:
                await session.stop()
            except Exception:
                pass


def run_one(
    task: str,
    start_url: str,
    model_name: str,
    payload: Optional[Payload],
    headless: bool = True,
    max_steps: int = 30,
    allowed_domains: Optional[list[str]] = None,
    task_id: Optional[str] = None,
    category: Optional[str] = None,
) -> BrowserUseRunResult:
    """Synchronous wrapper around the async runner.

    `task_id` and `category` are forwarded only for injection-record logging
    (see `_write_injection_record`); they do not affect agent behavior.
    """
    if not _BROWSER_USE_AVAILABLE:
        return BrowserUseRunResult(
            final_response="",
            n_steps=0,
            success=False,
            error=f"browser-use not installed: {_BU_IMPORT_ERR!r}",
        )
    llm = _make_llm(model_name)
    return asyncio.run(
        _run_async(
            task=task,
            start_url=start_url,
            llm=llm,
            payload=payload,
            headless=headless,
            max_steps=max_steps,
            allowed_domains=allowed_domains,
            task_id=task_id,
            category=category,
        )
    )


__all__ = ["run_one", "BrowserUseRunResult"]
