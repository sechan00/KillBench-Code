"""Monkey-patch BrowserGym's `_pre_extract` to bound DOM-marking with a hard
timeout. Without this patch, a Playwright `frame.evaluate()` on a detached or
context-destroyed frame waits forever — see the hang we saw on harm.56 where
SafeArena's `program_html` evaluator opened a new tab, closed it, and left the
main frame in a state where the next observation extraction blocked on
`frame.evaluate(js_frame_mark_elements, ...)` for 1.5 h.

We use SIGALRM (Unix-only, main-thread) to interrupt the blocking syscall.
When the alarm fires we raise a `playwright.sync_api.TimeoutError` so the
existing retry handler at `browsergym.core.env._get_obs` (which already catches
`playwright.sync_api.Error`) can record it as `err_msg` and move on.
"""

from __future__ import annotations

import logging
import os
import signal
import threading

import playwright.sync_api as pw
import browsergym.core.observation as bg_obs

log = logging.getLogger("killswitch.timeout_patch")

# Configurable via env var so we don't have to redeploy to retune.
DEFAULT_TIMEOUT_S = int(os.environ.get("KILLBENCH_OBS_TIMEOUT_S", "45"))

_PATCHED = False
_ORIG_PRE_EXTRACT = bg_obs._pre_extract


def _patched_pre_extract(page, *args, **kwargs):
    # signal.alarm only works on the main thread on Unix. On non-main threads
    # we fall back to the original (no timeout).
    if threading.current_thread() is not threading.main_thread():
        return _ORIG_PRE_EXTRACT(page, *args, **kwargs)

    def handler(signum, frame):
        # Raise a Playwright TimeoutError so browsergym's existing retry +
        # err_msg path handles it (it catches playwright.sync_api.Error which
        # TimeoutError is a subclass of).
        raise pw.TimeoutError(
            f"killswitch.timeout_patch: _pre_extract exceeded "
            f"{DEFAULT_TIMEOUT_S}s; treating as detached frame, skipping task."
        )

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(DEFAULT_TIMEOUT_S)
    try:
        return _ORIG_PRE_EXTRACT(page, *args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def install() -> None:
    """Patch BrowserGym (idempotent)."""
    global _PATCHED
    if _PATCHED:
        return
    bg_obs._pre_extract = _patched_pre_extract
    _PATCHED = True
    log.warning(
        f"timeout_patch: _pre_extract wrapped with SIGALRM timeout "
        f"= {DEFAULT_TIMEOUT_S}s. Hangs become TimeoutError → task is skipped."
    )


def uninstall() -> None:
    global _PATCHED
    if not _PATCHED:
        return
    bg_obs._pre_extract = _ORIG_PRE_EXTRACT
    _PATCHED = False
