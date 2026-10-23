"""Make WebArena/SafeArena per-site UI login robust under concurrency.

Problem
-------
`browsergym.webarena.instance.WebArenaInstance.ui_login` opens the site's
sign-in page and `.fill()`s the credentials using Playwright's default action
timeout (10s here). GitLab in particular (Rails, heavy pages) can take longer
than 10s to render the sign-in form when several agentlab workers reset/login
concurrently (n_jobs > 1). The `.fill()` then raises

    playwright._impl._errors.TimeoutError: Locator.fill: Timeout 10000ms
    exceeded. waiting for get_by_label("Username or email")

which aborts `env.reset()` → the task never runs (n_steps=0, ERROR row). This
is a concurrency-induced infra failure, not agent behaviour, and it pollutes
benign/harm measurements with artifactual ERROR tasks.

Fix
---
Wrap `ui_login` to (1) raise the login page's default timeout to a generous
value and (2) retry the whole login a few times with backoff. The backoff lets
the backend recover from a concurrent-reset burst, so transient overloads no
longer turn into hard task failures — they just make login a little slower.

The original creates its own login page internally
(`page = page.context.new_page()`), and a context's default timeout propagates
to pages created after it is set, so bumping `page.context.set_default_timeout`
on the passed-in page before delegating reaches the internal login page too.

Applied in the eval.py parent AND re-applied in loky worker subprocesses via
bootstrap/sitecustomize.py (workers re-import from scratch and never hit the
parent's install() call).
"""

from __future__ import annotations

import logging
import os
import time

import playwright.sync_api as pw
from browsergym.webarena.instance import WebArenaInstance

log = logging.getLogger("killswitch.login_patch")

# Tunable without redeploy.
LOGIN_TIMEOUT_MS = int(os.environ.get("KILLBENCH_LOGIN_TIMEOUT_MS", "60000"))
LOGIN_ATTEMPTS = int(os.environ.get("KILLBENCH_LOGIN_ATTEMPTS", "4"))
LOGIN_BACKOFF_S = float(os.environ.get("KILLBENCH_LOGIN_BACKOFF_S", "5"))

_PATCHED = False
_ORIG_UI_LOGIN = WebArenaInstance.ui_login


def _patched_ui_login(self, site, page):
    # Raise the default timeout on the context so the internally-created login
    # page (page.context.new_page()) inherits it. Best-effort.
    try:
        page.context.set_default_timeout(LOGIN_TIMEOUT_MS)
    except Exception:  # pragma: no cover - defensive
        pass

    last_exc = None
    for attempt in range(1, LOGIN_ATTEMPTS + 1):
        try:
            return _ORIG_UI_LOGIN(self, site, page)
        except pw.Error as e:
            last_exc = e
            wait = LOGIN_BACKOFF_S * attempt
            log.warning(
                f"login_patch: ui_login site={site} attempt "
                f"{attempt}/{LOGIN_ATTEMPTS} failed ({type(e).__name__}: "
                f"{str(e).splitlines()[0] if str(e) else e}); "
                f"retrying in {wait:.0f}s"
            )
            if attempt < LOGIN_ATTEMPTS:
                time.sleep(wait)
    # Exhausted retries — re-raise so the caller still sees a real failure
    # instead of silently proceeding logged-out.
    log.error(f"login_patch: ui_login site={site} failed after "
              f"{LOGIN_ATTEMPTS} attempts")
    raise last_exc


def install() -> None:
    """Patch WebArenaInstance.ui_login (idempotent)."""
    global _PATCHED
    if _PATCHED:
        return
    WebArenaInstance.ui_login = _patched_ui_login
    _PATCHED = True
    log.warning(
        f"login_patch: WebArenaInstance.ui_login wrapped "
        f"(timeout={LOGIN_TIMEOUT_MS}ms, attempts={LOGIN_ATTEMPTS}, "
        f"backoff={LOGIN_BACKOFF_S}s base)."
    )


def uninstall() -> None:
    global _PATCHED
    if not _PATCHED:
        return
    WebArenaInstance.ui_login = _ORIG_UI_LOGIN
    _PATCHED = False
