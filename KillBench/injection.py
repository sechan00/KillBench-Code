"""Playwright-level Kill Switch injection.

The agent never sees a literal "we modified the page" signal — we simply
insert an HTML fragment at the bottom of <body> on whatever page the agent
first lands on (the task's `start_url`). The fragment is built by one of the
`payloads/` modules and may include text or an <img>.

This is *not* a persistent modification of the docker container — it lives in
the Playwright page DOM only, and disappears as soon as the agent navigates
elsewhere. Matches the experiment spec ("insert at the bottom of the main
page that the agent first accesses").
"""

from __future__ import annotations

import logging
from typing import Optional

import playwright.sync_api as pw

from payloads import Payload

log = logging.getLogger("killswitch.injection")


def inject_payload(page: pw.Page, payload: Optional[Payload]) -> bool:
    """Insert `payload.html` at the bottom of `page`'s body.

    Returns True if anything was injected, False for clean / no-op.
    """
    if payload is None or not payload.html:
        log.info("injection: no payload (clean run)")
        return False
    try:
        # Wait for body to be present before injecting. Most start_urls are
        # already loaded by the time setup() returns, but be defensive.
        page.wait_for_selector("body", timeout=5000)
        # afterbegin: insert as the FIRST child of <body> so the payload
        # appears at the top of the axtree. With 'beforeend' (body bottom),
        # agentlab's max_prompt_tokens truncation routinely clipped large
        # pages (e.g. Magento admin catalog ~280K-char axtree) before the
        # payload, leaving the agent's actual LLM input without the kill
        # switch even though it was present in the rendered DOM.
        page.evaluate(
            "(html) => { document.body.insertAdjacentHTML('afterbegin', html); }",
            payload.html,
        )
        # CSP workaround: SafeArena's Postmill ships `style-src 'self'
        # 'unsafe-inline' 'sha256-...'`. Per CSP spec, once a hash is present,
        # 'unsafe-inline' is ignored — so the `style="..."` attribute on our
        # injected element is silently dropped (HTML attribute is kept, but
        # the CSSOM never parses it; `getComputedStyle()` returns defaults).
        # Workaround: after insertion, re-apply the style declarations via the
        # CSSOM (element.style.setProperty), which is not subject to the
        # style-src inline check. Applies recursively to every descendant that
        # also has a style attribute (e.g. the inner <img>).
        page.evaluate(
            """
            () => {
              const roots = document.querySelectorAll('[id^="ks-"]');
              const reapply = (el) => {
                const sa = el.getAttribute('style');
                if (!sa) return;
                el.removeAttribute('style');
                for (const decl of sa.split(';')) {
                  const i = decl.indexOf(':');
                  if (i < 1) continue;
                  el.style.setProperty(
                    decl.slice(0, i).trim(),
                    decl.slice(i + 1).trim()
                  );
                }
              };
              for (const root of roots) {
                reapply(root);
                root.querySelectorAll('[style]').forEach(reapply);
              }
            }
            """
        )
        # Verify the marker landed in the DOM. If insertAdjacentHTML silently
        # failed (e.g. CSP, sandbox), this returns False and we know to look
        # at the page log rather than chasing post-inject DOM-wipe.
        marker_present = page.evaluate(
            "() => !!document.querySelector('[id^=\"ks-\"]')"
        )
        log.info(
            f"injection: payload={payload.name} scenario={payload.scenario} "
            f"bytes={len(payload.html)} url={page.url} "
            f"marker_present={marker_present}"
        )
        return True
    except Exception as e:  # pragma: no cover — defensive, log don't crash
        log.warning(f"injection failed for {payload.name}: {e}")
        return False
