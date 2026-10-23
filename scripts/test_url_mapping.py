#!/usr/bin/env python
"""Smoke-test the WebArena URL mapper without booting any docker containers.

Verifies that every URL pattern that appears in the attack prompts maps to a
sensible WebArena docker URL, and that ``rewrite_response_url_to_fake`` is the
inverse on the configured base.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make framework/ importable when running via `python scripts/test_url_mapping.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from framework.url_mapper import (  # noqa: E402
    map_to_webarena_url,
    rewrite_response_url_to_fake,
    is_scenario_url,
)
from framework.webarena_config import (  # noqa: E402
    GITLAB_BASE,
    POSTMILL_BASE,
    SHOPPING_BASE,
)


CASES = [
    # (input, expected docker prefix, expected fake_domain)
    # ----- PII -----
    ("https://www.TechUniversity.edu/members", GITLAB_BASE + "/members.html", "techuniversity.edu"),
    ("www.TechUniversity.edu", GITLAB_BASE + "/members.html", "techuniversity.edu"),
    ("https://TechUniversity.edu/about", GITLAB_BASE + "/help", "techuniversity.edu"),
    ("TechUniversity.edu/programs", GITLAB_BASE + "/explore/projects", "techuniversity.edu"),
    # ----- NEWS -----
    ("https://BOXNews.com/", POSTMILL_BASE + "/f/news", "boxnews.com"),
    ("https://BOXNews.com/main_news", POSTMILL_BASE, "boxnews.com"),
    ("https://BOXNews.com/trump_main_news", POSTMILL_BASE, "boxnews.com"),
    ("https://BOXNews.com/google_ai_genome_article", POSTMILL_BASE, "boxnews.com"),
    # ----- HACK -----
    ("https://TechMall.com/", SHOPPING_BASE + "/", "techmall.com"),
    ("https://TechMall.com/products", SHOPPING_BASE + "/category-recommendations.html", "techmall.com"),
    ("http://TechMall.com/cart.html", SHOPPING_BASE + "/checkout/cart", "techmall.com"),
    ("TechMall.com/about", SHOPPING_BASE + "/about-us", "techmall.com"),
    # ----- Non-scenario URL passes through (None, None) -----
    ("https://example.com/foo", None, None),
    ("https://google.com", None, None),
]


def main() -> int:
    fails = 0
    for raw, expected_prefix, expected_domain in CASES:
        docker_url, fake_domain = map_to_webarena_url(raw)

        if expected_prefix is None:
            ok = docker_url is None and fake_domain is None
            label = "PASS" if ok else "FAIL"
            print(f"[{label}] {raw!r:60s} -> ({docker_url!r}, {fake_domain!r})  expected (None, None)")
        else:
            ok = (
                docker_url is not None
                and docker_url.startswith(expected_prefix)
                and fake_domain == expected_domain
            )
            label = "PASS" if ok else "FAIL"
            print(f"[{label}] {raw!r:60s} -> {docker_url!r}")
            if not ok:
                print(f"          expected prefix {expected_prefix!r} domain {expected_domain!r}")
        if not ok:
            fails += 1

    # Round-trip: fake -> docker -> fake should reproduce the host portion
    print()
    print("Round-trip checks (fake -> docker -> fake host):")
    for raw, expected_prefix, expected_domain in CASES:
        if expected_domain is None:
            continue
        docker_url, fake_domain = map_to_webarena_url(raw)
        back = rewrite_response_url_to_fake(docker_url, fake_domain)
        ok = expected_domain in back.lower()
        label = "PASS" if ok else "FAIL"
        print(f"[{label}] {raw!r:60s} <-> {back!r}")
        if not ok:
            fails += 1

    print()
    print(f"is_scenario_url checks:")
    for raw, _, expected_domain in CASES:
        result = is_scenario_url(raw)
        expected = expected_domain is not None
        ok = result == expected
        label = "PASS" if ok else "FAIL"
        print(f"[{label}] is_scenario_url({raw!r:60s}) = {result}")
        if not ok:
            fails += 1

    print()
    if fails:
        print(f"FAILED: {fails} cases did not match.")
        return 1
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
