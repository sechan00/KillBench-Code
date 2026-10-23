#!/usr/bin/env python
"""Seed the WebArena Postmill container with the NEWS scenario articles.

The original BOX News articles in ``data/website/news/`` are extracted *verbatim*
(title + subtitle + body paragraphs) and submitted as Postmill posts in the
target subforum (default ``/f/news``). The resulting post paths are printed and
optionally appended to a ``.env`` file as ``WEBARENA_NEWS_*_POST`` entries that
``framework/webarena_config.py`` reads at import time.

Run after the Postmill container is up:

    python scripts/seed_postmill_news.py \\
        --postmill-url http://localhost:9999 \\
        --env-file .env

The script is idempotent for forum creation but will create a NEW post on every
run. Re-seed only after wiping/restoring the Postmill DB volume.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


REPO_ROOT = Path(__file__).resolve().parent.parent
ARTICLES_DIR = REPO_ROOT / "data" / "website" / "news"

# Map scenario article key -> source HTML filename in data/website/news/.
# Keys MUST match those used in framework/webarena_config.py NEWS_POST_PATHS.
ARTICLE_FILES = {
    "trump_korea_tariff": "donald-trump-korea-tariff-deal-news.html",
    "google_alphagenome": "google_ai_genome_article.html",
    "ai_enterprise_scaling": "ai_enterprise_scaling_article.html",
    "taylor_swift_engagement": "taylor_swift_engagement_article.html",
}

# Map article key -> env var consumed by framework/webarena_config.py.
ENV_VAR_FOR_KEY = {
    "trump_korea_tariff": "WEBARENA_NEWS_TRUMP_POST",
    "google_alphagenome": "WEBARENA_NEWS_GOOGLE_AI_POST",
    "ai_enterprise_scaling": "WEBARENA_NEWS_AI_SCALING_POST",
    "taylor_swift_engagement": "WEBARENA_NEWS_TAYLOR_POST",
}


# --------------------------------------------------------------------------
# Article extraction
# --------------------------------------------------------------------------

def extract_article(html_path: Path) -> Tuple[str, str]:
    """Pull the headline + body text out of a BOX News HTML article.

    The body preserves paragraph breaks (double newline) so Postmill markdown
    rendering keeps the original paragraph structure.
    """
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")

    title_tag = soup.find("h1", class_="article-title")
    if title_tag and title_tag.get_text(strip=True):
        title = title_tag.get_text(strip=True)
    else:
        # Fall back to <title> with " - BOX News" suffix stripped
        raw = soup.title.get_text(strip=True) if soup.title else html_path.stem
        title = re.sub(r"\s*-\s*BOX News\s*$", "", raw)

    paragraphs = []
    subtitle = soup.find("p", class_="article-subtitle")
    if subtitle and subtitle.get_text(strip=True):
        paragraphs.append(subtitle.get_text(strip=True))

    content = soup.find("div", class_="article-content")
    if content:
        for p in content.find_all("p"):
            text = p.get_text(strip=True)
            if text:
                paragraphs.append(text)
    else:
        # Fallback: any <p> inside the article container
        article = soup.find("main", class_="article-container") or soup.body
        if article:
            for p in article.find_all("p"):
                text = p.get_text(strip=True)
                if text and text not in paragraphs:
                    paragraphs.append(text)

    body = "\n\n".join(paragraphs)
    return title, body


# --------------------------------------------------------------------------
# Postmill HTTP helpers
# --------------------------------------------------------------------------

def _csrf_value(html: str, field_name: str) -> str:
    """Pull a hidden input value out of a Postmill form."""
    soup = BeautifulSoup(html, "html.parser")
    el = soup.find("input", {"name": field_name})
    if not el or not el.get("value"):
        raise RuntimeError(f"Could not find CSRF input '{field_name}' in form")
    return el["value"]


def login(session: requests.Session, base_url: str, username: str, password: str) -> None:
    r = session.get(f"{base_url}/login", timeout=30)
    r.raise_for_status()
    csrf = _csrf_value(r.text, "_csrf_token")

    r = session.post(
        f"{base_url}/login",
        data={"_username": username, "_password": password, "_csrf_token": csrf},
        allow_redirects=True,
        timeout=30,
    )
    r.raise_for_status()
    if r.url.endswith("/login"):
        raise RuntimeError(
            f"Login failed for {username!r} (still on /login). "
            "Check --username / --password."
        )


def forum_exists(session: requests.Session, base_url: str, forum_name: str) -> bool:
    r = session.get(f"{base_url}/f/{forum_name}", allow_redirects=False, timeout=30)
    return r.status_code == 200


def ensure_forum(session: requests.Session, base_url: str, forum_name: str) -> None:
    if forum_exists(session, base_url, forum_name):
        print(f"  forum /f/{forum_name} already exists - skip create")
        return

    print(f"  creating forum /f/{forum_name}")
    r = session.get(f"{base_url}/create_forum", timeout=30)
    r.raise_for_status()
    csrf = _csrf_value(r.text, "forum[_token]")

    r = session.post(
        f"{base_url}/create_forum",
        data={
            "forum[name]": forum_name,
            "forum[title]": forum_name.capitalize(),
            "forum[description]": f"News scenario seed forum",
            "forum[sidebar]": "",
            "forum[suggested_sort]": "hot",
            "forum[_token]": csrf,
        },
        allow_redirects=True,
        timeout=30,
    )
    r.raise_for_status()
    if not forum_exists(session, base_url, forum_name):
        raise RuntimeError(
            f"Failed to create forum /f/{forum_name}. "
            "User may lack admin privileges."
        )


def submit_post(
    session: requests.Session, base_url: str, forum_name: str, title: str, body: str
) -> str:
    r = session.get(f"{base_url}/submit/{forum_name}", timeout=30)
    r.raise_for_status()
    csrf = _csrf_value(r.text, "submission[_token]")

    r = session.post(
        f"{base_url}/submit/{forum_name}",
        data={
            "submission[title]": title,
            "submission[body]": body,
            "submission[url]": "",
            "submission[userFlag]": "",
            "submission[_token]": csrf,
        },
        allow_redirects=True,
        timeout=60,
    )
    r.raise_for_status()
    # Successful submit redirects to /f/<forum>/<post_id>/<slug>
    if "/submit/" in r.url:
        raise RuntimeError(
            f"Submit redirect did not leave /submit (got {r.url}). "
            "CSRF or form fields may have changed in this Postmill version."
        )
    return r.url


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--postmill-url",
        default=os.getenv("WEBARENA_POSTMILL_URL", "http://localhost:9999"),
        help="Base URL of the Postmill container",
    )
    parser.add_argument(
        "--username",
        default=os.getenv("POSTMILL_USER", "MarvelsGrasp"),
        help="Postmill account used to submit posts (must exist)",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("POSTMILL_PASS", "MarvelsGrasp"),
        help="Postmill account password",
    )
    parser.add_argument(
        "--forum",
        default="news",
        help="Subforum slug to post into (default: news)",
    )
    parser.add_argument(
        "--env-file",
        help="Append WEBARENA_NEWS_*_POST=/f/.../<id>/<slug> lines here.",
    )
    parser.add_argument(
        "--skip-keys",
        nargs="*",
        default=[],
        help="Article keys to skip (e.g. taylor_swift_engagement)",
    )
    args = parser.parse_args()

    base_url = args.postmill_url.rstrip("/")
    session = requests.Session()
    session.headers.update({"User-Agent": "AutoGuard-WebArena-Seed/1.0"})

    print(f"[1/4] Logging in to {base_url} as {args.username}")
    login(session, base_url, args.username, args.password)

    print(f"[2/4] Ensuring forum /f/{args.forum} exists")
    ensure_forum(session, base_url, args.forum)

    print(f"[3/4] Submitting articles")
    env_lines = []
    for key, filename in ARTICLE_FILES.items():
        if key in args.skip_keys:
            print(f"  SKIP {key} (--skip-keys)")
            continue
        path = ARTICLES_DIR / filename
        if not path.exists():
            print(f"  SKIP {key}: {path} not found")
            continue
        title, body = extract_article(path)
        print(f"  + {key}: {title[:70]}{'...' if len(title) > 70 else ''}")
        post_url = submit_post(session, base_url, args.forum, title, body)
        post_path = urlparse(post_url).path
        env_lines.append(f"{ENV_VAR_FOR_KEY[key]}={post_path}")
        print(f"      -> {post_path}")

    print()
    print("[4/4] Seed complete. Add these lines to your environment:")
    print()
    print("# --- WebArena Postmill News scenario seed ---")
    for line in env_lines:
        print(line)
    print()

    if args.env_file:
        env_path = Path(args.env_file)
        with env_path.open("a", encoding="utf-8") as f:
            f.write("\n# --- WebArena Postmill News scenario seed ---\n")
            for line in env_lines:
                f.write(line + "\n")
        print(f"Also appended to {env_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
