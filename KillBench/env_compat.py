"""Environment compatibility shims for the AutoGuard conda env.

Two shims, both idempotent and safe to call multiple times:

1. **langchain.schema → langchain_classic.schema**
   `agentlab.llm.llm_utils` does `from langchain.schema import BaseMessage`,
   which existed in langchain<1.0 but was removed in 1.x. The conda env has
   langchain 1.1.3 + langchain_classic 1.0.0 (which preserves the old API).
   We alias the legacy module name so agentlab can import.

2. **agentlab.experiments.reproducibility_util._get_repo stub**
   The reproducibility tracker insists on opening a git repo at
   `agentlab.__file__`'s parent, but pip-installed agentlab is not a git
   checkout (and no ancestor directory is one either). We monkey-patch
   `_get_repo` to return a stub when the real lookup fails — downstream
   callers (`_get_git_username`, `_get_git_info`) already swallow exceptions.

Call `install()` once before any code path that imports `safearena.modeling`
or `agentlab.experiments.study`.
"""
from __future__ import annotations

import logging
import sys

log = logging.getLogger("killswitch.env_compat")

_INSTALLED = False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    # ---- (1) langchain.schema shim ----
    try:
        import langchain.schema  # noqa: F401
    except ImportError:
        try:
            import langchain_classic.schema as _lc_schema
            sys.modules["langchain.schema"] = _lc_schema
            log.info("env_compat: aliased langchain.schema -> langchain_classic.schema")
        except ImportError:
            log.warning("env_compat: neither langchain.schema nor "
                        "langchain_classic.schema importable")

    # ---- (2) _get_repo stub for non-git agentlab install ----
    try:
        from agentlab.experiments import reproducibility_util as _ru
        from git import InvalidGitRepositoryError, NoSuchPathError

        _orig_get_repo = _ru._get_repo

        class _StubRepo:
            """Minimal stub that raises InvalidGitRepositoryError on ANY
            attribute access. `_get_git_username` catches Exception inside
            its try blocks and returns None; `_get_git_info` catches
            InvalidGitRepositoryError and returns `(None, [])`. Both paths
            therefore degrade gracefully."""
            def __getattr__(self, name):
                raise InvalidGitRepositoryError(
                    f"no git repo (stub access: {name})"
                )

        def _patched_get_repo(module):
            try:
                return _orig_get_repo(module)
            except (InvalidGitRepositoryError, NoSuchPathError, Exception):
                return _StubRepo()

        _ru._get_repo = _patched_get_repo
        log.info("env_compat: patched reproducibility_util._get_repo (git-tolerant)")
    except ImportError:
        log.warning("env_compat: agentlab.experiments.reproducibility_util "
                    "not importable; skipping git stub")

    _INSTALLED = True
