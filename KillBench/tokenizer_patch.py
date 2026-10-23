"""Monkey-patch agentlab.get_tokenizer to fall back to gpt-4 tiktoken.

Background
----------
agentlab's `GenericAgent.get_action()` calls `dp.fit_tokens()` every step,
which calls `count_tokens(prompt, model=model_name)` -> `get_tokenizer(model)`.

For models not in tiktoken's catalogue (Gemma, Qwen, Llama, ...), agentlab
falls back to `AutoTokenizer.from_pretrained(model_name)` and only catches
`OSError`. transformers 4.57.3 has an internal bug:

    PreTrainedTokenizerFast.__init__
      -> _set_model_specific_special_tokens(special_tokens: list[str])
         -> list(special_tokens.keys())     # expects dict, gets list
         AttributeError: 'list' object has no attribute 'keys'

Gemma's fast tokenizer supplies `extra_special_tokens` as a list, so
loading any `google/gemma-*` model via HF AutoTokenizer crashes here.
The AttributeError isn't an OSError, so agentlab doesn't catch it and
the exception propagates up to BrowserGym's task-loop, aborting every
step of every task with ERROR.

Fix
---
Replace `agentlab.llm.llm_utils.get_tokenizer` with a version that
catches **any** exception during HF tokenizer init and falls back to
the gpt-4 tiktoken encoding. Token counts here are only used to size
the prompt budget; a slightly different tokenizer is acceptable.

Applied via `install()` from the parent process (eval.py) and from
sitecustomize.py in worker subprocesses. Idempotent.
"""

from __future__ import annotations

import logging
from functools import cache

logger = logging.getLogger(__name__)

_PATCHED_ATTR = "_killswitch_tokenizer_patch_applied"


def install() -> None:
    """Patch `agentlab.llm.llm_utils.get_tokenizer` with a broader fallback."""
    import tiktoken
    from agentlab.llm import llm_utils

    if getattr(llm_utils, _PATCHED_ATTR, False):
        return

    @cache
    def get_tokenizer(model_name: str = "gpt-4"):
        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            pass
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(model_name)
        except Exception as exc:
            logger.info(
                "tokenizer_patch: HF tokenizer load failed for %r (%s: %s); "
                "falling back to gpt-4 tiktoken",
                model_name, type(exc).__name__, exc,
            )
        return tiktoken.encoding_for_model("gpt-4")

    llm_utils.get_tokenizer = get_tokenizer

    # count_tokens robustness: tiktoken's Encoding.encode() defaults to
    # disallowed_special="all", so any page text that happens to contain a
    # special-token string (e.g. an adversarial GCG payload with
    # "<|fim_suffix|>") raises ValueError and aborts every step of every task.
    # Token counts here only size the prompt budget, so treat such substrings
    # as ordinary text (disallowed_special=()). HF tokenizers don't take that
    # kwarg — fall back to a plain encode for them.
    def count_tokens(text, model="openai/gpt-4"):
        enc = get_tokenizer(model)
        try:
            return len(enc.encode(text, disallowed_special=()))
        except TypeError:
            return len(enc.encode(text))

    llm_utils.count_tokens = count_tokens
    # dynamic_prompting binds `count_tokens` at import time
    # (`from ...llm_utils import count_tokens`), so patch that reference too.
    try:
        from agentlab.agents import dynamic_prompting as _dp
        _dp.count_tokens = count_tokens
    except Exception as exc:  # pragma: no cover
        logger.info("tokenizer_patch: could not patch dynamic_prompting.count_tokens (%s)", exc)

    # browsergym has its OWN count_tokens (browsergym.experiments.utils) used by
    # loop.py to compute per-observation n_token_* stats every step. It also
    # calls tiktoken.encode() with the default special-token guard and crashes
    # on adversarial payload text (<|fim_suffix|>). Patch it (and count_messages_
    # token resolves count_tokens by module-global name, so it's covered too).
    def bg_count_tokens(text, model="gpt-4"):
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.encoding_for_model("gpt-4")
        return len(enc.encode(text, disallowed_special=()))

    try:
        from browsergym.experiments import utils as _bg_utils
        _bg_utils.count_tokens = bg_count_tokens
        try:
            from browsergym.experiments import loop as _bg_loop
            if hasattr(_bg_loop, "count_tokens"):
                _bg_loop.count_tokens = bg_count_tokens
        except Exception:
            pass
    except Exception as exc:  # pragma: no cover
        logger.info("tokenizer_patch: could not patch browsergym count_tokens (%s)", exc)

    setattr(llm_utils, _PATCHED_ATTR, True)
    logger.info(
        "tokenizer_patch installed: get_tokenizer falls back to gpt-4 "
        "tiktoken on any HF init exception; count_tokens tolerates special tokens"
    )


__all__ = ["install"]
