"""Patch agentlab's ChatModel to support OpenAI reasoning models.

Reasoning models (gpt-5*, o1*, o3*, o4*) reject `temperature` and require
`max_completion_tokens` instead of `max_tokens`. agentlab's stock
`ChatModel.__call__` (chat_api.py:264) always passes both → every call
raises 400 BadRequest, which agentlab's outer retry loop surfaces as
"Connection error" and retries indefinitely with 60s backoff.

This module monkey-patches `ChatModel.__call__` to detect reasoning models
by name prefix and route them through a parameter-compatible call. Non-
reasoning models keep the original code path unchanged.

Idempotent: `install()` can be called from both the parent process and from
sitecustomize.py in worker subprocesses without re-patching.
"""
from __future__ import annotations

import logging

log = logging.getLogger("killswitch.openai_reasoning_patch")

_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")
_INSTALLED = False


def is_reasoning_model(name: str) -> bool:
    n = (name or "").lower()
    return any(n.startswith(p) for p in _REASONING_PREFIXES)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    try:
        import openai
        from agentlab.llm import tracking
        from agentlab.llm.chat_api import (
            ChatModel,
            OpenRouterError,
            RetryError,
            handle_error,
            make_assistant_message,
        )
    except ImportError as e:
        log.warning(f"openai_reasoning_patch: agentlab not importable: {e}")
        return

    _ORIGINAL_CALL = ChatModel.__call__

    def _patched_call(self, messages):
        if not is_reasoning_model(self.model_name):
            return _ORIGINAL_CALL(self, messages)

        # Reasoning models: omit temperature, use max_completion_tokens.
        self.retries = 0
        self.success = False
        self.error_types = []

        completion = None
        error_type = None
        for itr in range(self.max_retry):
            self.retries += 1
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_completion_tokens=self.max_tokens,
                )
                if completion.usage is None:
                    raise OpenRouterError(
                        "The completion object does not contain usage information."
                    )
                self.success = True
                break
            except openai.OpenAIError as e:
                error_type = handle_error(e, itr, self.min_retry_wait_time, self.max_retry)
                self.error_types.append(error_type)

        if not completion:
            raise RetryError(
                f"Failed to get a response from the API after {self.max_retry} retries\n"
                f"Last error: {error_type}"
            )

        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        cost = input_tokens * self.input_cost + output_tokens * self.output_cost

        if hasattr(tracking.TRACKER, "instance") and isinstance(
            tracking.TRACKER.instance, tracking.LLMTracker
        ):
            tracking.TRACKER.instance(input_tokens, output_tokens, cost)

        return make_assistant_message(completion.choices[0].message.content)

    ChatModel.__call__ = _patched_call
    _INSTALLED = True
    log.info(
        "openai_reasoning_patch: ChatModel.__call__ patched "
        f"(reasoning prefixes={_REASONING_PREFIXES})"
    )
