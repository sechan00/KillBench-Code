"""AgentLab-compatible chat adapter that talks to OpenRouter via its
OpenAI-compatible endpoint. Drop-in replacement for `OpenAIModelArgs` when
the agent's core LLM should be served through OpenRouter (e.g. Qwen3 MoE,
DeepSeek, Llama, Mistral hosted variants).

Usage:
    from openrouter_chat_adapter import prepare_openrouter
    agent_args = prepare_openrouter("qwen/qwen3.6-35b-a3b")

OpenRouter exposes an OpenAI-compatible Chat Completions API at
https://openrouter.ai/api/v1, so we reuse agentlab's `ChatModel` base class
and just swap `base_url` + api-key env var.

Auth: reads OPEN_ROUTER from the environment (matches .env key name used by
this project; agentlab's built-in OpenRouterChatModel hard-codes the
OPENROUTER_API_KEY env var which we do not use here).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx
from openai import OpenAI

from agentlab.llm.base_api import BaseModelArgs
from agentlab.llm.chat_api import ChatModel


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY_ENV = "OPEN_ROUTER"


class OpenRouterChatModel(ChatModel):
    """OpenRouter chat model via the OpenAI-compatible endpoint."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        max_retry: int = 4,
        min_retry_wait_time: float = 5,
        base_url: str | None = None,
    ):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retry=max_retry,
            min_retry_wait_time=min_retry_wait_time,
            api_key_env_var=OPENROUTER_API_KEY_ENV,
            client_class=OpenAI,
            client_args={
                "base_url": base_url or os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
                # Bound stalled streams (e.g. gemma-4-31b-it returning 200 headers
                # then hanging the body) so SDK retries instead of blocking until
                # the user sends SIGINT — that SIGINT permanently corrupts the
                # browsergym worker's Playwright event loop.
                "timeout": httpx.Timeout(120.0, connect=10.0, read=60.0),
                "max_retries": 2,
            },
            pricing_func=None,
        )
        if not self.api_key:
            raise RuntimeError(
                f"{OPENROUTER_API_KEY_ENV} not set in environment (.env). "
                "Add OPEN_ROUTER=<your-openrouter-key> to AutoGuard_ver2/.env."
            )

    def __call__(self, messages):
        msg = super().__call__(messages)
        # Reasoning models (Qwen3-thinking et al.) sometimes return content=None
        # when the response is cut off inside <think> or routed to a separate
        # `reasoning` field. agentlab's retry loop only catches ParseError, not
        # TypeError — so coerce None to "" and let parse_html_tags raise
        # ParseError which triggers a normal retry.
        if msg.get("content") is None:
            msg["content"] = ""
        return msg


@dataclass
class OpenRouterModelArgs(BaseModelArgs):
    """Serializable args for instantiating OpenRouterChatModel."""

    def make_model(self):
        return OpenRouterChatModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )


def prepare_openrouter(
    model_name: str = "qwen/qwen3.6-35b-a3b",
    max_new_tokens: int = 8192,
    max_prompt_tokens: int = 32768 - 8192,
    max_total_tokens: int = 32768,
    use_vision: bool = True,
    use_som: bool = True,
    enable_chat: bool = False,
):
    """Mirror of `prepare_claude_direct` / `prepare_gpt` / `prepare_grok`,
    routed to OpenRouter."""
    from agentlab.agents.generic_agent.agent_configs import GenericAgentArgs
    from safearena.modeling import get_default_flags

    chat_args = OpenRouterModelArgs(
        model_name=model_name,
        max_total_tokens=max_total_tokens,
        max_input_tokens=max_total_tokens - max_new_tokens,
        max_new_tokens=max_new_tokens,
        vision_support=use_vision,
    )
    flags = get_default_flags(
        max_prompt_tokens=max_prompt_tokens,
        use_screenshot=use_vision,
        use_som=use_som,
        enable_chat=enable_chat,
    )
    return GenericAgentArgs(chat_model_args=chat_args, flags=flags)
