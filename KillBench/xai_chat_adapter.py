"""AgentLab-compatible chat adapter that talks to xAI (Grok) via its
OpenAI-compatible endpoint. Drop-in replacement for `OpenAIModelArgs` when
the agent's core LLM should be Grok rather than GPT / Claude.

Usage:
    from xai_chat_adapter import prepare_grok
    agent_args = prepare_grok("grok-4-latest")

xAI exposes an OpenAI-compatible Chat Completions API at https://api.x.ai/v1,
so we reuse agentlab's `ChatModel` base class and just swap `base_url` +
api-key env var. Message format (including vision `image_url` blocks emitted
by BrowserGym for screenshots) is identical to OpenAI's.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI

from agentlab.llm.base_api import BaseModelArgs
from agentlab.llm.chat_api import ChatModel


XAI_BASE_URL = "https://api.x.ai/v1"
XAI_API_KEY_ENV = "XAI_API_KEY"


class XAIChatModel(ChatModel):
    """Grok chat model via xAI's OpenAI-compatible endpoint."""

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
            api_key_env_var=XAI_API_KEY_ENV,
            client_class=OpenAI,
            client_args={"base_url": base_url or os.getenv("XAI_BASE_URL", XAI_BASE_URL)},
            pricing_func=None,
        )
        if not self.api_key:
            raise RuntimeError(f"{XAI_API_KEY_ENV} not set")


@dataclass
class XAIModelArgs(BaseModelArgs):
    """Serializable args for instantiating XAIChatModel."""

    def make_model(self):
        return XAIChatModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )


def prepare_grok(
    model_name: str = "grok-4-latest",
    max_new_tokens: int = 1024,
    max_prompt_tokens: int = 16384 - 4096,
    max_total_tokens: int = 16384,
    use_vision: bool = True,
    use_som: bool = True,
    enable_chat: bool = False,
):
    """Mirror of `prepare_claude_direct` / `prepare_gpt`, but routed to xAI."""
    from agentlab.agents.generic_agent.agent_configs import GenericAgentArgs
    from safearena.modeling import get_default_flags

    chat_args = XAIModelArgs(
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
