"""AgentLab-compatible chat adapter that talks directly to Anthropic instead
of going through OpenRouter. Drop-in replacement for `OpenRouterModelArgs` for
Claude models when only ANTHROPIC_API_KEY is available.

Usage:
    from anthropic_chat_adapter import AnthropicModelArgs, prepare_claude_direct
    agent_args = prepare_claude_direct("claude-sonnet-4-5-20250929")

The adapter accepts OpenAI-format `messages` (list of {role, content[, image]}),
converts to Anthropic's API shape (top-level `system`, role-alternating `messages`
with image blocks in Anthropic format), and returns the result as
`make_assistant_message(text)`.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import anthropic
from agentlab.agents.generic_agent.agent_configs import GenericAgentArgs
from agentlab.llm.base_api import AbstractChatModel, BaseModelArgs
from agentlab.llm.chat_api import make_assistant_message

from safearena.modeling import get_default_flags  # uses safearena defaults

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message-shape conversion
# ---------------------------------------------------------------------------

_EMPTY_PLACEHOLDER = "[empty]"


def _coerce_content_blocks(content) -> list[dict]:
    """Take an OpenAI-format message `content` (string or list of blocks) and
    return Anthropic-format content blocks.

    Anthropic rejects messages with empty-string text blocks ("text content
    blocks must be non-empty"). We substitute a single-character placeholder
    in any text block whose body would otherwise be empty, since this happens
    naturally when an aligned model refuses by returning empty content and
    agentlab feeds that back as an assistant turn in the history.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content if content else _EMPTY_PLACEHOLDER}]

    out = []
    for block in content:
        # OpenAI-style image_url block
        if isinstance(block, dict) and block.get("type") == "image_url":
            url = block["image_url"]["url"] if isinstance(block["image_url"], dict) else block["image_url"]
            if url.startswith("data:"):
                # data:image/png;base64,XXXX
                header, b64 = url.split(",", 1)
                media_type = header.split(";")[0].split(":", 1)[1]
                out.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": b64},
                })
            else:
                # External URL — Anthropic supports URL source
                out.append({"type": "image", "source": {"type": "url", "url": url}})
        # OpenAI/Anthropic-style text block
        elif isinstance(block, dict) and block.get("type") == "text":
            t = block.get("text", "")
            out.append({"type": "text", "text": t if t else _EMPTY_PLACEHOLDER})
        # Bare string inside a list
        elif isinstance(block, str):
            out.append({"type": "text", "text": block if block else _EMPTY_PLACEHOLDER})
        else:
            # Unknown — coerce to text
            s = str(block)
            out.append({"type": "text", "text": s if s else _EMPTY_PLACEHOLDER})

    if not out:
        out = [{"type": "text", "text": _EMPTY_PLACEHOLDER}]
    return out


def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Pull leading system messages out and concatenate into a single string;
    return (system_str, remaining_messages)."""
    system_parts: list[str] = []
    rest: list[dict] = []
    for m in messages:
        if m.get("role") == "system":
            c = m.get("content")
            if isinstance(c, str):
                system_parts.append(c)
            elif isinstance(c, list):
                for blk in c:
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        system_parts.append(blk.get("text", ""))
                    elif isinstance(blk, str):
                        system_parts.append(blk)
        else:
            rest.append(m)
    return "\n\n".join(system_parts).strip(), rest


def _to_anthropic_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    system_str, rest = _split_system(messages)
    out = []
    for m in rest:
        role = m.get("role")
        if role not in ("user", "assistant"):
            # Treat tool / other as user
            role = "user"
        out.append({"role": role, "content": _coerce_content_blocks(m.get("content", ""))})
    # Anthropic requires alternating roles and non-empty content.
    return system_str, out


# ---------------------------------------------------------------------------
# Chat model + args
# ---------------------------------------------------------------------------

class AnthropicChatModel(AbstractChatModel):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        max_retry: int = 4,
        min_retry_wait_time: float = 5,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retry = max_retry
        self.min_retry_wait_time = min_retry_wait_time
        self.input_cost = 0.0
        self.output_cost = 0.0
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        # Tracker-compatibility attrs
        self.retries = 0
        self.success = False
        self.error_types: list[str] = []

    def __call__(self, messages: list[dict]) -> dict:
        system_str, mlist = _to_anthropic_messages(messages)
        last_err: Optional[Exception] = None
        for itr in range(self.max_retry):
            self.retries = itr + 1
            try:
                kwargs = dict(
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=mlist,
                )
                if system_str:
                    kwargs["system"] = system_str
                resp = self.client.messages.create(**kwargs)
                # Concatenate text blocks
                text = "".join(
                    b.text for b in resp.content if getattr(b, "type", None) == "text"
                )
                self.success = True
                return make_assistant_message(text)
            except (anthropic.APIError, anthropic.APIConnectionError) as e:
                last_err = e
                logger.warning(f"Anthropic API error ({itr+1}/{self.max_retry}): {e}")
                time.sleep(self.min_retry_wait_time * (2 ** itr))
            except Exception as e:  # pragma: no cover
                last_err = e
                logger.exception("Unexpected error calling Anthropic")
                time.sleep(self.min_retry_wait_time)
        raise RuntimeError(
            f"AnthropicChatModel failed after {self.max_retry} retries: {last_err}"
        )

    def get_stats(self):
        return {"n_retry_llm": self.retries}


@dataclass
class AnthropicModelArgs(BaseModelArgs):
    """Serializable args for instantiating AnthropicChatModel."""

    def make_model(self) -> AbstractChatModel:
        return AnthropicChatModel(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )


# ---------------------------------------------------------------------------
# Equivalent of SafeArena's prepare_claude — but uses Anthropic directly.
# ---------------------------------------------------------------------------

def prepare_claude_direct(
    model_name: str = "claude-sonnet-4-5-20250929",
    max_new_tokens: int = 1024,
    max_prompt_tokens: int = 16384 - 4096,
    max_total_tokens: int = 16384,
    use_vision: bool = True,
    use_som: bool = True,
    enable_chat: bool = False,
) -> GenericAgentArgs:
    chat_args = AnthropicModelArgs(
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
