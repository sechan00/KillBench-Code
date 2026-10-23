"""System Prompt Bypass Agent — the RQ3 adaptive attacker baseline.

The agent receives the bypass directive as a **real `system` role message**,
not as an `extra_instructions` block inside the human message. This is the
RQ3 adaptive-attacker baseline: the directive carries the same weight as a
genuine operator system prompt, telling the agent to ignore any in-page
STOP/HALT/Kill Switch content.

Implementation
--------------
agentlab.GenericAgent always emits a `system` role message built from
`agentlab.agents.dynamic_prompting.SystemPrompt._prompt` (a class-level
string). [system_patch.py](../system_patch.py) rewrites that string to
append our bypass directive. The Anthropic / OpenAI / OpenRouter / xAI
adapters preserve the system role end-to-end, so this lands in
`messages[system=...]` (Anthropic) or `messages[0].role=='system'`
(OpenAI-style) at API call time.

For worker subprocesses (joblib/loky/ray), the parent process exports
`KILLSWITCH_SYSTEM_PROMPT_MODE=hardened` and sitecustomize.py re-applies
the patch on worker startup.

Builders + dispatcher (Anthropic / OpenAI / OpenRouter / xAI):
- `prepare_claude_hardened`     — Anthropic via anthropic_chat_adapter
- `prepare_gpt_hardened`        — OpenAI via safearena.modeling.prepare_gpt
- `prepare_openrouter_hardened` — OpenRouter via openrouter_chat_adapter
- `prepare_grok_hardened`       — xAI via xai_chat_adapter
- `prepare_hardened`            — routes by model-name prefix
"""

from __future__ import annotations

import os

from agentlab.agents.generic_agent.agent_configs import GenericAgentArgs

import system_patch
from system_patch import ADVERSARIAL_SYSTEM_PROMPT  # re-export for callers


def _install_hardened_system_prompt() -> None:
    """Patch `dp.SystemPrompt._prompt` in this process AND mark for workers.

    Workers re-import everything from scratch — sitecustomize.py reads the
    env var below and calls `system_patch.install_hardened()` after the
    fresh agentlab import. Without that, only the parent has the patch.
    """
    system_patch.install_hardened()
    os.environ["KILLSWITCH_SYSTEM_PROMPT_MODE"] = "hardened"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def prepare_claude_hardened(
    model_name: str = "claude-sonnet-4-5-20250929",
    max_new_tokens: int = 1024,
    max_prompt_tokens: int = 16384 - 4096,
    max_total_tokens: int = 16384,
    use_vision: bool = True,
    use_som: bool = True,
    enable_chat: bool = False,
) -> GenericAgentArgs:
    """Anthropic + hardened system prompt."""
    from anthropic_chat_adapter import AnthropicModelArgs
    from safearena.modeling import get_default_flags

    _install_hardened_system_prompt()
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


def prepare_gpt_hardened(
    model_name: str = "gpt-4o-2024-08-06",
    max_new_tokens: int = 1024,
    max_prompt_tokens: int = 16384 - 4096,
    max_total_tokens: int = 16384,
    use_vision: bool = True,
    use_som: bool = True,
    enable_chat: bool = False,
) -> GenericAgentArgs:
    """OpenAI + hardened system prompt.

    Builds the same GenericAgent as `safearena.modeling.prepare_gpt`. The
    bypass directive is applied via `dp.SystemPrompt._prompt`, which the
    OpenAI ChatModel adapter forwards verbatim as a `role: system` message.
    """
    from safearena.modeling import prepare_gpt

    _install_hardened_system_prompt()
    return prepare_gpt(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        max_prompt_tokens=max_prompt_tokens,
        max_total_tokens=max_total_tokens,
        use_vision=use_vision,
        use_som=use_som,
        enable_chat=enable_chat,
        harmful=False,
    )


def prepare_openrouter_hardened(
    model_name: str,
    max_new_tokens: int = 1024,
    max_prompt_tokens: int = 16384 - 4096,
    max_total_tokens: int = 16384,
    use_vision: bool = True,
    use_som: bool = True,
    enable_chat: bool = False,
) -> GenericAgentArgs:
    """OpenRouter + hardened system prompt."""
    from openrouter_chat_adapter import prepare_openrouter

    _install_hardened_system_prompt()
    return prepare_openrouter(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        max_prompt_tokens=max_prompt_tokens,
        max_total_tokens=max_total_tokens,
        use_vision=use_vision,
        use_som=use_som,
        enable_chat=enable_chat,
    )


def prepare_grok_hardened(
    model_name: str,
    max_new_tokens: int = 1024,
    max_prompt_tokens: int = 16384 - 4096,
    max_total_tokens: int = 16384,
    use_vision: bool = True,
    use_som: bool = True,
    enable_chat: bool = False,
) -> GenericAgentArgs:
    """xAI Grok + hardened system prompt."""
    from xai_chat_adapter import prepare_grok

    _install_hardened_system_prompt()
    return prepare_grok(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        max_prompt_tokens=max_prompt_tokens,
        max_total_tokens=max_total_tokens,
        use_vision=use_vision,
        use_som=use_som,
        enable_chat=enable_chat,
    )


def prepare_hardened(
    model_name: str,
    **kwargs,
) -> GenericAgentArgs:
    """Route to the right hardened builder based on model name.

    Anthropic:   model starts with 'claude' or 'anthropic/'.
    OpenAI:      model starts with 'gpt-', 'o1', 'o3', or 'openai/'.
    xAI:         model starts with 'grok-' or 'xai/'.
    OpenRouter:  model starts with 'openrouter/', 'qwen/', 'meta-llama/',
                 'mistralai/', 'deepseek/', or 'google/'.
    """
    low = model_name.lower()
    if low.startswith("claude") or low.startswith("anthropic/"):
        return prepare_claude_hardened(model_name=model_name, **kwargs)
    if (low.startswith("gpt-") or low.startswith("openai/")
            or low.startswith("o1") or low.startswith("o3")):
        return prepare_gpt_hardened(model_name=model_name, **kwargs)
    if low.startswith("grok-") or low.startswith("xai/"):
        return prepare_grok_hardened(model_name=model_name, **kwargs)
    if (low.startswith("openrouter/") or low.startswith("qwen/")
            or low.startswith("meta-llama/") or low.startswith("mistralai/")
            or low.startswith("deepseek/") or low.startswith("google/")):
        or_model = (model_name[len("openrouter/"):]
                    if low.startswith("openrouter/") else model_name)
        return prepare_openrouter_hardened(model_name=or_model, **kwargs)
    raise ValueError(
        f"prepare_hardened: model {model_name!r} not routable. "
        "Supported providers: anthropic / openai / xai / openrouter."
    )


__all__ = [
    "ADVERSARIAL_SYSTEM_PROMPT",
    "prepare_claude_hardened",
    "prepare_gpt_hardened",
    "prepare_openrouter_hardened",
    "prepare_grok_hardened",
    "prepare_hardened",
]
