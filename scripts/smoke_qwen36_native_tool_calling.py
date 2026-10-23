#!/usr/bin/env python3
"""Smoke-test OpenAI-compatible native tool calls for Qwen3.6.

This verifies the exact contract test.py needs:
1. The raw OpenAI-compatible endpoint returns message.tool_calls.
2. LangChain ChatOpenAI.bind_tools also exposes AIMessage.tool_calls.
"""

from __future__ import annotations

import argparse
import json
import sys

from openai import OpenAI


def _summarize_tool_calls(tool_calls) -> list[dict]:
    summary = []
    for call in tool_calls or []:
        function = getattr(call, "function", None)
        summary.append(
            {
                "id": getattr(call, "id", None),
                "type": getattr(call, "type", None),
                "name": getattr(function, "name", None),
                "arguments": getattr(function, "arguments", None),
            }
        )
    return summary


def raw_openai_check(base_url: str, api_key: str, model: str, max_tokens: int) -> None:
    client = OpenAI(api_key=api_key, base_url=base_url)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Return the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name, for example Paris.",
                        }
                    },
                    "required": ["city"],
                },
            },
        }
    ]

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Call the get_weather tool for Paris. Do not answer in text.",
            }
        ],
        tools=tools,
        tool_choice="required",
        temperature=0,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    choice = response.choices[0]
    message = choice.message
    print("[raw-openai] finish_reason:", choice.finish_reason)
    print("[raw-openai] content:", repr(message.content))
    print("[raw-openai] tool_calls:", json.dumps(_summarize_tool_calls(message.tool_calls), ensure_ascii=False))

    if not message.tool_calls:
        raise RuntimeError("raw OpenAI-compatible response did not include message.tool_calls")


def langchain_check(base_url: str, api_key: str, model: str, max_tokens: int) -> None:
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI

    @tool
    def get_weather(city: str) -> str:
        """Return the current weather for a city."""

        return f"{city}: 21C and clear"

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    message = llm.bind_tools(
        [get_weather],
        tool_choice="required",
        parallel_tool_calls=False,
    ).invoke("Call the get_weather tool for Paris. Do not answer in text.")

    print("[langchain] content:", repr(message.content))
    print("[langchain] tool_calls:", json.dumps(message.tool_calls, ensure_ascii=False))

    if not message.tool_calls:
        raise RuntimeError("LangChain AIMessage.tool_calls was empty")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8007/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="qwen3.6-uncensored")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--skip-langchain", action="store_true")
    args = parser.parse_args()

    try:
        raw_openai_check(args.base_url, args.api_key, args.model, args.max_tokens)
        if not args.skip_langchain:
            langchain_check(args.base_url, args.api_key, args.model, args.max_tokens)
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    print("[OK] Native tool calling is available through raw OpenAI API and LangChain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
