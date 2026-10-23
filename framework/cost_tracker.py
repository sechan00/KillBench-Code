"""
CostTracker: Tracks API token usage and estimates costs across all LLM providers.
Singleton class that monkey-patches API clients to automatically record token usage.
"""

import json
import re
import threading
import functools
import logging
from collections import defaultdict
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)

# Regex for characters that break JSON: null bytes, control chars, lone surrogates
_PROBLEM_CHAR_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]|[\ud800-\udfff]')


def _scan_str(s: str, path: str):
    """Scan a single string for JSON-breaking characters and log findings."""
    if not isinstance(s, str):
        return
    # 1) regex scan for known bad chars
    for m in _PROBLEM_CHAR_RE.finditer(s):
        ch = m.group()
        ctx = repr(s[max(0, m.start() - 30): m.end() + 30])
        logger.error(
            f"[DEBUG-JSON] BAD CHAR at {path}[{m.start()}]: "
            f"U+{ord(ch):04X}  context={ctx}"
        )
    # 2) also try json.dumps with ensure_ascii=False to catch surrogates missed by regex
    try:
        json.dumps(s, ensure_ascii=False)
    except (ValueError, UnicodeEncodeError) as e:
        logger.error(
            f"[DEBUG-JSON] json.dumps FAILED at {path}: {e}\n"
            f"  value (first 300): {repr(s[:300])}"
        )


def _scan_obj(obj, path: str):
    """Recursively scan any object for JSON-breaking content."""
    import math
    if isinstance(obj, str):
        _scan_str(obj, path)
    elif isinstance(obj, bytes):
        logger.error(f"[DEBUG-JSON] BYTES OBJECT at {path}: len={len(obj)}, repr={repr(obj[:80])}")
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            logger.error(f"[DEBUG-JSON] NaN/Inf FLOAT at {path}: {obj}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _scan_obj(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _scan_obj(item, f"{path}[{i}]")
    elif obj is not None and not isinstance(obj, (int, bool)):
        # Unexpected type
        logger.error(f"[DEBUG-JSON] UNEXPECTED TYPE at {path}: {type(obj).__name__} = {repr(str(obj)[:100])}")


def _try_json_dumps(obj, label: str, ensure_ascii: bool):
    """Try json.dumps and log if it fails."""
    try:
        json.dumps(obj, ensure_ascii=ensure_ascii)
    except (ValueError, UnicodeEncodeError) as e:
        logger.error(f"[DEBUG-JSON] json.dumps(ensure_ascii={ensure_ascii}) FAILED at {label}: {e}")
    except TypeError as e:
        logger.error(f"[DEBUG-JSON] json.dumps TypeError at {label}: {e} — likely bytes/non-serializable type")
    except Exception as e:
        logger.error(f"[DEBUG-JSON] json.dumps UNEXPECTED error at {label}: {type(e).__name__}: {e}")


def debug_messages_for_json_error(messages, llm=None):
    """
    Called when a 400 BadRequestError occurs.
    Logs all message content and tries multiple JSON serialization paths
    to find exactly what breaks OpenAI's JSON parser.
    """
    logger.error("=" * 70)
    logger.error("[DEBUG-JSON] Scanning messages for JSON-breaking content ...")
    logger.error(f"[DEBUG-JSON] Total messages: {len(messages) if messages else 0}")

    if not messages:
        logger.error("[DEBUG-JSON] No messages to scan.")
        logger.error("=" * 70)
        return

    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__
        base = f"messages[{i}]({msg_type})"

        # --- Log message summary ---
        content = getattr(msg, 'content', None)
        content_summary = repr(content[:200]) if isinstance(content, str) else repr(str(content)[:200])
        logger.error(f"[DEBUG-JSON] {base}: content_type={type(content).__name__}, "
                     f"content_len={len(content) if isinstance(content, str) else 'N/A'}, "
                     f"content_preview={content_summary}")

        additional_kwargs = getattr(msg, 'additional_kwargs', {}) or {}
        if additional_kwargs:
            logger.error(f"[DEBUG-JSON] {base}.additional_kwargs keys: {list(additional_kwargs.keys())}")
            for k, v in additional_kwargs.items():
                v_repr = repr(str(v)[:300]) if not isinstance(v, (list, dict)) else repr(str(v)[:300])
                logger.error(f"[DEBUG-JSON]   additional_kwargs[{k!r}]: type={type(v).__name__}, value={v_repr}")

        response_metadata = getattr(msg, 'response_metadata', {}) or {}
        if response_metadata:
            logger.error(f"[DEBUG-JSON] {base}.response_metadata: {repr(str(response_metadata)[:800])}")

        # --- Deep scan for bad chars and unexpected types ---
        _scan_obj(content, f"{base}.content")

        tool_calls = getattr(msg, 'tool_calls', None)
        if tool_calls:
            _scan_obj(tool_calls, f"{base}.tool_calls")

        if additional_kwargs:
            _scan_obj(additional_kwargs, f"{base}.additional_kwargs")

        if response_metadata:
            _scan_obj(response_metadata, f"{base}.response_metadata")

        # --- Try json.dumps both ways (ensure_ascii=True is what OpenAI SDK uses) ---
        msg_dict = {
            "type": msg_type,
            "content": content,
            "tool_calls": getattr(msg, 'tool_calls', None),
            "additional_kwargs": additional_kwargs,
        }
        _try_json_dumps(msg_dict, f"{base} (ensure_ascii=False)", ensure_ascii=False)
        _try_json_dumps(msg_dict, f"{base} (ensure_ascii=True)", ensure_ascii=True)

    # --- Try langchain_openai's _get_request_payload (full payload including params) ---
    if llm is not None:
        try:
            if hasattr(llm, '_get_request_payload'):
                payload = llm._get_request_payload(messages)
                logger.error(f"[DEBUG-JSON] payload keys: {list(payload.keys())}")
                for i, d in enumerate(payload.get('messages', [])):
                    logger.error(f"[DEBUG-JSON] payload.messages[{i}]: {repr(str(d)[:600])}")
                    _scan_obj(d, f"payload.messages[{i}]")
                    _try_json_dumps(d, f"payload.messages[{i}] (ensure_ascii=True)", ensure_ascii=True)
                # Full payload serialization
                _try_json_dumps(payload, "full_payload (ensure_ascii=True)", ensure_ascii=True)
                try:
                    payload_json = json.dumps(payload, ensure_ascii=True)
                    logger.error(f"[DEBUG-JSON] full_payload JSON ok, len={len(payload_json)}, preview={payload_json[:500]}")
                except Exception as je:
                    logger.error(f"[DEBUG-JSON] full_payload json.dumps failed: {je}")
            else:
                logger.error(f"[DEBUG-JSON] llm has no _get_request_payload; type={type(llm).__name__}, methods={[m for m in dir(llm) if 'payload' in m or 'message' in m.lower()]}")
        except Exception as e:
            logger.error(f"[DEBUG-JSON] _get_request_payload failed: {type(e).__name__}: {e}")

    logger.error("[DEBUG-JSON] Scan complete.")
    logger.error("=" * 70)


# Pricing per 1M tokens: (input_price, output_price)
PRICING: Dict[str, Tuple[float, float]] = {
    # OpenAI (source: pricepertoken.com/pricing-page/provider/openai, Feb 2026)
    "gpt-5.2": (1.75, 14.00),
    "gpt-5.1": (1.25, 10.00),
    "gpt-5": (1.25, 10.00),
    "gpt-5-pro": (15.00, 120.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    # xAI (source: pricepertoken.com/pricing-page/provider/xai, Feb 2026)
    "grok-4-1-fast-reasoning": (0.20, 0.50),
    # Anthropic (source: platform.claude.com/docs/en/about-claude/pricing, Feb 2026)
    "claude-sonnet-4-5-20250929": (3.00, 15.00),
    # Google (source: pricepertoken.com/pricing-page/model/google-gemini-2.5-pro, Feb 2026)
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-3-pro-preview": (2.00, 12.00),
}


class CostTracker:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._usage = defaultdict(lambda: {"input_tokens": 0, "output_tokens": 0})
        self._wrapped_clients = set()
        self._autosave_path: Optional[str] = None

    @classmethod
    def get_instance(cls) -> "CostTracker":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def set_autosave(self, filepath: str):
        """Set a file path to auto-save usage after every record() call."""
        self._autosave_path = filepath

    def save(self, filepath: str):
        """Persist current usage to a JSON file (atomic write)."""
        try:
            data = {model: dict(usage) for model, usage in self._usage.items()}
            tmp = filepath + ".tmp"
            with open(tmp, 'w') as f:
                json.dump(data, f, indent=2)
            import os as _os
            _os.replace(tmp, filepath)
        except Exception as e:
            logger.warning(f"[CostTracker] Failed to save cost data to {filepath}: {e}")

    def load(self, filepath: str):
        """Load and accumulate usage from a previously saved JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            for model, usage in data.items():
                self._usage[model]["input_tokens"] += usage.get("input_tokens", 0)
                self._usage[model]["output_tokens"] += usage.get("output_tokens", 0)
            logger.info(f"[CostTracker] Loaded previous cost data from {filepath}")
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"[CostTracker] Failed to load cost data from {filepath}: {e}")

    def record(self, model: str, input_tokens: int, output_tokens: int):
        """Record token usage for a model."""
        if input_tokens or output_tokens:
            self._usage[model]["input_tokens"] += input_tokens
            self._usage[model]["output_tokens"] += output_tokens
            if self._autosave_path:
                self.save(self._autosave_path)

    def wrap_openai_client(self, client):
        """Wrap OpenAI client to track chat.completions.create and responses.create."""
        if client is None or id(client) in self._wrapped_clients:
            return
        self._wrapped_clients.add(id(client))
        tracker = self

        # Wrap chat.completions.create
        if hasattr(client, 'chat') and hasattr(client.chat, 'completions'):
            original_chat_create = client.chat.completions.create

            @functools.wraps(original_chat_create)
            def wrapped_chat_create(*args, **kwargs):
                response = original_chat_create(*args, **kwargs)
                try:
                    if hasattr(response, 'usage') and response.usage:
                        model_name = getattr(response, 'model', kwargs.get('model', 'unknown'))
                        tracker.record(
                            model_name,
                            getattr(response.usage, 'prompt_tokens', 0),
                            getattr(response.usage, 'completion_tokens', 0)
                        )
                except Exception:
                    pass
                return response

            client.chat.completions.create = wrapped_chat_create

        # Wrap responses.create (OpenAI Responses API)
        if hasattr(client, 'responses'):
            original_responses_create = client.responses.create

            @functools.wraps(original_responses_create)
            def wrapped_responses_create(*args, **kwargs):
                response = original_responses_create(*args, **kwargs)
                try:
                    if hasattr(response, 'usage') and response.usage:
                        model_name = getattr(response, 'model', kwargs.get('model', 'unknown'))
                        tracker.record(
                            model_name,
                            getattr(response.usage, 'input_tokens', 0),
                            getattr(response.usage, 'output_tokens', 0)
                        )
                except Exception:
                    pass
                return response

            client.responses.create = wrapped_responses_create

    def wrap_anthropic_client(self, client):
        """Wrap Anthropic client to track messages.create."""
        if client is None or id(client) in self._wrapped_clients:
            return
        self._wrapped_clients.add(id(client))
        tracker = self

        if hasattr(client, 'messages'):
            original_create = client.messages.create

            @functools.wraps(original_create)
            def wrapped_create(*args, **kwargs):
                response = original_create(*args, **kwargs)
                try:
                    if hasattr(response, 'usage') and response.usage:
                        model_name = getattr(response, 'model', kwargs.get('model', 'unknown'))
                        tracker.record(
                            model_name,
                            getattr(response.usage, 'input_tokens', 0),
                            getattr(response.usage, 'output_tokens', 0)
                        )
                except Exception:
                    pass
                return response

            client.messages.create = wrapped_create

    def wrap_langchain_model(self, llm):
        """Wrap a LangChain chat model to track .invoke() usage."""
        if llm is None or id(llm) in self._wrapped_clients:
            return
        self._wrapped_clients.add(id(llm))
        tracker = self

        original_invoke = llm.invoke

        @functools.wraps(original_invoke)
        def wrapped_invoke(*args, **kwargs):
            try:
                response = original_invoke(*args, **kwargs)
            except Exception as e:
                # 400 BadRequestError: dump every message to find the bad character
                e_str = str(e)
                if "400" in e_str and ("parse" in e_str.lower() or "invalid_request" in e_str.lower()):
                    messages = args[0] if args else kwargs.get('input', [])
                    model_name = getattr(llm, 'model_name', getattr(llm, 'model', 'unknown'))
                    logger.error(f"[DEBUG-JSON] 400 BadRequestError caught in wrapped_invoke (model={model_name})")
                    debug_messages_for_json_error(messages, llm=llm)
                raise

            try:
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    meta = response.usage_metadata
                    model_name = getattr(llm, 'model_name', getattr(llm, 'model', 'unknown'))
                    tracker.record(
                        model_name,
                        meta.get('input_tokens', 0),
                        meta.get('output_tokens', 0)
                    )
            except Exception:
                pass
            return response

        # Bypass Pydantic __setattr__ validation (LangChain models are Pydantic BaseModel)
        object.__setattr__(llm, 'invoke', wrapped_invoke)

    def record_gemini_response(self, response, model_name: str):
        """Record usage from a Gemini generate_content response."""
        try:
            if hasattr(response, 'usage_metadata'):
                meta = response.usage_metadata
                self.record(
                    model_name,
                    getattr(meta, 'prompt_token_count', 0),
                    getattr(meta, 'candidates_token_count', 0)
                )
        except Exception:
            pass

    def _get_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for given token usage."""
        pricing = PRICING.get(model)
        if not pricing:
            # Try prefix matching (e.g., "gpt-5.2-2025-..." matches "gpt-5.2")
            for key in PRICING:
                if model.startswith(key) or key.startswith(model):
                    pricing = PRICING[key]
                    break
        if not pricing:
            return 0.0
        input_price, output_price = pricing
        return (input_tokens * input_price + output_tokens * output_price) / 1_000_000

    def print_summary(self, logger: Optional[logging.Logger] = None):
        """Print cost summary table."""
        if not self._usage:
            msg = "\nNo API usage recorded."
            if logger:
                logger.info(msg)
            else:
                print(msg)
            return

        lines = []
        lines.append("")
        lines.append("=" * 80)
        lines.append("API COST SUMMARY")
        lines.append("=" * 80)
        lines.append(f"{'Model':<40} {'Input Tokens':>14} {'Output Tokens':>14} {'Cost ($)':>10}")
        lines.append("-" * 80)

        total_input = 0
        total_output = 0
        total_cost = 0.0

        # Sort by cost descending
        sorted_models = sorted(
            self._usage.items(),
            key=lambda x: self._get_cost(x[0], x[1]["input_tokens"], x[1]["output_tokens"]),
            reverse=True
        )

        for model, usage in sorted_models:
            inp = usage["input_tokens"]
            out = usage["output_tokens"]
            cost = self._get_cost(model, inp, out)
            total_input += inp
            total_output += out
            total_cost += cost

            cost_str = f"${cost:.2f}" if cost > 0 else "N/A"
            lines.append(f"{model:<40} {inp:>14,} {out:>14,} {cost_str:>10}")

        lines.append("-" * 80)
        lines.append(f"{'TOTAL':<40} {total_input:>14,} {total_output:>14,} {'$' + f'{total_cost:.2f}':>10}")
        lines.append("=" * 80)

        output = "\n".join(lines)
        if logger:
            logger.info(output)
        else:
            print(output)
