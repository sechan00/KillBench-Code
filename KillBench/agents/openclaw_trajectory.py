"""Convert an OpenClaw session JSONL into the same Trajectory shape that
`aria_judge.extract_trajectory` returns for agentlab step pickles.

OpenClaw runs as a Node subprocess and persists every message it sends/receives
as one JSON line in `<sessionId>.jsonl` (see investigation notes). We walk the
file chronologically and pair each assistant `toolCall` with its matching
`toolResult` to reconstruct an agent step in the form aggregate_results.py
expects (think + action + axtree_txt). The final assistant message (no tool
call) becomes a synthetic `send_msg_to_user(...)` step so the action prefilter
and refusal judge fire on it.

Sidecar `openclaw_meta.json` carries goal/start_url/task_id — agentlab writes
those into step_0's obs.pkl but we have no equivalent so the adapter writes a
small JSON when launching OpenClaw.

Schema reminders (verified against openclaw source):
  - session header line:  {"type": "session", "id": ..., ...}
  - assistant turn:       {"type": "message", "message": {"role": "assistant",
                            "content": [{"type":"text","text":...},
                                        {"type":"toolCall","id":...,"name":...,
                                         "arguments":{...}}]}}
  - tool result:          {"type": "message", "message": {"role": "toolResult",
                            "toolCallId": ..., "toolName": ...,
                            "content": [{"type":"text","text":...}]}}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger("killswitch.openclaw_trajectory")


# Re-declared here to keep this module importable independently. aria_judge
# imports both dataclasses from .openclaw_trajectory when dispatching.
@dataclass
class _StepRec:
    step_idx: int
    url: str
    think: str
    action: str
    axtree_txt: str = ""


@dataclass
class _Trajectory:
    task_id: str
    goal: str
    start_url: str
    steps: list
    cum_reward: float
    n_steps: int
    err_msg: Optional[str]


def _truncate(s: str, n: int) -> str:
    if not s:
        return ""
    s = s.strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def _concat_text_blocks(content) -> str:
    if not isinstance(content, list):
        return ""
    out = []
    for b in content:
        if isinstance(b, dict) and b.get("type") == "text":
            t = b.get("text") or ""
            if t:
                out.append(t)
    return "\n".join(out)


def _find_tool_call(content) -> Optional[dict]:
    """Return the LAST toolCall block in an assistant content array, or None.

    OpenClaw assistant turns can carry zero or one toolCall (multi-tool turns
    are rare in current models). We take the last one defensively.
    """
    if not isinstance(content, list):
        return None
    last = None
    for b in content:
        if isinstance(b, dict) and b.get("type") == "toolCall":
            last = b
    return last


def _format_action(tool_call: dict) -> str:
    """Render a toolCall block as a `name({json args})` string so the action
    prefilter substring-matches (send_msg_to_user / browser / etc.).
    """
    name = tool_call.get("name") or "unknown_tool"
    args = tool_call.get("arguments")
    if args is None:
        args = tool_call.get("input")  # defensive: some persisters use this
    try:
        arg_str = json.dumps(args or {}, ensure_ascii=False)
    except (TypeError, ValueError):
        arg_str = str(args)
    return f"{name}({arg_str})"


def _is_browser_snapshot(tool_call: Optional[dict]) -> bool:
    if not isinstance(tool_call, dict):
        return False
    name = (tool_call.get("name") or "").lower()
    # OpenClaw browser plugin exposes browser_snapshot / browser_navigate /
    # browser_act. snapshot is the one that returns plain-text page state.
    return name in ("browser_snapshot", "browser_observe")


def _is_browser_navigate(tool_call: Optional[dict]) -> bool:
    if not isinstance(tool_call, dict):
        return False
    return (tool_call.get("name") or "").lower() in ("browser_navigate", "browser_goto")


def _extract_obs_text(tool_result_msg: dict) -> str:
    """Pull the text portion out of a toolResult message. For browser_snapshot
    in `ai` format this is the LLM-visible accessibility tree string —
    substring-matchable for our payload marker.
    """
    return _concat_text_blocks(tool_result_msg.get("content"))


def _navigated_url(tool_call: dict) -> str:
    args = tool_call.get("arguments") or tool_call.get("input") or {}
    if isinstance(args, dict):
        for k in ("url", "to", "target"):
            v = args.get(k)
            if isinstance(v, str) and v:
                return v
    return ""


def extract_openclaw_trajectory(
    subdir: Path,
    max_think_chars: int = 400,
    axtree_max_chars: int = 30000,
):
    """Walk an openclaw task subdirectory and return a Trajectory.

    Expects:
      subdir/openclaw_session.jsonl   — raw session transcript (one JSON / line)
      subdir/openclaw_meta.json       — {task_id, goal, start_url, sessionId,
                                          n_steps?, err_msg?, cum_reward?}

    Returns None if either file is missing or unreadable.
    """
    session_path = subdir / "openclaw_session.jsonl"
    meta_path = subdir / "openclaw_meta.json"
    if not session_path.exists() or not meta_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning(f"{subdir.name}: bad openclaw_meta.json: {e}")
        return None

    task_id = meta.get("task_id") or subdir.name
    goal = meta.get("goal") or ""
    start_url = meta.get("start_url") or ""

    # Track the most recent URL the agent navigated to so each StepRec carries
    # the URL it was reasoning against (mirrors agentlab obs["url"]).
    cur_url = start_url

    pending: Optional[dict] = None  # {"think": str, "tool_call": dict}
    steps: list[_StepRec] = []
    step_idx = 0

    try:
        with session_path.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    line = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if line.get("type") != "message":
                    continue
                msg = line.get("message") or {}
                # OpenClaw re-emits each raw model reply as a second message
                # with api="cli" (the cleaned text shown in the CLI). Skip
                # those so we don't double-count assistant turns. The raw
                # entry is what carries tool_use blocks anyway.
                if msg.get("api") == "cli":
                    continue
                role = msg.get("role")
                content = msg.get("content")

                if role == "assistant":
                    text = _concat_text_blocks(content)
                    tool_call = _find_tool_call(content)
                    if tool_call is not None:
                        # Carry through to the next toolResult.
                        pending = {"think": text, "tool_call": tool_call}
                        if _is_browser_navigate(tool_call):
                            nav = _navigated_url(tool_call)
                            if nav:
                                cur_url = nav
                    else:
                        # Final reply (or any tool-less turn). Synthesize a
                        # `send_msg_to_user(...)` action so the action filter
                        # routes it to the refusal judge. Mirrors aria_judge's
                        # plain-text recovery path.
                        if text:
                            steps.append(_StepRec(
                                step_idx=step_idx,
                                url=cur_url,
                                think=_truncate(text, max_think_chars),
                                action=f"send_msg_to_user({text[:400]!r})",
                                axtree_txt="",
                            ))
                            step_idx += 1

                elif role == "toolResult":
                    if pending is None:
                        # Orphan tool result — skip, can't attribute think/action.
                        continue
                    think = pending["think"]
                    tool_call = pending["tool_call"]
                    obs_text = _extract_obs_text(msg)

                    # For browser snapshots, the payload marker lives in
                    # obs_text. For other tools obs_text is short and the
                    # anchor logic naturally skips it. Either way, axtree_txt
                    # is the LLM-visible string — same semantics as agentlab.
                    ax = obs_text or ""
                    if len(ax) > axtree_max_chars:
                        # Payload is at body.beforeend so the marker lives
                        # near the END of the snapshot — keep the tail.
                        ax = ax[-axtree_max_chars:]

                    steps.append(_StepRec(
                        step_idx=step_idx,
                        url=cur_url,
                        think=_truncate(think, max_think_chars),
                        action=_format_action(tool_call),
                        axtree_txt=ax,
                    ))
                    step_idx += 1
                    pending = None
    except Exception as e:
        log.warning(f"{subdir.name}: openclaw_session.jsonl parse error: {e}")
        return None

    return _Trajectory(
        task_id=task_id,
        goal=goal,
        start_url=start_url,
        steps=steps,
        cum_reward=float(meta.get("cum_reward", 0.0)),
        n_steps=int(meta.get("n_steps", len(steps))),
        err_msg=meta.get("err_msg"),
    )
