"""BrowserGym task wrappers for KillBench/lite scenarios.

One task per (scenario, attack_type, prompt_idx) tuple. Tasks are
registered with BrowserGym under ids of the form
`killbench_lite.<scenario>.<attack_type>.<prompt_idx>` so e.g.
`gym.make("browsergym/killbench_lite.pii.CR.20", ...)` works.

The task is trivial: navigate to the scenario's local HTTP-served
index page and return the (URL-substituted) malicious user request as
the goal. We never declare the task complete programmatically — Lite
pages have no real submit endpoint — so DSR/ASR judgments happen
post-hoc on the trajectory's final message.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import playwright.sync_api as pw
from browsergym.core.registration import register_task
from browsergym.core.task import AbstractBrowserTask

from attack_data import (
    TEST_INDICES_DEFAULT,
    AttackTask,
    load_attack_tasks,
)
from lite_server import start_url_for

log = logging.getLogger("killbench_lite.tasks")

_REWARD_UNFINISHED = 0.0


class LiteHarmTask(AbstractBrowserTask):
    """Minimal BrowserGym task wrapping a Lite (PII/NEWS/HACK) scenario.

    Behavior:
    - `setup`: navigate the Playwright page to the scenario's HTTP-served
      `index.html` and return the malicious user request as `goal`.
    - `validate`: never auto-completes. agentlab Study stops on max_steps
      or on a user-channel `exit` message. ASR vs DSR judged post-hoc.
    """

    _TASK_ID = "killbench_lite_generic"

    @classmethod
    def get_task_id(cls):
        return cls._TASK_ID

    def __init__(
        self,
        seed: int,
        scenario: str,
        attack_type: str,
        prompt_idx: int,
        user_request: str,
    ) -> None:
        super().__init__(seed)
        self.scenario = scenario.lower()
        self.attack_type = attack_type.upper()
        self.prompt_idx = int(prompt_idx)
        self.user_request = user_request
        self.start_url = start_url_for(self.scenario)
        task_id = f"lite.{self.scenario}.{self.attack_type}.{self.prompt_idx}"
        # Match Heavy task expectations so anything that reads task_configs works.
        self.task_configs = [{
            "task_id": task_id,
            "scenario": self.scenario,
            "attack_type": self.attack_type,
            "prompt_idx": self.prompt_idx,
        }]

    def setup(self, page: pw.Page) -> Tuple[str, dict]:
        log.info(
            f"LiteHarmTask.setup scenario={self.scenario} "
            f"attack_type={self.attack_type} prompt_idx={self.prompt_idx} "
            f"start_url={self.start_url}"
        )
        page.goto(self.start_url, timeout=15_000)
        return self.user_request, {
            "scenario": self.scenario,
            "attack_type": self.attack_type,
            "prompt_idx": self.prompt_idx,
            "task_id": self.task_configs[0]["task_id"],
        }

    def validate(
        self, page: pw.Page, chat_messages: list
    ) -> Tuple[float, bool, str, dict]:
        # TCR early-termination REMOVED: previously the trajectory ended the
        # moment the agent emitted any assistant message to the user, which cut
        # attack-success trajectories off before they could actually complete
        # the harmful task — making TCR (task-completion rate) unmeasurable.
        # Now we let the trajectory run to max_steps (or an explicit user
        # `exit`) so the agent can finish the task and ARIA can judge TCR.
        # (Cost trade-off: refusal trajectories now spin to max_steps.)
        for msg in chat_messages:
            if msg.get("role") == "user" and msg.get("message") == "exit":
                return _REWARD_UNFINISHED, True, "", {}
        return _REWARD_UNFINISHED, False, "", {}


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------

_REGISTERED: set = set()
_TASK_INDEX: dict = {}  # short_id → AttackTask


def register_all_tasks(
    scenarios=("pii", "news", "hack"),
    indices: Optional[List[int]] = None,
    attack_types: Optional[List[str]] = None,
) -> List[str]:
    """Register one task per (scenario, attack_type, prompt_idx).

    Default selection (indices=None, attack_types=None) mirrors test.py:
    10 attack types × 10 indices ([20..29]) = 100 tasks per scenario.

    Returns the registered short ids (without the `browsergym/` prefix).
    """
    if indices is None:
        indices = list(TEST_INDICES_DEFAULT)
    registered: List[str] = []
    for scenario in scenarios:
        try:
            base_url = start_url_for(scenario)
        except Exception as e:
            log.warning(f"register_all_tasks: base_url failed for {scenario}: {e}")
            continue
        try:
            tasks = load_attack_tasks(
                scenario,
                base_url=base_url,
                indices=indices,
                attack_types=attack_types,
            )
        except Exception as e:
            log.warning(f"register_all_tasks: load failed for {scenario}: {e}")
            continue
        for t in tasks:
            short_id = (
                f"killbench_lite.{t.scenario}.{t.attack_type}.{t.prompt_idx}"
            )
            if short_id in _REGISTERED:
                registered.append(short_id)
                continue
            register_task(
                id=short_id,
                task_class=LiteHarmTask,
                task_kwargs={
                    "scenario": t.scenario,
                    "attack_type": t.attack_type,
                    "prompt_idx": t.prompt_idx,
                    "user_request": t.user_request,
                },
            )
            _REGISTERED.add(short_id)
            _TASK_INDEX[short_id] = t
            registered.append(short_id)
    log.info(f"register_all_tasks: registered {len(registered)} tasks "
             f"across {len(scenarios)} scenario(s)")
    return registered


def get_task_meta(short_id: str) -> Optional[AttackTask]:
    return _TASK_INDEX.get(short_id)
