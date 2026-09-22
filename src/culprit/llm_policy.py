"""
A real LLM agent over the same tool surface as the scripted policy.

This exists to answer the largest open question in the project: every result
elsewhere was measured on traces produced by rule-based Python, and whether
those findings survive contact with an actual model is untested. The fault
taxonomy was drawn from production failure modes and the simulation is
faithful, but faithful is not the same as real.

`LLMPolicy` subclasses `ScriptedPolicy` and overrides *only* `_propose`. The
loop guard, the step budget and the best-effort fallback are inherited
unchanged, so the two arms differ in the decision-making and in nothing else.
A difference in results is then attributable to the policy rather than to the
scaffolding around it.

Two properties make this affordable. Decisions are taken at temperature 0 and
cached by prompt, so replaying a prefix the agent has already walked costs
nothing -- which matters enormously, because localization replays each trace
several times. And the agent is *credulous by default*, like the scripted one:
nothing in the prompt tells it to distrust tool output. Telling it to be
suspicious would be testing a different agent than the one most teams deploy.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .agent import ScriptedPolicy
from .tools import TOOL_SPECS, Task

SYSTEM = (
    "You are a careful data-analysis agent. You answer questions by calling tools, "
    "one at a time, and then giving a final answer.\n\n"
    "Available tools:\n"
    + "\n".join(
        f"- {t['name']}({', '.join(t['args'])}): {t['description']}" for t in TOOL_SPECS
    )
    + "\n\nRespond with a single JSON object and nothing else, in one of two forms:\n"
    '  {"action": "call", "tool": "<name>", "args": {...}}\n'
    '  {"action": "finish", "answer": "<final answer>"}\n\n'
    "Follow the answer format the question asks for exactly. When a listing tool "
    "reports has_more=true, request the next page before concluding."
)

PROMPT = """Question: {question}

Steps so far:
{history}

What is your next action? JSON only."""


def _render_history(observations: list[dict[str, Any]]) -> str:
    if not observations:
        return "(none yet -- this is your first action)"
    lines = []
    for i, o in enumerate(observations, 1):
        call = f"{o.get('tool')}({json.dumps(o.get('args', {}), sort_keys=True)})"
        if o.get("ok"):
            lines.append(f"{i}. {call}\n   -> {json.dumps(o.get('result', {}), sort_keys=True)}")
        else:
            lines.append(f"{i}. {call}\n   -> ERROR: {o.get('error')}")
    return "\n".join(lines)


def _parse_action(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if out.get("action") == "finish":
        return {"action": "finish", "answer": str(out.get("answer", ""))}
    if out.get("action") == "call" and out.get("tool"):
        args = out.get("args") or {}
        return {
            "action": "call",
            "tool": str(out["tool"]),
            "args": args if isinstance(args, dict) else {},
        }
    return None


class LLMPolicy(ScriptedPolicy):
    """The same agent scaffolding, with a model making the decisions."""

    name = "llm"

    def __init__(self, llm: Any, **kw: Any) -> None:
        super().__init__(**kw)
        self.llm = llm
        self.n_unparseable = 0

    def _propose(self, task: Task, observations: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = PROMPT.format(question=task.question, history=_render_history(observations))
        reply = self.llm.complete(prompt, system=SYSTEM)
        action = _parse_action(reply.text)
        if action is None:
            # An unparseable reply is a real agent failure mode, but it is not
            # the one under study here, so it is counted and turned into a
            # best-effort finish rather than silently becoming a tool error.
            self.n_unparseable += 1
            return {"action": "finish", "answer": self._best_effort(task, observations)}
        return action
