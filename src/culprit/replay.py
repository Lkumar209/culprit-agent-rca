"""
Counterfactual replay: the causal test at the centre of this project.

The question a localizer has to answer is *which span caused this failure*,
and the honest way to answer a causal question is to intervene. For a span
`k` we repair that span's output and replay the run forward from it. If the
final answer changes, `k` is causally implicated. If nothing changes, `k` was
a bystander no matter how suspicious it looks.

Two design points carry most of the validity of the results:

1. **Replay happens in the same broken environment.** The armed injector comes
   along for the ride, so a persistent fault re-fires on replay exactly as it
   would if you re-ran the agent in production. The alternative -- replaying
   into a healthy world -- would make every span upstream of the fault look
   like a fix and the whole method would collapse into "blame span 1".

2. **The repair oracle is deliberately fallible.** `repair_success_prob`
   models the fact that in production the repaired value comes from an LLM
   asked "what should this step have returned?", and that proposal is
   sometimes wrong. Setting it to 1.0 measures the method's ceiling; the
   benchmark sweeps it downward to measure what a user would actually get.

The outcome signal comes in two flavours. `fixed` compares against the gold
answer and is only available in the benchmark. `changed` merely asks whether
the final answer moved, needs no ground truth, and is therefore what a real
Phoenix user can actually compute. Reporting both is how we find out how much
the deployable signal costs you.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

from .agent import MAX_STEPS, ScriptedPolicy, SpanRec, Trace
from .environment import Environment, as_environment
from .world import World


@dataclass
class ReplayResult:
    changed: bool           # final answer differs from the original run
    fixed: bool             # final answer matches gold (benchmark only)
    new_answer: str | None
    n_tool_calls: int       # replay cost, in tool executions
    n_policy_calls: int     # replay cost, in policy/LLM invocations


@dataclass
class Step:
    """One decision/tool pair recovered from a recorded trace."""

    decision_span: SpanRec
    tool_span: SpanRec | None
    decision: dict[str, Any]
    observation: dict[str, Any] | None


def parse_steps(trace: Trace) -> list[Step]:
    """Recover the agent's step structure from a flat span list."""
    steps: list[Step] = []
    pending: SpanRec | None = None
    for s in trace.spans:
        if s.span_kind == "AGENT":
            continue
        if s.name in ("agent.decide", "agent.answer"):
            if pending is not None:
                steps.append(
                    Step(pending, None, dict(pending.output.get("decision", {})), None)
                )
            if s.name == "agent.answer":
                pending = None
                continue
            pending = s
        elif s.span_kind == "TOOL" and pending is not None:
            obs = {
                "tool": s.input.get("tool"),
                "args": s.input.get("args", {}),
                "ok": bool(s.output.get("ok", s.error is None)),
                "result": s.output.get("result", {}),
                "error": s.error,
                "is_retry": bool(pending.output.get("decision", {}).get("is_retry")),
            }
            steps.append(Step(pending, s, dict(pending.output.get("decision", {})), obs))
            pending = None
    if pending is not None:
        steps.append(Step(pending, None, dict(pending.output.get("decision", {})), None))
    return steps


class ReplayEngine:
    """
    Executes single-span interventions against a recorded trace.

    `env_injector` should be an armed clone of whatever corrupted the original
    run (see `Injector.clone_armed`). Pass `None` to replay into a healthy
    world -- useful for sanity checks, misleading for localization.
    """

    def __init__(
        self,
        world: World | Environment,
        policy: ScriptedPolicy | None = None,
        env_injector: Any | None = None,
        repair_success_prob: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.env = as_environment(world)
        self.world = world
        self.policy = policy or ScriptedPolicy()
        self.env_injector = env_injector
        self.repair_success_prob = repair_success_prob
        self.rng = random.Random(seed)

    # -- helpers ------------------------------------------------------------
    def _execute(
        self, decision: dict[str, Any], step_no: int, honor_decision: bool = False
    ) -> dict[str, Any]:
        """
        Run one tool call through the (possibly still broken) environment.

        `honor_decision` suppresses decision-phase corruption for this step.
        It is set when the step *is* the intervention: a decision fault is the
        agent's own error, so repairing it has to be the final word. Without
        this, the environment re-corrupts the repaired decision on its way out
        and no decision-phase fault is ever localizable.
        """
        if self.env_injector is not None and not honor_decision:
            decision, _ = self.env_injector("decision", step_no, decision)
        tool, args = decision.get("tool"), decision.get("args", {})
        result, err = self.env.execute(tool, args)
        obs = {
            "tool": tool, "args": args, "ok": err is None, "result": result,
            "error": err, "is_retry": bool(decision.get("is_retry")),
        }
        if self.env_injector is not None:
            obs, _ = self.env_injector("observation", step_no, obs)
        return obs

    def _repair_succeeds(self) -> bool:
        return self.rng.random() < self.repair_success_prob

    # -- the intervention ---------------------------------------------------
    def intervene(self, trace: Trace, span_id: str, gold: str | None = None) -> ReplayResult:
        """
        Repair the span `span_id` and replay the run forward from it.

        Steps before the target are replayed verbatim from the recording --
        they are not re-executed, because re-executing them would be a
        different (and more expensive) intervention. The target step is
        repaired. Everything after it runs live.
        """
        steps = parse_steps(trace)
        target = next(
            (
                (i, "decision" if st.decision_span.span_id == span_id else "tool")
                for i, st in enumerate(steps)
                if st.decision_span.span_id == span_id
                or (st.tool_span is not None and st.tool_span.span_id == span_id)
            ),
            None,
        )
        if target is None:
            return ReplayResult(False, False, trace.final_answer, 0, 0)
        t_idx, phase = target

        observations: list[dict[str, Any]] = []
        n_tools = n_policy = 0

        # 1. Prefix: replay verbatim.
        for st in steps[:t_idx]:
            if st.observation is not None:
                observations.append(dict(st.observation))

        # 2. Target: repair.
        st = steps[t_idx]
        if phase == "decision":
            decision = dict(st.decision)
            if self._repair_succeeds():
                decision = self.policy._propose(trace.task, observations)
                n_policy += 1
            if decision.get("action") == "finish":
                return self._finish(trace, str(decision.get("answer", "")), gold, n_tools, n_policy)
            obs = self._execute(decision, t_idx, honor_decision=True)
            n_tools += 1
            observations.append(obs)
        else:
            obs = dict(st.observation or {})
            if self._repair_succeeds():
                # Re-run the call honestly: what the tool *should* have returned
                # for the arguments the agent actually sent.
                tool, args = obs.get("tool"), obs.get("args", {})
                result, err = self.env.execute(tool, args)
                obs = {**obs, "ok": err is None, "result": result, "error": err}
                n_tools += 1
            observations.append(obs)

        # 3. Suffix: run live in the same environment.
        answer: str | None = None
        for step_no in range(t_idx + 1, MAX_STEPS):
            decision = self.policy.decide(trace.task, observations)
            n_policy += 1
            if decision.get("action") == "finish":
                answer = str(decision.get("answer", ""))
                break
            observations.append(self._execute(decision, step_no))
            n_tools += 1

        return self._finish(trace, answer, gold, n_tools, n_policy)

    # -- prefix intervention (for bisection) --------------------------------
    def intervene_prefix(
        self,
        trace: Trace,
        upto_step: int,
        gold: str | None = None,
        protect: tuple[str, ...] = ("decision", "observation"),
    ) -> ReplayResult:
        """
        Replay the whole run live, shielding every step up to `upto_step`.

        This is the monotone probe that makes binary search possible. Write
        Q(k) = "shielding steps 0..k fixes the run". If the fault lives at step
        j then every k >= j includes it and Q is true, and every k < j leaves
        it in place and Q is false. Q is therefore a step function and the
        culprit is the smallest k where it flips, which bisection finds in
        O(log n) replays instead of the O(n) an exhaustive scan needs.

        Shielding is probabilistic: `repair_success_prob` decides, per step,
        whether the repair actually lands. A fallible repair oracle makes Q
        noisy rather than clean, which is precisely the regime the sweep in
        the results section explores.
        """
        observations: list[dict[str, Any]] = []
        n_tools = n_policy = 0
        answer: str | None = None

        for step_no in range(MAX_STEPS):
            shielded = step_no <= upto_step
            decision = self.policy.decide(trace.task, observations)
            n_policy += 1

            protect_d = shielded and "decision" in protect and self._repair_succeeds()
            if self.env_injector is not None and not protect_d:
                decision, _ = self.env_injector("decision", step_no, decision)

            if decision.get("action") == "finish":
                answer = str(decision.get("answer", ""))
                break

            tool, args = decision.get("tool"), decision.get("args", {})
            result, err = self.env.execute(tool, args)
            obs = {
                "tool": tool, "args": args, "ok": err is None, "result": result,
                "error": err, "is_retry": bool(decision.get("is_retry")),
            }
            protect_o = shielded and "observation" in protect and self._repair_succeeds()
            if self.env_injector is not None and not protect_o:
                obs, _ = self.env_injector("observation", step_no, obs)
            observations.append(obs)
            n_tools += 1

        return self._finish(trace, answer, gold, n_tools, n_policy)

    def _finish(
        self, trace: Trace, answer: str | None, gold: str | None, n_tools: int, n_policy: int
    ) -> ReplayResult:
        original = (trace.final_answer or "").strip()
        new = (answer or "").strip()
        return ReplayResult(
            changed=new != original,
            fixed=bool(gold is not None and new == gold.strip()),
            new_answer=answer,
            n_tool_calls=n_tools,
            n_policy_calls=n_policy,
        )
