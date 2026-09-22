"""
The replay environment: the one thing `culprit` needs that Phoenix cannot give it.

A trace records what *did* happen. Counterfactual localization asks what *would*
have happened, and no amount of stored telemetry answers that -- you have to be
able to run the agent's tools again. That is the honest precondition for this
whole approach, and putting it behind an explicit protocol makes it a stated
requirement rather than a hidden assumption buried in the benchmark.

To use `culprit` against a real Phoenix project you implement two methods:
`execute` (run a tool call) and `propose` (ask the agent's policy for its next
action given a set of observations). Both already exist in any agent framework;
they are its inner loop. The benchmark supplies `BenchEnvironment`, which is
the same interface over the synthetic world.

Tools must be safe to re-execute. Read-only tools -- lookups, retrieval, search
-- qualify. A tool that charges a card does not, and `culprit` has no way to
know the difference, so the caller declares it via `replayable_tools`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .tools import ToolError, call_tool
from .world import World


@runtime_checkable
class Environment(Protocol):
    """What replay needs from the system under test."""

    def execute(self, tool: str, args: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        """Run a tool call. Returns (result, error_message)."""
        ...

    def replayable(self, tool: str) -> bool:
        """May this tool be re-executed safely? Side-effecting tools must say no."""
        ...


class BenchEnvironment:
    """The synthetic world behind the `Environment` protocol."""

    def __init__(self, world: World, replayable_tools: set[str] | None = None) -> None:
        self.world = world
        # Every tool in the benchmark world is a read-only lookup.
        self.replayable_tools = replayable_tools

    def execute(self, tool: str, args: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        try:
            return call_tool(self.world, tool, args), None
        except ToolError as exc:
            return {}, str(exc)

    def replayable(self, tool: str) -> bool:
        return self.replayable_tools is None or tool in self.replayable_tools


def as_environment(obj: Any) -> Environment:
    """Accept either a World or an Environment, so callers need not care."""
    if isinstance(obj, World):
        return BenchEnvironment(obj)
    return obj
