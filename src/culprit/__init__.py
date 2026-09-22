"""culprit -- root-cause localization for failed agent traces in Arize Phoenix."""

__version__ = "0.1.0"

from .agent import SpanRec, Trace, run_agent
from .environment import BenchEnvironment, Environment
from .faults import FAULT_CATALOG, make_injector
from .replay import ReplayEngine, ReplayResult

__all__ = [
    "__version__",
    "SpanRec", "Trace", "run_agent",
    "Environment", "BenchEnvironment",
    "FAULT_CATALOG", "make_injector",
    "ReplayEngine", "ReplayResult",
]
