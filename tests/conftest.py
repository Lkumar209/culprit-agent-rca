import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from culprit.tools import build_tasks
from culprit.world import build_world


@pytest.fixture(scope="session")
def world():
    return build_world(seed=7)


@pytest.fixture(scope="session")
def tasks(world):
    return build_tasks(world, seed=11, n_per_kind=4)
