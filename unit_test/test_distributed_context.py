from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goattm.runtime.distributed import DistributedContext, sum_array_mapping  # noqa: E402


class _FakeComm:
    def __init__(self, gathered):
        self.gathered = gathered

    def Get_rank(self) -> int:
        return 0

    def Get_size(self) -> int:
        return 2

    def allgather(self, value):
        _ = value
        return self.gathered

    def allreduce(self, value):
        return value

    def bcast(self, value, root=0):
        _ = root
        return value

    def Barrier(self) -> None:
        return None


class DistributedContextTest(unittest.TestCase):
    def test_allreduce_array_sum_rejects_shape_mismatch(self) -> None:
        context = DistributedContext(rank=0, size=2, _comm=_FakeComm(gathered=[(2,), (3,)]))
        with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            context.allreduce_array_sum(np.ones(2))

    def test_sum_array_mapping_rejects_key_mismatch(self) -> None:
        context = DistributedContext(rank=0, size=2, _comm=_FakeComm(gathered=[("a",), ("b",)]))
        with self.assertRaisesRegex(RuntimeError, "key mismatch"):
            sum_array_mapping({"a": np.ones(1)}, context)


if __name__ == "__main__":
    unittest.main()
