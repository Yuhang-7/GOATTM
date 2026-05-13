from __future__ import annotations

from dataclasses import dataclass
import os

import torch


def _mpi_environment_is_active() -> bool:
    markers = (
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "PMIX_RANK",
        "MPI_LOCALNRANKS",
        "SLURM_NTASKS",
    )
    return any(name in os.environ for name in markers)


@dataclass(frozen=True)
class TorchMPIContext:
    rank: int = 0
    size: int = 1
    comm: object | None = None
    mpi: object | None = None

    @classmethod
    def from_mpi4py(cls) -> "TorchMPIContext":
        if not _mpi_environment_is_active():
            return cls()
        try:
            from mpi4py import MPI  # type: ignore
        except Exception:
            return cls()
        comm = MPI.COMM_WORLD
        size = int(comm.Get_size())
        if size <= 1:
            return cls()
        return cls(rank=int(comm.Get_rank()), size=size, comm=comm, mpi=MPI)

    @property
    def is_root(self) -> bool:
        return self.rank == 0

    def allreduce_float_sum(self, value: float) -> float:
        if self.comm is None or self.mpi is None:
            return float(value)
        return float(self.comm.allreduce(float(value), op=self.mpi.SUM))

    def allreduce_int_sum(self, value: int) -> int:
        if self.comm is None or self.mpi is None:
            return int(value)
        return int(self.comm.allreduce(int(value), op=self.mpi.SUM))

    def allreduce_gradients_sum(self, parameters: list[torch.nn.Parameter]) -> None:
        if self.comm is None or self.mpi is None:
            return
        for param in parameters:
            if param.grad is None:
                continue
            grad = param.grad.detach()
            if grad.device.type != "cpu":
                raise RuntimeError("MPI gradient allreduce currently supports CPU tensors only.")
            array = grad.numpy()
            self.comm.Allreduce(self.mpi.IN_PLACE, array, op=self.mpi.SUM)

    def barrier(self) -> None:
        if self.comm is not None:
            self.comm.Barrier()
