from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np


def sum_array_mapping(mapping: dict[str, np.ndarray], context: "DistributedContext") -> dict[str, np.ndarray]:
    context.ensure_same_mapping_keys(mapping, label="sum_array_mapping")
    return {key: context.allreduce_array_sum(value) for key, value in mapping.items()}


def _mpi_environment_is_active() -> bool:
    markers = [
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "PMIX_RANK",
        "MPI_LOCALNRANKS",
        "SLURM_NTASKS",
    ]
    return any(name in os.environ for name in markers)


def _comm_uses_mpi4py(comm: object) -> bool:
    return type(comm).__module__.startswith("mpi4py.")


def _load_mpi_module():
    try:
        from mpi4py import MPI  # type: ignore
    except ImportError:  # pragma: no cover
        return None
    return MPI


@dataclass(frozen=True)
class DistributedContext:
    rank: int = 0
    size: int = 1
    _comm: object | None = None

    @classmethod
    def from_comm(cls, comm: object | None = None) -> "DistributedContext":
        if comm is None:
            if not _mpi_environment_is_active():
                return cls()
            mpi_module = _load_mpi_module()
            if mpi_module is None:
                return cls()
            comm = mpi_module.COMM_WORLD
            if int(comm.Get_size()) <= 1:
                return cls()
        return cls(rank=int(comm.Get_rank()), size=int(comm.Get_size()), _comm=comm)

    def allreduce_scalar_sum(self, value: float) -> float:
        if self._comm is None:
            return float(value)
        if not _comm_uses_mpi4py(self._comm):
            return float(self._comm.allreduce(float(value)))
        mpi_module = _load_mpi_module()
        if mpi_module is None:
            raise RuntimeError("mpi4py communicator detected but mpi4py could not be imported.")
        return float(self._comm.allreduce(float(value), op=mpi_module.SUM))

    def allreduce_int_sum(self, value: int) -> int:
        if self._comm is None:
            return int(value)
        if not _comm_uses_mpi4py(self._comm):
            return int(self._comm.allreduce(int(value)))
        mpi_module = _load_mpi_module()
        if mpi_module is None:
            raise RuntimeError("mpi4py communicator detected but mpi4py could not be imported.")
        return int(self._comm.allreduce(int(value), op=mpi_module.SUM))

    def allreduce_bool_any(self, value: bool) -> bool:
        return self.allreduce_int_sum(1 if value else 0) > 0

    def allgather_object(self, value: object) -> tuple[object, ...]:
        if self._comm is None:
            return (value,)
        if hasattr(self._comm, "allgather"):
            return tuple(self._comm.allgather(value))
        return (value,)

    def allreduce_array_sum(self, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        self.ensure_same_array_shape(array, label="allreduce_array_sum")
        if self._comm is None:
            return array.copy()
        if not _comm_uses_mpi4py(self._comm):
            return np.asarray(self._comm.allreduce(array), dtype=np.float64)
        mpi_module = _load_mpi_module()
        if mpi_module is None:
            raise RuntimeError("mpi4py communicator detected but mpi4py could not be imported.")
        reduced = np.zeros_like(array)
        self._comm.Allreduce(array, reduced, op=mpi_module.SUM)
        return reduced

    def allreduce_array_max(self, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        self.ensure_same_array_shape(array, label="allreduce_array_max")
        if self._comm is None:
            return array.copy()
        if not _comm_uses_mpi4py(self._comm):
            return np.asarray(self._comm.allreduce(array), dtype=np.float64)
        mpi_module = _load_mpi_module()
        if mpi_module is None:
            raise RuntimeError("mpi4py communicator detected but mpi4py could not be imported.")
        reduced = np.zeros_like(array)
        self._comm.Allreduce(array, reduced, op=mpi_module.MAX)
        return reduced

    def ensure_same_array_shape(self, value: np.ndarray, label: str = "array collective") -> None:
        if self._comm is None or self.size <= 1:
            return
        local_shape = tuple(np.asarray(value).shape)
        gathered = self.allgather_object(local_shape)
        if len(gathered) != self.size:
            return
        if any(tuple(shape) != local_shape for shape in gathered):
            raise RuntimeError(f"MPI {label} shape mismatch across ranks: {gathered}")

    def ensure_same_mapping_keys(self, mapping: dict[str, object], label: str = "mapping collective") -> None:
        if self._comm is None or self.size <= 1:
            return
        local_keys = tuple(sorted(str(key) for key in mapping.keys()))
        gathered = self.allgather_object(local_keys)
        if len(gathered) != self.size:
            return
        if any(tuple(keys) != local_keys for keys in gathered):
            raise RuntimeError(f"MPI {label} key mismatch across ranks: {gathered}")

    def bcast_array(self, value: np.ndarray, root: int = 0) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if self._comm is None:
            return array.copy()
        broadcasted = self._comm.bcast(array if self.rank == root else None, root=root)
        return np.asarray(broadcasted, dtype=np.float64)

    def bcast_object(self, value: object, root: int = 0) -> object:
        if self._comm is None:
            return value
        return self._comm.bcast(value if self.rank == root else None, root=root)

    def barrier(self) -> None:
        if self._comm is None:
            return
        if hasattr(self._comm, "Barrier"):
            self._comm.Barrier()
            return
        if hasattr(self._comm, "barrier"):
            self._comm.barrier()
