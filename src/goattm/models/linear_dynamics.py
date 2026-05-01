from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from goattm.core.parametrization import compressed_quadratic_dimension, mu_h_dimension


@dataclass(frozen=True)
class LinearDynamics:
    a: np.ndarray
    c: np.ndarray
    b: np.ndarray | None = None
    _h_matrix: np.ndarray = field(init=False, repr=False)
    _mu_h: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.a.ndim != 2 or self.a.shape[0] != self.a.shape[1]:
            raise ValueError(f"a must be square, got shape {self.a.shape}")
        r = self.a.shape[0]
        if self.c.ndim != 1 or self.c.shape[0] != r:
            raise ValueError(f"c must have shape ({r},), got {self.c.shape}")
        if self.b is not None and (self.b.ndim != 2 or self.b.shape[0] != r):
            raise ValueError(f"b must have shape ({r}, dp), got {self.b.shape}")
        object.__setattr__(self, "_h_matrix", np.zeros((r, compressed_quadratic_dimension(r)), dtype=np.float64))
        object.__setattr__(self, "_mu_h", np.zeros(mu_h_dimension(r), dtype=np.float64))

    @property
    def dimension(self) -> int:
        return self.a.shape[0]

    @property
    def input_dimension(self) -> int:
        return 0 if self.b is None else self.b.shape[1]

    @property
    def h_matrix(self) -> np.ndarray:
        return self._h_matrix

    @property
    def mu_h(self) -> np.ndarray:
        return self._mu_h

    def validate_state(self, u: np.ndarray, name: str = "u") -> None:
        if u.ndim != 1 or u.shape[0] != self.dimension:
            raise ValueError(f"{name} must have shape ({self.dimension},), got {u.shape}")

    def validate_input(self, p: np.ndarray, name: str = "p") -> None:
        if self.b is None:
            raise ValueError("This dynamics object does not define an input matrix b.")
        if p.ndim != 1 or p.shape[0] != self.input_dimension:
            raise ValueError(f"{name} must have shape ({self.input_dimension},), got {p.shape}")

    def forcing(self, p: np.ndarray | None = None) -> np.ndarray:
        if self.b is None or p is None:
            return self.c
        self.validate_input(p)
        return self.b @ p + self.c

    def rhs(self, u: np.ndarray, p: np.ndarray | None = None) -> np.ndarray:
        self.validate_state(u)
        return self.a @ u + self.forcing(p)

    def rhs_at_time(
        self,
        u: np.ndarray,
        t: float,
        input_function: Callable[[float], np.ndarray] | None = None,
    ) -> np.ndarray:
        p = None if input_function is None else np.asarray(input_function(t), dtype=np.float64)
        return self.rhs(u, p=p)

    def quadratic_bilinear(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        self.validate_state(u, "u")
        self.validate_state(v, "v")
        return np.zeros(self.dimension, dtype=np.float64)

    def quadratic_jacobian(self, u: np.ndarray) -> np.ndarray:
        self.validate_state(u)
        return np.zeros((self.dimension, self.dimension), dtype=np.float64)

    def quadratic_bilinear_action_matrix(self, u: np.ndarray) -> np.ndarray:
        self.validate_state(u)
        return np.zeros((self.dimension, compressed_quadratic_dimension(self.dimension)), dtype=np.float64)

    def rhs_jacobian(self, u: np.ndarray) -> np.ndarray:
        self.validate_state(u)
        return self.a

    def energy_preserving_defect(self, u: np.ndarray) -> float:
        self.validate_state(u)
        return 0.0

    def pullback_h_gradient_to_mu_h(self, h_grad: np.ndarray) -> np.ndarray:
        if h_grad.shape != self.h_matrix.shape:
            raise ValueError(f"h_grad must have shape {self.h_matrix.shape}, got {h_grad.shape}")
        return np.zeros_like(self.mu_h, dtype=np.float64)
