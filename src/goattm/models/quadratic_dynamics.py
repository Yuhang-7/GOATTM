from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from goattm.core.parametrization import compressed_h_gradient_to_mu_h, compressed_h_to_mu_h, mu_h_dimension, mu_h_to_compressed_h
from goattm.core.quadratic import energy_preserving_defect, quadratic_bilinear_action_matrix, quadratic_eval, quadratic_jacobian_matrix


@dataclass(frozen=True)
class QuadraticDynamics:
    a: np.ndarray
    mu_h: np.ndarray
    c: np.ndarray
    b: np.ndarray | None = None
    _h_matrix: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.a.ndim != 2 or self.a.shape[0] != self.a.shape[1]:
            raise ValueError(f"a must be square, got shape {self.a.shape}")

        r = self.a.shape[0]
        if self.c.ndim != 1 or self.c.shape[0] != r:
            raise ValueError(f"c must have shape ({r},), got {self.c.shape}")
        if self.mu_h.ndim != 1 or self.mu_h.shape[0] != mu_h_dimension(r):
            raise ValueError(f"mu_h must have shape ({mu_h_dimension(r)},), got {self.mu_h.shape}")
        if self.b is not None:
            if self.b.ndim != 2 or self.b.shape[0] != r:
                raise ValueError(f"b must have shape ({r}, dp), got {self.b.shape}")

        object.__setattr__(self, "_h_matrix", mu_h_to_compressed_h(self.mu_h.astype(np.float64), r))

    @property
    def dimension(self) -> int:
        return self.a.shape[0]

    @property
    def input_dimension(self) -> int:
        return 0 if self.b is None else self.b.shape[1]

    @property
    def h_matrix(self) -> np.ndarray:
        return self._h_matrix

    @classmethod
    def from_h_matrix(
        cls,
        a: np.ndarray,
        h_matrix: np.ndarray,
        c: np.ndarray,
        b: np.ndarray | None = None,
    ) -> "QuadraticDynamics":
        if h_matrix.ndim != 2:
            raise ValueError(f"h_matrix must be rank-2, got shape {h_matrix.shape}")
        r = a.shape[0]
        return cls(a=a, mu_h=compressed_h_to_mu_h(h_matrix.astype(np.float64), r), c=c, b=b)

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
        return self.a @ u + quadratic_eval(self.h_matrix, u) + self.forcing(p)

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
        return quadratic_eval(self.h_matrix, u, v)

    def quadratic_jacobian(self, u: np.ndarray) -> np.ndarray:
        self.validate_state(u)
        return quadratic_jacobian_matrix(self.h_matrix, u)

    def quadratic_bilinear_action_matrix(self, u: np.ndarray) -> np.ndarray:
        self.validate_state(u)
        return quadratic_bilinear_action_matrix(self.h_matrix, u)

    def rhs_jacobian(self, u: np.ndarray) -> np.ndarray:
        self.validate_state(u)
        return self.a + self.quadratic_jacobian(u)

    def energy_preserving_defect(self, u: np.ndarray) -> float:
        self.validate_state(u)
        return energy_preserving_defect(self.h_matrix, u)

    def pullback_h_gradient_to_mu_h(self, h_grad: np.ndarray) -> np.ndarray:
        if h_grad.shape != self.h_matrix.shape:
            raise ValueError(f"h_grad must have shape {self.h_matrix.shape}, got {h_grad.shape}")
        return compressed_h_gradient_to_mu_h(h_grad.astype(np.float64), self.dimension)
