from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from goattm.core.parametrization import (
    compressed_h_gradient_to_skew_cp,
    skew_cp_parameter_action,
    skew_cp_quadratic_eval,
    skew_cp_quadratic_jacobian_matrix,
    skew_cp_to_compressed_h,
)


@dataclass(frozen=True)
class SkewCPQuadraticDynamics:
    a: np.ndarray
    skew_u: np.ndarray
    skew_v: np.ndarray
    skew_z: np.ndarray
    c: np.ndarray
    b: np.ndarray | None = None
    _h_matrix: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.a.ndim != 2 or self.a.shape[0] != self.a.shape[1]:
            raise ValueError(f"a must be square, got shape {self.a.shape}")

        d = self.a.shape[0]
        expected_factor_shape = (d, self.skew_u.shape[1]) if self.skew_u.ndim == 2 else None
        if self.skew_u.ndim != 2 or self.skew_u.shape[0] != d:
            raise ValueError(f"skew_u must have shape ({d}, R), got {self.skew_u.shape}")
        if self.skew_v.shape != expected_factor_shape:
            raise ValueError(f"skew_v must have shape {expected_factor_shape}, got {self.skew_v.shape}")
        if self.skew_z.shape != expected_factor_shape:
            raise ValueError(f"skew_z must have shape {expected_factor_shape}, got {self.skew_z.shape}")
        if self.c.ndim != 1 or self.c.shape[0] != d:
            raise ValueError(f"c must have shape ({d},), got {self.c.shape}")
        if self.b is not None and (self.b.ndim != 2 or self.b.shape[0] != d):
            raise ValueError(f"b must have shape ({d}, dp), got {self.b.shape}")

        object.__setattr__(
            self,
            "_h_matrix",
            skew_cp_to_compressed_h(
                self.skew_u.astype(np.float64),
                self.skew_v.astype(np.float64),
                self.skew_z.astype(np.float64),
            ),
        )

    @property
    def dimension(self) -> int:
        return self.a.shape[0]

    @property
    def quadratic_rank(self) -> int:
        return self.skew_u.shape[1]

    @property
    def input_dimension(self) -> int:
        return 0 if self.b is None else self.b.shape[1]

    @property
    def h_matrix(self) -> np.ndarray:
        return self._h_matrix

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

    def quadratic(self, u: np.ndarray) -> np.ndarray:
        self.validate_state(u)
        return skew_cp_quadratic_eval(
            self.skew_u.astype(np.float64),
            self.skew_v.astype(np.float64),
            self.skew_z.astype(np.float64),
            u.astype(np.float64),
        )

    def rhs(self, u: np.ndarray, p: np.ndarray | None = None) -> np.ndarray:
        self.validate_state(u)
        return self.a @ u + self.quadratic(u) + self.forcing(p)

    def rhs_at_time(
        self,
        u: np.ndarray,
        t: float,
        input_function: Callable[[float], np.ndarray] | None = None,
    ) -> np.ndarray:
        p = None if input_function is None else np.asarray(input_function(t), dtype=np.float64)
        return self.rhs(u, p=p)

    def quadratic_jacobian(self, u: np.ndarray) -> np.ndarray:
        self.validate_state(u)
        return skew_cp_quadratic_jacobian_matrix(
            self.skew_u.astype(np.float64),
            self.skew_v.astype(np.float64),
            self.skew_z.astype(np.float64),
            u.astype(np.float64),
        )

    def quadratic_bilinear(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        self.validate_state(u, "u")
        self.validate_state(v, "v")
        alpha_u = self.skew_u.T @ u
        beta_u = self.skew_v.T @ u
        gamma_u = self.skew_z.T @ u
        alpha_v = self.skew_u.T @ v
        beta_v = self.skew_v.T @ v
        gamma_v = self.skew_z.T @ v
        return (
            self.skew_u @ (gamma_u * beta_v + gamma_v * beta_u)
            - self.skew_v @ (gamma_u * alpha_v + gamma_v * alpha_u)
        )

    def quadratic_bilinear_action_matrix(self, u: np.ndarray) -> np.ndarray:
        self.validate_state(u)
        return 0.5 * self.quadratic_jacobian(u)

    def rhs_jacobian(self, u: np.ndarray) -> np.ndarray:
        self.validate_state(u)
        return self.a + self.quadratic_jacobian(u)

    def energy_preserving_defect(self, u: np.ndarray) -> float:
        self.validate_state(u)
        return float(np.dot(u, self.quadratic(u)))

    def quadratic_parameter_action(
        self,
        d_skew_u: np.ndarray,
        d_skew_v: np.ndarray,
        d_skew_z: np.ndarray,
        state: np.ndarray,
    ) -> np.ndarray:
        self.validate_state(state, "state")
        return skew_cp_parameter_action(
            self.skew_u.astype(np.float64),
            self.skew_v.astype(np.float64),
            self.skew_z.astype(np.float64),
            d_skew_u.astype(np.float64),
            d_skew_v.astype(np.float64),
            d_skew_z.astype(np.float64),
            state.astype(np.float64),
        )

    def pullback_h_gradient_to_skew_cp(self, h_grad: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return compressed_h_gradient_to_skew_cp(
            h_grad.astype(np.float64),
            self.skew_u.astype(np.float64),
            self.skew_v.astype(np.float64),
            self.skew_z.astype(np.float64),
        )
