from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from goattm.core.parametrization import (
    a_from_stabilized_params,
    a_gradient_to_stabilized_params,
    mu_h_dimension,
    skew_symmetric_dimension,
    upper_triangular_dimension,
)
from goattm.models.quadratic_dynamics import QuadraticDynamics


@dataclass(frozen=True)
class StabilizedQuadraticDynamics:
    s_params: np.ndarray
    w_params: np.ndarray
    mu_h: np.ndarray
    c: np.ndarray
    b: np.ndarray | None = None
    _explicit_dynamics: QuadraticDynamics = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.c.ndim != 1:
            raise ValueError(f"c must be rank-1, got shape {self.c.shape}")

        r = self.c.shape[0]
        if self.s_params.ndim != 1 or self.s_params.shape[0] != upper_triangular_dimension(r):
            raise ValueError(
                f"s_params must have shape ({upper_triangular_dimension(r)},), got {self.s_params.shape}"
            )
        if self.w_params.ndim != 1 or self.w_params.shape[0] != skew_symmetric_dimension(r):
            raise ValueError(
                f"w_params must have shape ({skew_symmetric_dimension(r)},), got {self.w_params.shape}"
            )
        if self.mu_h.ndim != 1 or self.mu_h.shape[0] != mu_h_dimension(r):
            raise ValueError(f"mu_h must have shape ({mu_h_dimension(r)},), got {self.mu_h.shape}")
        if self.b is not None and (self.b.ndim != 2 or self.b.shape[0] != r):
            raise ValueError(f"b must have shape ({r}, dp), got {self.b.shape}")

        a_matrix = a_from_stabilized_params(
            self.s_params.astype(np.float64),
            self.w_params.astype(np.float64),
            r,
        )
        object.__setattr__(
            self,
            "_explicit_dynamics",
            QuadraticDynamics(a=a_matrix, mu_h=self.mu_h, c=self.c, b=self.b),
        )

    @property
    def explicit_dynamics(self) -> QuadraticDynamics:
        return self._explicit_dynamics

    @property
    def a(self) -> np.ndarray:
        return self.explicit_dynamics.a

    @property
    def h_matrix(self) -> np.ndarray:
        return self.explicit_dynamics.h_matrix

    @property
    def dimension(self) -> int:
        return self.explicit_dynamics.dimension

    @property
    def input_dimension(self) -> int:
        return self.explicit_dynamics.input_dimension

    def validate_state(self, u: np.ndarray, name: str = "u") -> None:
        self.explicit_dynamics.validate_state(u, name)

    def validate_input(self, p: np.ndarray, name: str = "p") -> None:
        self.explicit_dynamics.validate_input(p, name)

    def forcing(self, p: np.ndarray | None = None) -> np.ndarray:
        return self.explicit_dynamics.forcing(p)

    def rhs(self, u: np.ndarray, p: np.ndarray | None = None) -> np.ndarray:
        return self.explicit_dynamics.rhs(u, p=p)

    def rhs_at_time(
        self,
        u: np.ndarray,
        t: float,
        input_function: Callable[[float], np.ndarray] | None = None,
    ) -> np.ndarray:
        return self.explicit_dynamics.rhs_at_time(u, t, input_function=input_function)

    def quadratic_bilinear(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return self.explicit_dynamics.quadratic_bilinear(u, v)

    def quadratic_jacobian(self, u: np.ndarray) -> np.ndarray:
        return self.explicit_dynamics.quadratic_jacobian(u)

    def quadratic_bilinear_action_matrix(self, u: np.ndarray) -> np.ndarray:
        return self.explicit_dynamics.quadratic_bilinear_action_matrix(u)

    def rhs_jacobian(self, u: np.ndarray) -> np.ndarray:
        return self.explicit_dynamics.rhs_jacobian(u)

    def energy_preserving_defect(self, u: np.ndarray) -> float:
        return self.explicit_dynamics.energy_preserving_defect(u)

    def pullback_h_gradient_to_mu_h(self, h_grad: np.ndarray) -> np.ndarray:
        return self.explicit_dynamics.pullback_h_gradient_to_mu_h(h_grad)

    def pullback_a_gradient_to_stabilized_params(self, a_grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if a_grad.shape != self.a.shape:
            raise ValueError(f"a_grad must have shape {self.a.shape}, got {a_grad.shape}")
        return a_gradient_to_stabilized_params(
            a_grad.astype(np.float64),
            self.s_params.astype(np.float64),
            self.w_params.astype(np.float64),
            self.dimension,
        )
