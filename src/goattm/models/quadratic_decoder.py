from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from goattm.core.parametrization import compressed_quadratic_dimension, quadratic_features
from goattm.core.quadratic import quadratic_jacobian_matrix


@dataclass(frozen=True)
class QuadraticDecoder:
    v1: np.ndarray
    v2: np.ndarray
    v0: np.ndarray

    def __post_init__(self) -> None:
        if self.v1.ndim != 2:
            raise ValueError(f"v1 must be rank-2, got shape {self.v1.shape}")
        dq, r = self.v1.shape
        if self.v2.ndim != 2 or self.v2.shape != (dq, compressed_quadratic_dimension(r)):
            raise ValueError(
                f"v2 must have shape ({dq}, {compressed_quadratic_dimension(r)}), got {self.v2.shape}"
            )
        if self.v0.ndim != 1 or self.v0.shape[0] != dq:
            raise ValueError(f"v0 must have shape ({dq},), got {self.v0.shape}")

    @property
    def latent_dimension(self) -> int:
        return self.v1.shape[1]

    @property
    def output_dimension(self) -> int:
        return self.v1.shape[0]

    def validate_state(self, u: np.ndarray, name: str = "u") -> None:
        if u.ndim != 1 or u.shape[0] != self.latent_dimension:
            raise ValueError(f"{name} must have shape ({self.latent_dimension},), got {u.shape}")

    def decode(self, u: np.ndarray) -> np.ndarray:
        self.validate_state(u)
        return self.v1 @ u + self.v2 @ quadratic_features(u) + self.v0

    def jacobian(self, u: np.ndarray) -> np.ndarray:
        self.validate_state(u)
        return self.v1 + quadratic_jacobian_matrix(self.v2, u)
