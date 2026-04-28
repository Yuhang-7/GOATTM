from __future__ import annotations

from typing import Callable, Literal

import numpy as np

from goattm.models.quadratic_dynamics import QuadraticDynamics
from goattm.solvers.explicit_euler import (
    rollout_explicit_euler,
    rollout_explicit_euler_tangent_from_base_rollout,
    rollout_explicit_euler_to_observation_times,
)
from goattm.solvers.implicit_midpoint import (
    RolloutResult,
    rollout_implicit_midpoint,
    rollout_implicit_midpoint_tangent_from_base_rollout,
    rollout_implicit_midpoint_to_observation_times,
)
from goattm.solvers.rk4 import (
    rollout_rk4,
    rollout_rk4_tangent_from_base_rollout,
    rollout_rk4_to_observation_times,
)


TimeIntegrator = Literal["implicit_midpoint", "explicit_euler", "rk4"]


def validate_time_integrator(time_integrator: str) -> TimeIntegrator:
    if time_integrator not in ("implicit_midpoint", "explicit_euler", "rk4"):
        raise ValueError(
            f"Unsupported time integrator '{time_integrator}'. "
            "Supported values are 'implicit_midpoint', 'explicit_euler', and 'rk4'."
        )
    return time_integrator  # type: ignore[return-value]


def rollout_to_final_time(
    dynamics: QuadraticDynamics,
    u0: np.ndarray,
    t_final: float,
    max_dt: float,
    input_function: Callable[[float], np.ndarray] | None = None,
    time_integrator: TimeIntegrator = "implicit_midpoint",
    dt_shrink: float = 0.8,
    dt_min: float = 1e-10,
    tol: float = 1e-10,
    max_iter: int = 25,
) -> RolloutResult:
    integrator = validate_time_integrator(time_integrator)
    if integrator == "implicit_midpoint":
        return rollout_implicit_midpoint(
            dynamics=dynamics,
            u0=u0,
            t_final=t_final,
            dt_initial=max_dt,
            input_function=input_function,
            dt_shrink=dt_shrink,
            dt_min=dt_min,
            tol=tol,
            max_iter=max_iter,
        )
    if integrator == "explicit_euler":
        return rollout_explicit_euler(
            dynamics=dynamics,
            u0=u0,
            t_final=t_final,
            max_dt=max_dt,
            input_function=input_function,
        )
    return rollout_rk4(
        dynamics=dynamics,
        u0=u0,
        t_final=t_final,
        max_dt=max_dt,
        input_function=input_function,
    )


def rollout_to_observation_times(
    dynamics: QuadraticDynamics,
    u0: np.ndarray,
    observation_times: np.ndarray,
    max_dt: float,
    input_function: Callable[[float], np.ndarray] | None = None,
    time_integrator: TimeIntegrator = "implicit_midpoint",
    dt_shrink: float = 0.8,
    dt_min: float = 1e-10,
    tol: float = 1e-10,
    max_iter: int = 25,
) -> tuple[RolloutResult, np.ndarray]:
    integrator = validate_time_integrator(time_integrator)
    if integrator == "implicit_midpoint":
        return rollout_implicit_midpoint_to_observation_times(
            dynamics=dynamics,
            u0=u0,
            observation_times=observation_times,
            max_dt=max_dt,
            input_function=input_function,
            dt_shrink=dt_shrink,
            dt_min=dt_min,
            tol=tol,
            max_iter=max_iter,
        )
    if integrator == "explicit_euler":
        return rollout_explicit_euler_to_observation_times(
            dynamics=dynamics,
            u0=u0,
            observation_times=observation_times,
            max_dt=max_dt,
            input_function=input_function,
        )
    return rollout_rk4_to_observation_times(
        dynamics=dynamics,
        u0=u0,
        observation_times=observation_times,
        max_dt=max_dt,
        input_function=input_function,
    )


def rollout_tangent_from_base_rollout(
    dynamics: QuadraticDynamics,
    base_rollout: RolloutResult,
    parameter_action: Callable[[np.ndarray, float], np.ndarray] | None = None,
    input_function: Callable[[float], np.ndarray] | None = None,
    time_integrator: TimeIntegrator = "implicit_midpoint",
) -> np.ndarray:
    integrator = validate_time_integrator(time_integrator)
    if integrator == "implicit_midpoint":
        return rollout_implicit_midpoint_tangent_from_base_rollout(
            dynamics=dynamics,
            base_rollout=base_rollout,
            parameter_action=parameter_action,
        )
    if integrator == "explicit_euler":
        return rollout_explicit_euler_tangent_from_base_rollout(
            dynamics=dynamics,
            base_rollout=base_rollout,
            parameter_action=parameter_action,
        )
    return rollout_rk4_tangent_from_base_rollout(
        dynamics=dynamics,
        base_rollout=base_rollout,
        parameter_action=parameter_action,
        input_function=input_function,
    )
