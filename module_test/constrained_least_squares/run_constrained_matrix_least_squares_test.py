from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goattm.core.parametrization import mu_h_dimension, mu_h_to_compressed_h, quadratic_features  # noqa: E402
from goattm.core.quadratic import energy_preserving_defect  # noqa: E402
from goattm.preprocess.constrained_least_squares import (  # noqa: E402
    build_energy_preserving_compressed_h_basis,
    solve_basis_constrained_matrix_least_squares,
)


def main() -> None:
    rng = np.random.default_rng(20260428)
    r = 5
    sample_count = 250
    regularization = 1e-10

    mu_true = 0.25 * rng.standard_normal(mu_h_dimension(r))
    h_true = mu_h_to_compressed_h(mu_true, r)

    latent_states = 0.8 * rng.standard_normal((r, sample_count))
    regressor = np.column_stack([quadratic_features(latent_states[:, idx]) for idx in range(sample_count)])
    target = h_true @ regressor

    basis = build_energy_preserving_compressed_h_basis(r)
    result = solve_basis_constrained_matrix_least_squares(
        regressor=regressor,
        target=target,
        basis_matrix=basis,
        regularization=regularization,
    )

    recovered_h = result.solution_matrix
    relative_error = np.linalg.norm(recovered_h - h_true) / np.linalg.norm(h_true)
    defect_test_states = 0.7 * rng.standard_normal((r, 64))
    defect = max(abs(energy_preserving_defect(recovered_h, defect_test_states[:, idx])) for idx in range(defect_test_states.shape[1]))
    kkt_residual = np.linalg.norm(result.design_matrix.T @ (result.design_matrix @ result.coefficients - target.reshape(-1, order="F")) + regularization * result.coefficients)

    if relative_error >= 1e-8:
        raise AssertionError(f"relative recovery error is too large: {relative_error}")
    if defect >= 1e-10:
        raise AssertionError(f"energy-preserving defect is too large: {defect}")
    if kkt_residual >= 1e-8:
        raise AssertionError(f"constrained normal-equation residual is too large: {kkt_residual}")

    output_dir = ROOT / "module_test" / "output_plots" / "constrained_least_squares"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "r": r,
        "sample_count": sample_count,
        "regularization": regularization,
        "relative_error": float(relative_error),
        "energy_preserving_defect": float(defect),
        "kkt_residual": float(kkt_residual),
        "basis_shape": list(basis.shape),
    }
    summary_path = output_dir / "constrained_matrix_least_squares_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
