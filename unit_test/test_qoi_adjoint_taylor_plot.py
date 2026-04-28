from __future__ import annotations

import sys
from pathlib import Path
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goattm.core.parametrization import (  # noqa: E402
    compressed_quadratic_dimension,
    mu_h_dimension,
    skew_symmetric_dimension,
    upper_triangular_dimension,
)
from goattm.losses.qoi_loss import rollout_qoi_loss_and_gradients  # noqa: E402
from goattm.models.quadratic_decoder import QuadraticDecoder  # noqa: E402
from goattm.models.stabilized_quadratic_dynamics import StabilizedQuadraticDynamics  # noqa: E402


OUTPUT_DIR = ROOT / "unit_test" / "output_plots" / "qoi_adjoint_taylor"


class QoiAdjointTaylorPlotTest(unittest.TestCase):
    def test_generate_qoi_adjoint_taylor_curve(self) -> None:
        rng = np.random.default_rng(9201)
        r = 15
        dq = 10
        dp = 1

        def pack_upper_triangular(matrix: np.ndarray) -> np.ndarray:
            out = []
            for i in range(matrix.shape[0]):
                for j in range(i, matrix.shape[1]):
                    out.append(matrix[i, j])
            return np.asarray(out, dtype=float)

        def pack_skew_upper(matrix: np.ndarray) -> np.ndarray:
            out = []
            for i in range(matrix.shape[0]):
                for j in range(i + 1, matrix.shape[1]):
                    out.append(matrix[i, j])
            return np.asarray(out, dtype=float)

        # Handcrafted "bad" parameters:
        # - S has highly separated diagonal scales, so -S S^T contributes a stiff,
        #   strongly anisotropic linear block.
        # - W adds nontrivial skew couplings on top of that.
        # - mu_H is scaled up to make the quadratic term noticeably larger.
        s_matrix = np.zeros((r, r), dtype=float)
        diag_scales = np.geomspace(12.0, 2.0e-2, r)
        for i in range(r):
            s_matrix[i, i] = diag_scales[i]
            for j in range(i + 1, r):
                distance = j - i
                s_matrix[i, j] = 0.18 * ((-1.0) ** distance) * np.sqrt(diag_scales[i] * diag_scales[j]) / distance

        w_matrix = np.zeros((r, r), dtype=float)
        for i in range(r):
            for j in range(i + 1, r):
                value = 1.5 * np.sin(i + 2.0 * j) / (1.0 + 0.15 * (j - i))
                w_matrix[i, j] = value
                w_matrix[j, i] = -value

        s_params = pack_upper_triangular(s_matrix)
        w_params = pack_skew_upper(w_matrix)
        mu_h = 0.7 * rng.standard_normal(mu_h_dimension(r))
        b = (0.2 * np.geomspace(1.0, 1.0e-2, r)).reshape(r, 1)
        c = 0.05 * np.geomspace(1.0, 1.0e-3, r)
        decoder = QuadraticDecoder(
            v1=0.12 * rng.standard_normal((dq, r)),
            v2=0.08 * rng.standard_normal((dq, compressed_quadratic_dimension(r))),
            v0=np.linspace(-0.2, 0.25, dq, dtype=float),
        )
        u0 = 0.01 * rng.standard_normal(r)
        t_final = 6.0e-4
        dt_initial = 2.0e-4
        qoi_observations = rng.standard_normal((4, dq))

        def input_function(t: float) -> np.ndarray:
            return np.array([0.25 - 0.05 * t], dtype=float)

        def loss_from_dynamics(
            s_vec: np.ndarray,
            w_vec: np.ndarray,
            mu_vec: np.ndarray,
            b_mat: np.ndarray,
            c_vec: np.ndarray,
        ) -> float:
            dynamics = StabilizedQuadraticDynamics(
                s_params=s_vec,
                w_params=w_vec,
                mu_h=mu_vec,
                b=b_mat,
                c=c_vec,
            )
            return rollout_qoi_loss_and_gradients(
                dynamics=dynamics,
                decoder=decoder,
                u0=u0,
                t_final=t_final,
                dt_initial=dt_initial,
                qoi_observations=qoi_observations,
                input_function=input_function,
            ).loss

        result = rollout_qoi_loss_and_gradients(
            dynamics=StabilizedQuadraticDynamics(
                s_params=s_params,
                w_params=w_params,
                mu_h=mu_h,
                b=b,
                c=c,
            ),
            decoder=decoder,
            u0=u0,
            t_final=t_final,
            dt_initial=dt_initial,
            qoi_observations=qoi_observations,
            input_function=input_function,
        )
        explicit_dynamics = StabilizedQuadraticDynamics(
            s_params=s_params,
            w_params=w_params,
            mu_h=mu_h,
            b=b,
            c=c,
        ).explicit_dynamics
        a_eigs = np.linalg.eigvals(explicit_dynamics.a)
        h_norm = float(np.linalg.norm(explicit_dynamics.h_matrix, ord=2))
        eig_abs = np.abs(a_eigs)
        eig_scale_ratio = float(np.max(eig_abs) / max(np.min(eig_abs), 1e-14))

        ds = rng.standard_normal(s_params.shape)
        dw = rng.standard_normal(w_params.shape)
        dmu = rng.standard_normal(mu_h.shape)
        db = rng.standard_normal(b.shape)
        dc = rng.standard_normal(c.shape)

        ds /= max(np.linalg.norm(ds), 1e-14)
        dw /= max(np.linalg.norm(dw), 1e-14)
        dmu /= max(np.linalg.norm(dmu), 1e-14)
        db /= max(np.linalg.norm(db), 1e-14)
        dc /= max(np.linalg.norm(dc), 1e-14)

        directional_derivative = (
            float(np.dot(result.dynamics_gradients["s_params"], ds))
            + float(np.dot(result.dynamics_gradients["w_params"], dw))
            + float(np.dot(result.dynamics_gradients["mu_h"], dmu))
            + float(np.sum(result.dynamics_gradients["b"] * db))
            + float(np.dot(result.dynamics_gradients["c"], dc))
        )

        eps_values = np.logspace(-9, -2, 32)
        curve_values = np.zeros_like(eps_values)

        for i, eps in enumerate(eps_values):
            trial_loss = loss_from_dynamics(
                s_params + eps * ds,
                w_params + eps * dw,
                mu_h + eps * dmu,
                b + eps * db,
                c + eps * dc,
            )
            curve_values[i] = (trial_loss - result.loss - eps * directional_derivative) / eps

        abs_curve_values = np.abs(curve_values)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        figure_path = OUTPUT_DIR / "qoi_adjoint_taylor_curve_bad_params.png"
        data_path = OUTPUT_DIR / "qoi_adjoint_taylor_curve_bad_params_data.npz"

        fig, axes = plt.subplots(2, 1, figsize=(7.0, 8.0), constrained_layout=True)

        axes[0].plot(eps_values, curve_values, marker="o", markersize=3, linewidth=1.0)
        axes[0].set_xscale("log")
        axes[0].set_yscale("symlog", linthresh=max(np.max(abs_curve_values) * 1e-6, 1e-16))
        axes[0].set_xlabel("eps")
        axes[0].set_ylabel(r"$(J(x+\epsilon dx)-J(x)-\epsilon \nabla J(x)^T dx)/\epsilon$")
        axes[0].set_title("QoI adjoint Taylor remainder per epsilon (bad params)")
        axes[0].grid(True, which="both", alpha=0.3)

        axes[1].plot(eps_values, abs_curve_values, marker="o", markersize=3, linewidth=1.0)
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("eps")
        axes[1].set_ylabel("absolute value")
        axes[1].set_title("Absolute Taylor remainder per epsilon")
        axes[1].grid(True, which="both", alpha=0.3)

        fig.suptitle(
            "Bad-parameter QoI Taylor test\n"
            f"H-norm={h_norm:.3e}, |eig(A)| scale ratio={eig_scale_ratio:.3e}",
            fontsize=10,
        )

        fig.savefig(figure_path, dpi=200)
        plt.close(fig)

        np.savez(
            data_path,
            eps_values=eps_values,
            curve_values=curve_values,
            abs_curve_values=abs_curve_values,
            directional_derivative=np.array([directional_derivative], dtype=float),
            base_loss=np.array([result.loss], dtype=float),
            s_params=s_params,
            w_params=w_params,
            mu_h=mu_h,
            b=b,
            c=c,
            u0=u0,
            a_eigs=a_eigs,
            eig_scale_ratio=np.array([eig_scale_ratio], dtype=float),
            h_norm=np.array([h_norm], dtype=float),
        )

        self.assertTrue(figure_path.exists())
        self.assertTrue(data_path.exists())
        self.assertTrue(np.all(np.isfinite(curve_values)))
        self.assertGreater(float(abs_curve_values.max()), 0.0)


if __name__ == "__main__":
    unittest.main()
