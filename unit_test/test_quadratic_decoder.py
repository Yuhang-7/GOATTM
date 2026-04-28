from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goattm.core.parametrization import compressed_quadratic_dimension, quadratic_features  # noqa: E402
from goattm.models.quadratic_decoder import QuadraticDecoder  # noqa: E402


class QuadraticDecoderTest(unittest.TestCase):
    def test_decode_matches_manual_formula(self) -> None:
        rng = np.random.default_rng(51)
        dq = 3
        r = 4
        v1 = rng.standard_normal((dq, r))
        v2 = rng.standard_normal((dq, compressed_quadratic_dimension(r)))
        v0 = rng.standard_normal(dq)
        u = rng.standard_normal(r)

        decoder = QuadraticDecoder(v1=v1, v2=v2, v0=v0)
        expected = v1 @ u + v2 @ quadratic_features(u) + v0
        np.testing.assert_allclose(decoder.decode(u), expected)

    def test_decoder_jacobian_matches_finite_difference(self) -> None:
        rng = np.random.default_rng(52)
        dq = 2
        r = 5
        v1 = rng.standard_normal((dq, r))
        v2 = rng.standard_normal((dq, compressed_quadratic_dimension(r)))
        v0 = rng.standard_normal(dq)
        u = rng.standard_normal(r)
        direction = rng.standard_normal(r)
        eps = 1e-7

        decoder = QuadraticDecoder(v1=v1, v2=v2, v0=v0)
        fd = (decoder.decode(u + eps * direction) - decoder.decode(u - eps * direction)) / (2.0 * eps)
        jacobian_action = decoder.jacobian(u) @ direction
        np.testing.assert_allclose(jacobian_action, fd, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
