from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goattm.train import load_metrics_jsonl, plot_loss_curve  # noqa: E402


class PlotLossCurveTest(unittest.TestCase):
    def test_plot_loss_curve_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metrics_path = root / "metrics.jsonl"
            output_path = root / "loss_curve.png"
            records = [
                {
                    "iteration": 0,
                    "train_objective": 1.2,
                    "train_data_loss": 1.1,
                    "train_decoder_regularization_loss": 0.1,
                    "test_data_loss": 1.3,
                    "gradient_norm": 0.5,
                    "step_norm": 0.0,
                },
                {
                    "iteration": 1,
                    "train_objective": 0.8,
                    "train_data_loss": 0.72,
                    "train_decoder_regularization_loss": 0.08,
                    "test_data_loss": 0.95,
                    "gradient_norm": 0.22,
                    "step_norm": 0.11,
                },
            ]
            metrics_path.write_text(
                "\n".join(json.dumps(record, ensure_ascii=True) for record in records) + "\n",
                encoding="utf-8",
            )

            parsed = load_metrics_jsonl(metrics_path)
            self.assertEqual(len(parsed), 2)

            written_path = plot_loss_curve(metrics_path, output_path=output_path)
            self.assertEqual(written_path, output_path)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
