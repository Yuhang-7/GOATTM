from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from quadrode_gpu_goattm import DenseLinearA

from .models import CascadiaLDNet


def apply_pod_initializer(model: CascadiaLDNet, path: str | Path) -> dict[str, Any]:
    """Install POD/OpInf parameters into an LDNet model.

    The initializer is the same npz used by the variable-projection Cascadia
    pipeline.  It initializes dense A, source B/c, and the decoder linear
    readout.  The neural correction is left at its zero initialization, so the
    initial decoder is exactly the POD/OpInf linear readout.
    """

    data = np.load(Path(path), allow_pickle=True)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    source_dim = int(data["a_matrix"].shape[0])
    latent_dim = int(model.dynamics.latent_dim)
    if source_dim > latent_dim:
        raise ValueError(f"initializer latent_dim {source_dim} exceeds model latent_dim {latent_dim}")

    with torch.no_grad():
        a_matrix = torch.as_tensor(data["a_matrix"], device=device, dtype=dtype)
        if not isinstance(model.dynamics.linear, DenseLinearA):
            raise TypeError("POD initializer currently expects dense A")
        model.dynamics.linear.A.zero_()
        model.dynamics.linear.A[:source_dim, :source_dim].copy_(a_matrix)

        b_matrix = torch.as_tensor(data["b_matrix"], device=device, dtype=dtype)
        rows = min(source_dim, model.dynamics.source.B.shape[0])
        cols = min(int(b_matrix.shape[1]), model.dynamics.source.B.shape[1])
        model.dynamics.source.B.zero_()
        model.dynamics.source.B[:rows, :cols].copy_(b_matrix[:rows, :cols])
        if model.dynamics.source.c is not None:
            c_vector = torch.as_tensor(data["c_vector"], device=device, dtype=dtype)
            model.dynamics.source.c.zero_()
            model.dynamics.source.c[:rows].copy_(c_vector[:rows])

        if hasattr(model.decoder, "init_linear_readout"):
            weight = torch.as_tensor(data["decoder_template_v1"], device=device, dtype=dtype)
            bias = torch.as_tensor(data["decoder_template_v0"], device=device, dtype=dtype)
            model.decoder.init_linear_readout(weight, bias)
        else:
            raise TypeError("decoder does not expose init_linear_readout")

    return {
        "path": str(path),
        "source_latent_dim": source_dim,
        "target_latent_dim": latent_dim,
        "linear_init": "dense_opinf_block",
        "source_init": "opinf_B_c",
        "decoder_init": "linear_readout_from_decoder_template_v0_v1",
        "correction_init": "zero",
        "max_real_before_shift": float(data["max_real_before_shift"]) if "max_real_before_shift" in data else None,
        "max_real_after_shift": float(data["max_real_after_shift"]) if "max_real_after_shift" in data else None,
        "symmetric_part_lambda_max": float(data["symmetric_part_lambda_max"]) if "symmetric_part_lambda_max" in data else None,
    }
