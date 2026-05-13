from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goattm.ldnet import (  # noqa: E402
    LDNetBatch,
    LDNetModel,
    SelectiveSSMDecoder,
    SelectiveSSMDecoderConfig,
    TorchQuadraticLatentDynamics,
)


def make_batch(dtype: torch.dtype) -> LDNetBatch:
    generator = torch.Generator().manual_seed(2026)
    times = torch.linspace(0.0, 0.4, 11, dtype=dtype)
    u0 = 0.1 * torch.randn(4, 3, generator=generator, dtype=dtype)
    qoi = torch.randn(4, 11, 2, generator=generator, dtype=dtype)
    return LDNetBatch(times=times, u0=u0, qoi=qoi)


def loss_value(model: LDNetModel, batch: LDNetBatch) -> torch.Tensor:
    return F.mse_loss(model(batch), batch.qoi, reduction="mean")


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(2027)
    dtype = torch.float64
    batch = make_batch(dtype)
    dynamics = TorchQuadraticLatentDynamics(3, dtype=dtype, h_scale=1e-4)
    decoder = SelectiveSSMDecoder(SelectiveSSMDecoderConfig(latent_dim=3, qoi_dim=2, state_dim=5, dtype=dtype))
    model = LDNetModel(dynamics, decoder)
    parameters = [param for param in model.parameters() if param.requires_grad]

    base_loss = loss_value(model, batch)
    base_loss.backward()

    generator = torch.Generator().manual_seed(2028)
    directions = [torch.randn(param.shape, generator=generator, dtype=param.dtype) for param in parameters]
    norm = torch.sqrt(sum(torch.sum(direction * direction) for direction in directions))
    directions = [direction / norm for direction in directions]
    directional_derivative = sum(torch.sum(param.grad * direction) for param, direction in zip(parameters, directions, strict=True))

    with torch.no_grad():
        base_values = [param.detach().clone() for param in parameters]

    print("eps, first_order_residual, residual/eps, residual/eps^2")
    previous_second = None
    for eps in [1e-1, 5e-2, 2.5e-2, 1.25e-2, 6.25e-3, 3.125e-3]:
        with torch.no_grad():
            for param, base, direction in zip(parameters, base_values, directions, strict=True):
                param.copy_(base + eps * direction)
        perturbed_loss = loss_value(model, batch)
        residual = torch.abs(perturbed_loss - base_loss - eps * directional_derivative).detach()
        first_scaled = residual / eps
        second_scaled = residual / (eps * eps)
        ratio_text = ""
        if previous_second is not None:
            ratio_text = f", second_scaled_ratio={float(second_scaled / previous_second):.3f}"
        previous_second = second_scaled.detach()
        print(
            f"{eps:.6e}, {float(residual):.6e}, {float(first_scaled):.6e}, {float(second_scaled):.6e}{ratio_text}",
            flush=True,
        )

    with torch.no_grad():
        for param, base in zip(parameters, base_values, strict=True):
            param.copy_(base)


if __name__ == "__main__":
    main()
