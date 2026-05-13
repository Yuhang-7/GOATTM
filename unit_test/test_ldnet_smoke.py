from __future__ import annotations

import torch

from goattm.ldnet import (
    AlternatingLDNetConfig,
    LDNetBatch,
    LDNetModel,
    SelectiveSSMDecoder,
    SelectiveSSMDecoderConfig,
    TorchQuadraticLatentDynamics,
    alternating_train_ldnet,
)
from goattm.ldnet.mpi import TorchMPIContext


def test_ldnet_forward_and_short_training_smoke() -> None:
    torch.set_num_threads(1)
    dtype = torch.float64
    times = torch.linspace(0.0, 0.5, 8, dtype=dtype)
    u0 = torch.randn(5, 3, dtype=dtype) * 0.1
    qoi = torch.randn(5, 8, 2, dtype=dtype) * 0.1
    batch = LDNetBatch(times=times, u0=u0, qoi=qoi)
    dynamics = TorchQuadraticLatentDynamics(3, dtype=dtype, h_scale=1e-4)
    decoder = SelectiveSSMDecoder(SelectiveSSMDecoderConfig(latent_dim=3, qoi_dim=2, state_dim=8, dtype=dtype))
    model = LDNetModel(dynamics, decoder)
    prediction = model(batch)
    assert prediction.shape == qoi.shape

    config = AlternatingLDNetConfig(
        outer_cycles=1,
        dynamics_steps_per_cycle=1,
        decoder_steps_per_cycle=1,
        dynamics_learning_rate=1e-3,
        decoder_learning_rate=1e-3,
        echo_progress=False,
    )
    history = alternating_train_ldnet(model, batch, config, context=TorchMPIContext())
    assert len(history.records) == 2
