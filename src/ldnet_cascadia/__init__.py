"""Neural-decoder latent dynamics tools for Cascadia packed datasets."""

from .data import CascadiaPackedBatch, batch_from_payload, load_packed_batch, load_packed_payload
from .diagnostics import finite_difference_check
from .initialization import apply_pod_initializer
from .losses import LDNetLossConfig, dense_a_symmetric_energy_penalty, ldnet_loss, relative_error
from .models import (
    CascadiaLDNet,
    InitialConditionEncoder,
    MLP,
    NeuralDecoder,
    make_denseA_lowrankH_neural_decoder_model,
)
from .optim import armijo_step, blockwise_armijo_step, grad_norm, named_parameter_blocks

__all__ = [
    "CascadiaLDNet",
    "CascadiaPackedBatch",
    "InitialConditionEncoder",
    "LDNetLossConfig",
    "MLP",
    "NeuralDecoder",
    "apply_pod_initializer",
    "armijo_step",
    "batch_from_payload",
    "blockwise_armijo_step",
    "dense_a_symmetric_energy_penalty",
    "finite_difference_check",
    "grad_norm",
    "ldnet_loss",
    "load_packed_batch",
    "load_packed_payload",
    "make_denseA_lowrankH_neural_decoder_model",
    "named_parameter_blocks",
    "relative_error",
]
