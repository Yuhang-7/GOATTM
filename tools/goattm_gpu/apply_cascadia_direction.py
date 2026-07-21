from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


DEFAULT_GOATTM_ROOT = Path("/global/homes/y/yuuuhang/quad_goattm")
GOATTM_ROOT = Path(os.environ.get("QUAD_GOATTM_ROOT", str(DEFAULT_GOATTM_ROOT))).expanduser()
sys.path.insert(0, str(GOATTM_ROOT))

from tools.train_cascadia_packed import (  # noqa: E402
    MaskedCrossQuadraticReadoutDecoder,
    apply_pod_initializer,
    load_packed_payload,
    make_dynamics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a saved Cascadia direction to a checkpoint.")
    parser.add_argument("--direction-file", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output-checkpoint", required=True)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    direction_payload = torch.load(Path(args.direction_file), map_location="cpu")
    source_checkpoint = torch.load(Path(direction_payload["checkpoint"]), map_location="cpu")
    metadata = dict(source_checkpoint.get("metadata", {}))

    train_payload, _ = load_packed_payload(Path(metadata["train_packed"]))
    input_dim = int(train_payload["input_values"].shape[-1])
    output_dim = int(train_payload["qoi"].shape[-1])
    latent_dim = int(metadata["latent_dim"])
    dynamics = make_dynamics(
        latent_dim,
        input_dim,
        0.01,
        torch.device("cpu"),
        linear_a=metadata["linear_a"],
        a_rank=int(metadata["linear_a_rank"]),
        damping_init=0.1,
        damping_shift=0.0,
        quadratic=metadata["quadratic"],
        h_reduced_rank=int(metadata["h_reduced_rank"]),
        h_tt_rank=int(metadata["h_tt_rank"]),
    )
    decoder = MaskedCrossQuadraticReadoutDecoder(
        latent_dim,
        output_dim,
        cross_terms=int(metadata["decoder_cross_terms"]),
        mask_seed=int(metadata["decoder_mask_seed"]),
        bias=True,
    ).double()
    apply_pod_initializer(dynamics, decoder, Path(metadata["initializer"]["path"]))
    dynamics.load_state_dict(source_checkpoint["dynamics_state_dict"])
    decoder.load_state_dict(source_checkpoint["decoder_state_dict"])

    names = tuple(direction_payload["names"])
    values = tuple(direction_payload["values"])
    direction = tuple(direction_payload["newton_direction"])
    by_name = {
        full_name.removeprefix("dynamics."): value + float(args.alpha) * delta
        for full_name, value, delta in zip(names, values, direction)
    }
    params = dict(dynamics.named_parameters())
    with torch.no_grad():
        for name, value in by_name.items():
            params[name].copy_(value)

    metadata = {
        **metadata,
        "applied_second_order_direction": {
            "direction_file": str(Path(args.direction_file)),
            "source_checkpoint": str(direction_payload["checkpoint"]),
            "alpha": float(args.alpha),
        },
    }
    out = Path(args.output_checkpoint)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": metadata,
            "dynamics_state_dict": dynamics.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
        },
        out,
    )
    print(json.dumps({"output_checkpoint": str(out), "alpha": float(args.alpha)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
