from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_cascadia_packed import MaskedCrossQuadraticReadoutDecoder, make_dynamics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Print Cascadia GOATTM model parameter dimensions.")
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--input-dim", type=int, default=150)
    parser.add_argument("--output-dim", type=int, default=50)
    parser.add_argument("--linear-a", choices=("dense", "dissipative_skew"), default="dissipative_skew")
    parser.add_argument("--a-rank", type=int, default=20)
    parser.add_argument("--a-damping-init", type=float, default=0.1)
    parser.add_argument("--a-damping-shift", type=float, default=0.0)
    parser.add_argument("--quadratic", choices=("energy_dense", "energy_tucker"), default="energy_tucker")
    parser.add_argument("--h-reduced-rank", type=int, default=50)
    parser.add_argument("--h-tt-rank", type=int, default=16)
    parser.add_argument("--decoder-cross-terms", type=int, default=2000)
    parser.add_argument("--decoder-mask-seed", type=int, default=20260705)
    args = parser.parse_args()

    dynamics = make_dynamics(
        args.latent_dim,
        args.input_dim,
        0.01,
        torch.device("cpu"),
        linear_a=args.linear_a,
        a_rank=args.a_rank,
        damping_init=args.a_damping_init,
        damping_shift=args.a_damping_shift,
        quadratic=args.quadratic,
        h_reduced_rank=args.h_reduced_rank,
        h_tt_rank=args.h_tt_rank,
    )
    decoder = MaskedCrossQuadraticReadoutDecoder(
        args.latent_dim,
        args.output_dim,
        cross_terms=args.decoder_cross_terms,
        mask_seed=args.decoder_mask_seed,
        bias=True,
    ).double()

    print("DYNAMICS")
    for name, param in dynamics.named_parameters():
        print(name, tuple(param.shape), param.numel())
    print("DECODER")
    for name, param in decoder.named_parameters():
        print(name, tuple(param.shape), param.numel())
    dynamics_total = sum(param.numel() for param in dynamics.parameters() if param.requires_grad)
    decoder_total = sum(param.numel() for param in decoder.parameters() if param.requires_grad)
    print("dynamics_total", dynamics_total)
    print("decoder_total", decoder_total)
    print("total", dynamics_total + decoder_total)
    print("decoder_feature_dim", decoder.feature_dim)
    print("cross_terms", decoder.quadratic_i.numel())


if __name__ == "__main__":
    main()
