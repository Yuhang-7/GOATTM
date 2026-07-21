from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch


DEFAULT_GOATTM_ROOT = Path("/global/homes/y/yuuuhang/quad_goattm")
GOATTM_ROOT = Path(os.environ.get("QUAD_GOATTM_ROOT", str(DEFAULT_GOATTM_ROOT))).expanduser()
sys.path.insert(0, str(GOATTM_ROOT))

from quadrode_gpu_goattm.reduced import ReducedObjective  # noqa: E402
from quadrode_gpu_goattm.reduced_gn import (  # noqa: E402
    ReducedGNWorkspace,
    countsketch_apply,
    countsketch_apply_matrix,
    direction_norm,
    make_countsketch,
)
from quadrode_gpu_goattm.steppers import DenseLaggedMidpointStepper  # noqa: E402
from tools.train_cascadia_packed import (  # noqa: E402
    MaskedCrossQuadraticReadoutDecoder,
    apply_pod_initializer,
    batch_from_payload,
    limit_packed_payload_samples,
    load_packed_payload,
    make_dynamics,
)


def parse_counts(text: str) -> list[int]:
    return [int(piece) for piece in str(text).replace(" ", "").split(",") if piece]


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cuda_peak_mib(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / 1024**2)


def clear_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def nested_get(mapping: dict, path: str, default=None):
    value = mapping
    for piece in path.split("."):
        if not isinstance(value, dict) or piece not in value:
            return default
        value = value[piece]
    return value


def config_from_checkpoint(path: Path | None) -> tuple[dict, dict | None]:
    if path is None:
        return {}, None
    checkpoint = torch.load(path, map_location="cpu")
    return dict(checkpoint.get("metadata", {})), checkpoint


def make_decoder(latent_dim: int, output_dim: int, args: argparse.Namespace, metadata: dict, device: torch.device):
    mode = args.decoder_quadratic_mode or metadata.get("decoder_quadratic_mode", "masked_cross")
    if mode == "masked_cross":
        decoder = MaskedCrossQuadraticReadoutDecoder(
            latent_dim,
            output_dim,
            cross_terms=int(args.decoder_cross_terms or metadata.get("decoder_cross_terms") or 2000),
            mask_seed=int(args.decoder_mask_seed or metadata.get("decoder_mask_seed") or 20260705),
            bias=True,
        )
    else:
        from quadrode_gpu_goattm import QuadraticReadoutDecoder

        decoder = QuadraticReadoutDecoder(
            latent_dim,
            output_dim,
            include_quadratic=mode == "full",
            bias=True,
        )
    return decoder.double().to(device)


def build_model(input_dim: int, output_dim: int, args: argparse.Namespace, metadata: dict, checkpoint: dict | None, device: torch.device):
    latent_dim = int(args.latent_dim or metadata.get("latent_dim") or 120)
    linear_a = args.linear_a or metadata.get("linear_a", "dissipative_skew")
    linear_a_rank = int(args.a_rank or metadata.get("linear_a_rank") or 20)
    quadratic = args.quadratic or metadata.get("quadratic", "energy_tucker")
    h_reduced_rank = int(args.h_reduced_rank or metadata.get("h_reduced_rank") or 40)
    h_tt_rank = int(args.h_tt_rank or metadata.get("h_tt_rank") or 40)
    damping_shift = float(args.a_damping_shift if args.a_damping_shift is not None else 1.0)
    dynamics = make_dynamics(
        latent_dim,
        input_dim,
        float(args.init_scale),
        device,
        linear_a=linear_a,
        a_rank=linear_a_rank,
        damping_init=float(args.a_damping_init),
        damping_shift=damping_shift,
        quadratic=quadratic,
        h_reduced_rank=h_reduced_rank,
        h_tt_rank=h_tt_rank,
    )
    decoder = make_decoder(latent_dim, output_dim, args, metadata, device)
    initializer = args.initializer or nested_get(metadata, "initializer.path")
    if initializer:
        apply_pod_initializer(dynamics, decoder, Path(initializer))
    if checkpoint is not None:
        dynamics.load_state_dict(checkpoint["dynamics_state_dict"], strict=True)
        decoder.load_state_dict(checkpoint["decoder_state_dict"], strict=True)
    stepper = DenseLaggedMidpointStepper(picard_iters=int(args.picard_iters or nested_get(metadata, "optimizer.picard_iters", 2)))
    objective = ReducedObjective(
        dynamics,
        decoder,
        stepper,
        decoder_ridge=float(args.decoder_ridge if args.decoder_ridge is not None else nested_get(metadata, "optimizer.decoder_ridge", 1.0e-5)),
        dynamics_ridge=float(args.dynamics_ridge if args.dynamics_ridge is not None else nested_get(metadata, "optimizer.dynamics_ridge", 1.0e-7)),
        normal_chunk_size=int(args.normal_chunk_size if args.normal_chunk_size is not None else nested_get(metadata, "optimizer.normal_chunk_size", 4096)),
        gradient_mode="lagged_adjoint",
    )
    return objective, {
        "latent_dim": latent_dim,
        "linear_a": linear_a,
        "linear_a_rank": linear_a_rank,
        "quadratic": quadratic,
        "h_reduced_rank": h_reduced_rank,
        "h_tt_rank": h_tt_rank,
        "decoder_feature_dim": int(getattr(decoder, "feature_dim", decoder.readout.in_features)),
    }


def random_orthonormal_flat_directions(gradient: torch.Tensor, rank: int, seed: int, include_gradient: bool) -> torch.Tensor:
    gen = torch.Generator(device=gradient.device)
    gen.manual_seed(int(seed))
    vectors: list[torch.Tensor] = []
    if include_gradient:
        norm = torch.linalg.vector_norm(gradient)
        if float(norm.detach().cpu()) > 0.0:
            vectors.append(-gradient / norm)
    while len(vectors) < int(rank):
        candidate = torch.randn(gradient.shape, generator=gen, device=gradient.device, dtype=gradient.dtype)
        for vec in vectors:
            candidate = candidate - torch.dot(vec, candidate) * vec
        norm = torch.linalg.vector_norm(candidate)
        if float(norm.detach().cpu()) > 1.0e-12:
            vectors.append(candidate / norm)
    return torch.stack(vectors, dim=1)


def run_one_count(payload: dict, metadata: dict, checkpoint: dict | None, args: argparse.Namespace, sample_count: int, device: torch.device) -> dict:
    clear_cuda(device)
    start_total = time.perf_counter()
    stage = "start"
    try:
        stage = "batch_to_device"
        batch = batch_from_payload(payload, device, sample_slice=slice(0, int(sample_count)), time_mode=args.time_mode)
        input_dim = int(batch.input_values.shape[-1])
        output_dim = int(batch.qoi.shape[-1])
        stage = "build_model"
        objective, model_record = build_model(input_dim, output_dim, args, metadata, checkpoint, device)
        sync(device)
        build_start = time.perf_counter()
        stage = "build_reduced_gn_cache"
        workspace = ReducedGNWorkspace(objective, batch)
        sync(device)
        cache_seconds = time.perf_counter() - build_start
        gradient = None
        value_grad = None
        grad_seconds = 0.0
        if args.direction_source == "gradient_first":
            stage = "value_and_grad"
            grad_start = time.perf_counter()
            gradient, value_grad = workspace.gradient_flat()
            sync(device)
            grad_seconds = time.perf_counter() - grad_start
        if gradient is None:
            gradient = torch.zeros(workspace.parameter_dim, device=device, dtype=next(objective.dynamics.parameters()).dtype)
        directions = random_orthonormal_flat_directions(
            gradient,
            int(args.sketch_rank),
            int(args.seed) + int(sample_count),
            bool(args.include_gradient_vector and args.direction_source == "gradient_first"),
        )
        z_columns = []
        jvp_records = []
        sketch_indices = None
        sketch_signs = None
        residual_dim = None
        computed_columns = min(int(args.sketch_rank), int(args.max_computed_columns))
        if args.jvp_mode == "single_pass_streaming_batched_sketch":
            stage = "single_pass_streaming_batched_sketch"
            batch_jvp_start = time.perf_counter()
            result = workspace.sketched_jvp_batch_from_flat_single_pass(
                directions[:, :computed_columns],
                sketch_dim=int(args.residual_sketch_dim),
                seed=int(args.seed) + 17 * int(sample_count),
                sample_chunk_size=int(args.streaming_sample_chunk_size),
            )
            sync(device)
            residual_dim = int(result.residual_dim)
            batch_seconds = time.perf_counter() - batch_jvp_start
            z_matrix_computed = result.sketch_matrix.detach()
            z_columns.extend([z_matrix_computed[:, column] for column in range(computed_columns)])
            for column in range(computed_columns):
                direction = workspace.direction_from_flat(directions[:, column])
                jvp_records.append(
                    {
                        "column": column,
                        "seconds": batch_seconds,
                        "batch_seconds": batch_seconds,
                        "direction_norm": float(direction_norm(direction, objective.dynamics).detach().cpu()),
                        "sketched_jvp_norm": float(torch.linalg.vector_norm(result.sketch_matrix[:, column]).detach().cpu()),
                        "quadratic_form": float(result.quadratic_form[column].detach().cpu()),
                        "peak_memory_mib": cuda_peak_mib(device),
                    }
                )
            del result
        elif args.jvp_mode == "streaming_batched_sketch":
            stage = "streaming_batched_sketch"
            batch_jvp_start = time.perf_counter()
            result = workspace.sketched_jvp_batch_from_flat_streaming(
                directions[:, :computed_columns],
                sketch_dim=int(args.residual_sketch_dim),
                seed=int(args.seed) + 17 * int(sample_count),
                sample_chunk_size=int(args.streaming_sample_chunk_size),
            )
            sync(device)
            residual_dim = int(result.residual_dim)
            batch_seconds = time.perf_counter() - batch_jvp_start
            z_matrix_computed = result.sketch_matrix.detach()
            z_columns.extend([z_matrix_computed[:, column] for column in range(computed_columns)])
            for column in range(computed_columns):
                direction = workspace.direction_from_flat(directions[:, column])
                jvp_records.append(
                    {
                        "column": column,
                        "seconds": batch_seconds,
                        "batch_seconds": batch_seconds,
                        "direction_norm": float(direction_norm(direction, objective.dynamics).detach().cpu()),
                        "sketched_jvp_norm": float(torch.linalg.vector_norm(result.sketch_matrix[:, column]).detach().cpu()),
                        "quadratic_form": float(result.quadratic_form[column].detach().cpu()),
                        "peak_memory_mib": cuda_peak_mib(device),
                    }
                )
            del result
        elif args.jvp_mode == "batched":
            stage = "jvp_batch"
            batch_jvp_start = time.perf_counter()
            result = workspace.jvp_batch_from_flat(directions[:, :computed_columns])
            sync(device)
            residual_dim = int(result.residual_vectors.shape[0])
            if sketch_indices is None:
                if int(args.residual_sketch_dim) > 0:
                    sketch_indices, sketch_signs = make_countsketch(
                        residual_dim,
                        int(args.residual_sketch_dim),
                        device=device,
                        seed=int(args.seed) + 17 * int(sample_count),
                        dtype=result.residual_vectors.dtype,
                    )
            if sketch_indices is None:
                z_matrix_computed = result.residual_vectors.detach()
            else:
                z_matrix_computed = countsketch_apply_matrix(
                    result.residual_vectors,
                    sketch_indices,
                    sketch_signs,
                    int(args.residual_sketch_dim),
                ).detach()
            batch_seconds = time.perf_counter() - batch_jvp_start
            z_columns.extend([z_matrix_computed[:, column] for column in range(computed_columns)])
            for column in range(computed_columns):
                direction = workspace.direction_from_flat(directions[:, column])
                jvp_records.append(
                    {
                        "column": column,
                        "seconds": batch_seconds,
                        "batch_seconds": batch_seconds,
                        "direction_norm": float(direction_norm(direction, objective.dynamics).detach().cpu()),
                        "jvp_norm": float(torch.linalg.vector_norm(result.residual_vectors[:, column]).detach().cpu()),
                        "quadratic_form": float(result.quadratic_form[column].detach().cpu()),
                        "peak_memory_mib": cuda_peak_mib(device),
                    }
                )
            del result
        else:
            for column in range(computed_columns):
                stage = f"jvp_column_{column}"
                direction = workspace.direction_from_flat(directions[:, column])
                jvp_start = time.perf_counter()
                result = workspace.jvp(direction)
                sync(device)
                if residual_dim is None:
                    residual_dim = int(result.residual_vector.numel())
                    if int(args.residual_sketch_dim) > 0:
                        sketch_indices, sketch_signs = make_countsketch(
                            residual_dim,
                            int(args.residual_sketch_dim),
                            device=device,
                            seed=int(args.seed) + 17 * int(sample_count),
                            dtype=result.residual_vector.dtype,
                        )
                if sketch_indices is None:
                    z = result.residual_vector.detach()
                else:
                    z = countsketch_apply(
                        result.residual_vector,
                        sketch_indices,
                        sketch_signs,
                        int(args.residual_sketch_dim),
                    ).detach()
                z_columns.append(z)
                jvp_records.append(
                    {
                        "column": column,
                        "seconds": time.perf_counter() - jvp_start,
                        "direction_norm": float(direction_norm(direction, objective.dynamics).detach().cpu()),
                        "jvp_norm": float(torch.linalg.vector_norm(result.residual_vector).detach().cpu()),
                        "quadratic_form": float(result.quadratic_form.detach().cpu()),
                        "peak_memory_mib": cuda_peak_mib(device),
                    }
                )
                del result
        if computed_columns < int(args.sketch_rank):
            if z_columns:
                z_template = z_columns[0]
            else:
                z_template = torch.zeros(
                    int(args.residual_sketch_dim) if int(args.residual_sketch_dim) > 0 else int(residual_dim or 0),
                    device=device,
                    dtype=next(objective.dynamics.parameters()).dtype,
                )
            z_columns.extend([torch.zeros_like(z_template) for _ in range(int(args.sketch_rank) - computed_columns)])
        z_matrix = torch.stack(z_columns, dim=1)
        stage = "projected_matrix"
        projected = z_matrix.T @ z_matrix
        eigvals = torch.linalg.eigvalsh(0.5 * (projected + projected.T)).detach().cpu()
        return {
            "sample_count": int(sample_count),
            "status": "ok",
            "device": str(device),
            "cache_seconds": cache_seconds,
            "gradient_seconds": grad_seconds,
            "total_seconds": time.perf_counter() - start_total,
            "base_loss": None if value_grad is None else float(value_grad.loss.detach().cpu()),
            "base_data_loss": None if value_grad is None else float(value_grad.data_loss.detach().cpu()),
            "grad_norm": None if value_grad is None else float(torch.linalg.vector_norm(gradient).detach().cpu()),
            "direction_source": args.direction_source,
            "parameter_dim": int(workspace.parameter_dim),
            "residual_dim": int(residual_dim or 0),
            "sketch_rank": int(args.sketch_rank),
            "jvp_mode": args.jvp_mode,
            "computed_columns": int(computed_columns),
            "residual_sketch_dim": int(args.residual_sketch_dim),
            "streaming_sample_chunk_size": int(args.streaming_sample_chunk_size),
            "projected_eigenvalues": [float(v) for v in eigvals.flip(0)],
            "peak_memory_mib": cuda_peak_mib(device),
            "model": model_record,
            "jvp_records": jvp_records,
        }
    except RuntimeError as exc:
        message = str(exc)
        status = "oom" if "out of memory" in message.lower() else "runtime_error"
        clear_cuda(device)
        return {
            "sample_count": int(sample_count),
            "status": status,
            "stage": stage,
            "error": message[-3000:],
            "total_seconds": time.perf_counter() - start_total,
            "peak_memory_mib": cuda_peak_mib(device),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="OOM/memory probe for analytic reduced-GN sketch directions.")
    parser.add_argument("--train-packed", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--initializer")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-counts", default="512,1024,2048")
    parser.add_argument("--sketch-rank", type=int, default=4)
    parser.add_argument("--max-computed-columns", type=int, default=999999)
    parser.add_argument(
        "--jvp-mode",
        choices=("streaming", "batched", "streaming_batched_sketch", "single_pass_streaming_batched_sketch"),
        default="streaming",
    )
    parser.add_argument("--residual-sketch-dim", type=int, default=8192)
    parser.add_argument("--streaming-sample-chunk-size", type=int, default=64)
    parser.add_argument("--include-gradient-vector", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--direction-source", choices=("gradient_first", "random"), default="gradient_first")
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--linear-a", choices=("dense", "dissipative_skew"))
    parser.add_argument("--a-rank", type=int)
    parser.add_argument("--a-damping-init", type=float, default=0.1)
    parser.add_argument("--a-damping-shift", type=float)
    parser.add_argument("--quadratic", choices=("energy_dense", "energy_tucker"))
    parser.add_argument("--h-reduced-rank", type=int)
    parser.add_argument("--h-tt-rank", type=int)
    parser.add_argument("--decoder-quadratic-mode", choices=("full", "masked_cross", "none"))
    parser.add_argument("--decoder-cross-terms", type=int)
    parser.add_argument("--decoder-mask-seed", type=int)
    parser.add_argument("--decoder-ridge", type=float)
    parser.add_argument("--dynamics-ridge", type=float)
    parser.add_argument("--normal-chunk-size", type=int)
    parser.add_argument("--picard-iters", type=int)
    parser.add_argument("--init-scale", type=float, default=0.01)
    parser.add_argument("--time-mode", choices=("normalized", "step_index"), default="normalized")
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    checkpoint_metadata, checkpoint = config_from_checkpoint(checkpoint_path)
    payload, payload_metadata = load_packed_payload(Path(args.train_packed))
    max_count = max(parse_counts(args.sample_counts))
    payload = limit_packed_payload_samples(payload, max_count)
    metadata = {**dict(payload.get("metadata", payload_metadata)), **checkpoint_metadata}
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "reduced_gn_memory_probe.jsonl"
    with log_path.open("w", encoding="utf-8") as log:
        header = {
            "event": "metadata",
            "train_packed": str(Path(args.train_packed)),
            "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
            "sample_counts": parse_counts(args.sample_counts),
            "device": str(device),
            "torch_version": torch.__version__,
            "payload_metadata": dict(payload.get("metadata", payload_metadata)),
        }
        log.write(json.dumps(header, sort_keys=True) + "\n")
        log.flush()
        print(json.dumps(header, sort_keys=True), flush=True)
        for count in parse_counts(args.sample_counts):
            record = run_one_count(payload, metadata, checkpoint, args, count, device)
            log.write(json.dumps({"event": "probe", **record}, sort_keys=True) + "\n")
            log.flush()
            print(json.dumps({"event": "probe", **{k: record[k] for k in record if k not in {"jvp_records", "projected_eigenvalues", "error"}}}, sort_keys=True), flush=True)
            if record["status"] == "oom":
                break
    print(json.dumps({"log": str(log_path)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
