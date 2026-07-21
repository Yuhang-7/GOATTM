from pathlib import Path

p = Path("/global/homes/y/yuuuhang/quad_goattm/tools/train_cascadia_packed.py")
s = p.read_text()

insert = '''def project_dense_a_symmetric_positive_part(
    dynamics: QuadraticDynamics,
    *,
    tolerance: float = 0.0,
) -> dict[str, float]:
    if not isinstance(dynamics.linear, DenseLinearA):
        return {"applied": False}
    with torch.no_grad():
        a_matrix = dynamics.linear.A
        symmetric = 0.5 * (a_matrix + a_matrix.T)
        skew = 0.5 * (a_matrix - a_matrix.T)
        eigvals, eigvecs = torch.linalg.eigh(symmetric)
        clipped = torch.clamp(eigvals, max=float(tolerance))
        projected_symmetric = (eigvecs * clipped.unsqueeze(0)) @ eigvecs.T
        a_matrix.copy_(skew + projected_symmetric)
        positive = torch.clamp(eigvals - float(tolerance), min=0.0)
        return {
            "applied": True,
            "tolerance": float(tolerance),
            "lambda_max_before": float(eigvals[-1].detach().cpu()),
            "lambda_max_after": float(clipped[-1].detach().cpu()),
            "positive_fro_norm_removed": float(torch.linalg.norm(positive).detach().cpu()),
            "positive_rank_removed": int((positive > 0).sum().detach().cpu()),
        }


'''

anchor = '''def dense_a_symmetric_spectral_terms(
    dynamics: QuadraticDynamics,
    temperature: float,
) -> dict[str, torch.Tensor]:
'''
if "def project_dense_a_symmetric_positive_part(" not in s:
    if anchor not in s:
        raise SystemExit("anchor for projection function not found")
    s = s.replace(anchor, insert + anchor)

old_arg = '    parser.add_argument("--dense-a-symmetric-temperature", type=float, default=1.0e-2)\n'
new_arg = '''    parser.add_argument("--dense-a-symmetric-temperature", type=float, default=1.0e-2)
    parser.add_argument("--project-initial-dense-a-symmetric-positive", action="store_true")
    parser.add_argument("--project-initial-dense-a-symmetric-tolerance", type=float, default=0.0)
'''
if "--project-initial-dense-a-symmetric-positive" not in s:
    if old_arg not in s:
        raise SystemExit("anchor for projection args not found")
    s = s.replace(old_arg, new_arg)

old_init = '''    init_report = apply_pod_initializer(dynamics, decoder, Path(args.initializer))
    if args.load_checkpoint:
        checkpoint_path = Path(args.load_checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        dynamics.load_state_dict(checkpoint["dynamics_state_dict"])
        decoder.load_state_dict(checkpoint["decoder_state_dict"])
        init_report["loaded_checkpoint"] = str(checkpoint_path)
'''
new_init = '''    init_report = apply_pod_initializer(dynamics, decoder, Path(args.initializer))
    if args.load_checkpoint:
        checkpoint_path = Path(args.load_checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        dynamics.load_state_dict(checkpoint["dynamics_state_dict"])
        decoder.load_state_dict(checkpoint["decoder_state_dict"])
        init_report["loaded_checkpoint"] = str(checkpoint_path)
    if args.project_initial_dense_a_symmetric_positive:
        init_report["dense_a_symmetric_projection"] = project_dense_a_symmetric_positive_part(
            dynamics,
            tolerance=float(args.project_initial_dense_a_symmetric_tolerance),
        )
'''
if 'init_report["dense_a_symmetric_projection"]' not in s:
    if old_init not in s:
        raise SystemExit("anchor for projection call not found")
    s = s.replace(old_init, new_init)

p.write_text(s)
