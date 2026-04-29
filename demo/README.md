# GOATTM Configurable Reduced-QoI Demo

This demo runs the maintained synthetic GOATTM problem with configurable
training and test sizes.

The problem setting matches the small 10/10 workflow:

- Synthetic scalar input `p(t)` is sampled on `t=0:0.1:1`.
- QoI observations use `q_j(t)=exp(a_j p(t)) + b_j (p(t)-a_j)^2`.
- Raw data is normalized before OpInf initialization.
- OpInf uses midpoint finite-difference regression for `A,H,B,c`.
- `S` is initialized as identity in the stabilized dynamics.
- OpInf forward validation and training both use `rk4`.
- Training uses L-BFGS and writes train/test loss plus relative error.

Run the default 10 train / 10 test case:

```bash
bash demo/run_reduced_qoi_demo.sh
```

This wrapper is now geared toward compute-node execution:

- it activates conda env `fenicsx-clean` by default;
- it writes outputs under `demo/outputs/reduced_qoi_optimization_demo` by default.

Run a larger case:

```bash
bash demo/run_reduced_qoi_demo.sh 100 100 --max-iterations 50
```

Change the latent rank from the shell caller:

```bash
LATENT_RANK=10 bash demo/run_reduced_qoi_demo.sh 10 10
```

By default the wrapper uses `mpirun -n NTRAIN`, so each train case gets one MPI
rank. If the train set is larger than the available MPI slots, override the
rank count:

```bash
MPI_RANKS=20 bash demo/run_reduced_qoi_demo.sh 100 100 --max-iterations 50
```

Useful Python options:

- Problem setting:
- `--ntrain 10`
- `--ntest 10`
- `--observation-dt 0.1`
- `--output-dimension 20`
- `--seed 20260428`

- Model and solver setting:
- `--latent-rank 4`
- `--max-dt 0.01`

- Optimization setting:
- `--optimizer lbfgs`
- `--max-iterations 50`
- `--lbfgs-maxcor 20`
- `--lbfgs-ftol 1e-12`
- `--lbfgs-gtol 1e-8`
- `--lbfgs-maxls 30`

- Regularization setting:
- `--opinf-reg-w 1e-4`
- `--opinf-reg-h 1e-4`
- `--opinf-reg-b 1e-4`
- `--opinf-reg-c 1e-6`
- `--decoder-reg-v1 1e-7`
- `--decoder-reg-v2 1e-7`
- `--decoder-reg-v0 1e-7`
- `--dynamics-reg-s 1e-4`
- `--dynamics-reg-w 1e-4`
- `--dynamics-reg-mu-h 1e-4`
- `--dynamics-reg-b 1e-4`
- `--dynamics-reg-c 1e-4`

- Output:
- `--output-dir /path/to/output`

Useful shell variables:

- `MPI_RANKS`, default `NTRAIN`
- `CONDA_ENV_NAME`, default `fenicsx-clean`
- `GOATTM_SKIP_CONDA_ACTIVATE=1` to skip activation when the env is already active
- `LATENT_RANK`, default `4`
- `OPTIMIZER`, default `lbfgs`
- `MAX_ITERATIONS`, default `50`
- `MAX_DT`, default `0.01`
- `SEED`, default `20260428`
- `LBFGS_MAXCOR`, `LBFGS_FTOL`, `LBFGS_GTOL`, `LBFGS_MAXLS`
- `OPINF_REG_W`, `OPINF_REG_H`, `OPINF_REG_B`, `OPINF_REG_C`
- `DECODER_REG_V1`, `DECODER_REG_V2`, `DECODER_REG_V0`
- `DYNAMICS_REG_S`, `DYNAMICS_REG_W`, `DYNAMICS_REG_MU_H`, `DYNAMICS_REG_B`, `DYNAMICS_REG_C`
- `OUTPUT_DIR`, default `demo/outputs/reduced_qoi_optimization_demo`

Outputs are written under the selected `OUTPUT_DIR`. The latest top-level
summary is `latest_summary.json`; per-run logs, metrics, checkpoints, and
OpInf logs are linked from that summary.
