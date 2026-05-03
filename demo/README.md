# GOATTM Configurable Reduced-QoI Demo

This demo runs the maintained synthetic GOATTM problem with configurable
training and test sizes.

The problem setting matches the small 10/10 workflow:

- Synthetic scalar input `p(t)` is sampled on `t=0:0.1:1`.
- QoI observations use `q_j(t)=exp(a_j p(t)) + b_j (p(t)-a_j)^2`.
- Raw data is normalized before OpInf initialization.
- OpInf uses midpoint finite-difference regression for `A,H,B,c`.
- `S` is initialized as identity in the stabilized dynamics.
- OpInf forward validation and training use `implicit_midpoint` by default.
- Training uses L-BFGS and writes train/test loss plus relative error.

This wrapper is geared toward compute-node execution:

- It activates conda env `fenicsx-clean` by default.
- It writes outputs under `demo/outputs/reduced_qoi_optimization_demo` by default.

Run the default 10 train / 10 test case:

```bash
bash demo/run_reduced_qoi_demo.sh
```

Run a larger case:

```bash
bash demo/run_reduced_qoi_demo.sh 100 100 --max-iterations 50
```

Change the latent rank from the shell caller:

```bash
LATENT_RANK=10 bash demo/run_reduced_qoi_demo.sh 10 10
```

By default the wrapper uses `mpirun -n NTRAIN`, so each train case gets one MPI
rank when no allocation is detected. If the train set is larger than the
available MPI slots, override the rank count:

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
- `--oldgoam` to use direct-A oldGOAM-style dynamics instead of stabilized `A=-SS^T+W`
- `--max-dt 0.01`
- `--time-integrator implicit_midpoint`
- `--normalization-target-max-abs 0.9`

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
- `--dynamics-reg-a 1e-4`
- `--dynamics-reg-s 1e-4`
- `--dynamics-reg-w 1e-4`
- `--dynamics-reg-mu-h 1e-4`
- `--dynamics-reg-b 1e-4`
- `--dynamics-reg-c 1e-4`
- `--dynamics-reg-spectral-abscissa 0.0`, deactivated by default
- `--dynamics-reg-spectral-alpha 0.0`

- Output:
- `--output-dir /path/to/output`

Useful shell variables:

- `MPI_RANKS`, default to the launcher allocation when available, otherwise `NTRAIN`
- `CONDA_ENV_NAME`, default `fenicsx-clean`
- `GOATTM_SKIP_CONDA_ACTIVATE=1` to skip activation when the env is already active
- `LATENT_RANK`, default `10`
- `OPTIMIZER`, default `lbfgs`
- `MAX_ITERATIONS`, default `500`
- `MAX_DT`, default `0.01`
- `TIME_INTEGRATOR`, default `implicit_midpoint`
- `NORMALIZATION_TARGET_MAX_ABS`, default `0.9`
- `SEED`, default `20260428`
- `LBFGS_MAXCOR`, `LBFGS_FTOL`, `LBFGS_GTOL`, `LBFGS_MAXLS`
- `OPINF_REG_W`, `OPINF_REG_H`, `OPINF_REG_B`, `OPINF_REG_C`
- `DECODER_REG_V1`, `DECODER_REG_V2`, `DECODER_REG_V0`
- `DYNAMICS_REG_A`, `DYNAMICS_REG_S`, `DYNAMICS_REG_W`, `DYNAMICS_REG_MU_H`, `DYNAMICS_REG_B`, `DYNAMICS_REG_C`
- `DYNAMICS_REG_SPECTRAL_ABSCISSA`, default `0.0`
- `DYNAMICS_REG_SPECTRAL_ALPHA`, default `0.0`
- `OLDGOAM=1` to pass `--oldgoam`
- `OUTPUT_DIR`, default `demo/outputs/reduced_qoi_optimization_demo`

Outputs are written under the selected `OUTPUT_DIR`. The latest top-level
summary is `latest_summary.json`; per-run logs, loss histories
(`loss_history.csv` and `loss_history.md`), metrics, checkpoints, timing
summaries, OpInf logs, and the human-readable `optimization_report.md` are
linked from that summary.
