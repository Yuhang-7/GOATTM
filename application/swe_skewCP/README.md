# SWE skewCP application

This application is for running SWE reduced-QoI training with `SkewCPQuadraticDynamics` on the same processed SWE manifest used by `application/swe_problem`.

The first script is a login-node smoke test. It intentionally uses a tiny train/test split, a strided QoI time grid, and one optimizer iteration, but defaults to `latent_rank=32` so that the high-rank latent embedding path is exercised for SWE QoI dimension 30.

Initialization follows the simple library logic: run ordinary OpInf in `ABc` form to initialize only `A,B,c`, then create a `SkewCPQuadraticDynamics` model with the skewCP quadratic term initialized from zero factors by default.

Default data source:

```bash
/work2/08667/yuuuhang/stampede3/GOATTM/application/swe_problem/data/processed_data_kl_m200/manifest.npz
```

Run from the repository root:

```bash
bash application/swe_skewCP/codes/run_swe_skewcp_login_smoke.sh
```

Useful overrides:

```bash
LATENT_RANK=40 SKEW_CP_RANK=10 MAX_ITERATIONS=2 \
  bash application/swe_skewCP/codes/run_swe_skewcp_login_smoke.sh
```

This is not an sbatch script. It is only meant to verify that the tuckerTT branch, SWE manifest, OpInf preprocessing, skewCP initialization, and trainer can run together on a login node before preparing production Slurm jobs.
