# GOATTM ADR Application

See `docs/adr_quadp_problem.md` for the canonical ADR_quadp PDE, QoI, input-channel, and reduced-basis comparison specification. `LinearADR` is deprecated for this RBniCS/DEIM-facing ADR workflow.

This directory is local-only and ignored by git via `/storage/yuhang/Myresearch/GOATTM/.gitignore`.

## Data

The default GOAM source file is the ADR_quadp 256/200 split:

`/storage/yuhang/Myresearch/GOAM_clean/Example/ADR/dataset/ADR_quadp/ADR_quadp_trainsize=256_testsize=200.hdf5`

The GOATTM processed dataset is:

`application/ADR/data/processed_data/manifest.npz`

It contains 456 samples: the first 256 are train samples and the last 200 are test samples, matching the GOAM HDF5 order. Each sample has 501 observation times on `0, 0.004, ..., 2.0`, 11 QoI observations, and 2 boundary input channels `[g(t), g(t)^2]`. This corresponds to GOAM HDF5 time indices `0,2,4,...,1000`.

## Preprocess

```bash
application/ADR/codes/run_preprocess_adr.sh
```

Optional smoke conversion:

```bash
application/ADR/codes/run_preprocess_adr.sh --limit 6 --output-root application/ADR/data/processed_data_smoke
```

## Local Smoke Test Only

A tiny run is acceptable for code integrity checks:

```bash
python application/ADR/codes/run_adr_rank_demo.py \
  --manifest-path application/ADR/data/processed_data_smoke/manifest.npz \
  --sample-count 6 --ntrain 4 --ntest 2 \
  --latent-rank 3 --max-iterations 1 \
  --output-dir application/ADR/outputs/smoke_train
```

Do not run the full experiment on a login node.

## Application Run Wrapper

The main run wrapper mirrors `application/swe_problem/codes/run_swe_rank8_demo.sh`:

```bash
bash application/ADR/codes/run_adr_rank10_demo.sh --help
```

Do not run the full experiment on a login node. For a local code-path check, use the smoke-test command above.

## Submit Full Rank-10 Job

From a Slurm login node, submit the thin Slurm wrapper, which calls `run_adr_rank10_demo.sh` with `MPIRUN_BIN=ibrun` and `MPI_RANKS=${SLURM_NTASKS}`:

```bash
sbatch application/ADR/codes/submit_adr_rank10_spr.slurm
```

Useful overrides:

```bash
LATENT_RANK=10 MAX_ITERATIONS=100 OPTIMIZER=lbfgs sbatch application/ADR/codes/submit_adr_rank10_spr.slurm
LATENT_RANK=16 LATENT_EMBEDDING_MODE=qoi_augmentation sbatch application/ADR/codes/submit_adr_rank10_spr.slurm
```

For the default POD latent embedding, `LATENT_RANK` must be `<= 11` because ADR has 11 QoI channels. Use `LATENT_EMBEDDING_MODE=qoi_augmentation` only when deliberately testing ranks larger than the QoI dimension.
