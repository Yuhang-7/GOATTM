# ADR_quadp Rerun Cases

Updated: 2026-05-03.

## Current primary rerun set

Run LinearADR data through GOATTM oldGOAM/direct-A mode.

```text
source dataset: $WORK/GOAM_clean/goam_clean/Example/LinearADR/dataset/linearADR.hdf5
raw NPZ: application/ADR_quadp/data/raw_npz/linearadr_all_samples/0.npz
processed manifest: application/ADR_quadp/data/processed_data/linearadr_all_samples_stride1/manifest.npz
source sample count: 1104
QoI shape after conversion: (time=501, qoi_dim=8) per sample
input shape after conversion: (time=501, input_dim=1) per sample
```

## Cases to rerun

```text
training sizes: 16, 32, 64, 128, 256
test size: 104
latent ranks: 12, 14, 16
```

Submit layout:

```text
ntrain = 16, 32, 64: one serial Slurm job, 1 node, 64 MPI ranks
ntrain = 128: one Slurm job, 1 node, 64 MPI ranks
ntrain = 256: one Slurm job, 2 nodes, 64 tasks per node, 128 MPI ranks
```

## Model/training settings

```text
dynamic form: AHBc
decoder form: V1V2v
oldGOAM/direct-A: enabled
latent embedding mode: qoi_augmentation
optimizer: bfgs
max iterations: 20000
time integrator: rk4
max dt: 0.01
decoder/observable regularization reg_f: 1e-7
dynamics regularization reg_g: 1e-9
spectral/eigenvalue regularization: 0
```

Notes:

- `qoi_augmentation` is required because ADR QoI dimension is 8 while the target latent ranks are 12, 14, and 16.
- We keep RK4; no need to switch integrators for this rerun.
- This rerun is meant to test whether the previous ADR results were hurt by overly large regularization.
- The most important paper-facing ranks are `r = 14` and `r = 16`; `r = 12` is the lower-rank reference.

## Prepared scripts

```bash
cd /work2/08667/yuuuhang/stampede3/GOATTM
sbatch application/ADR_quadp/submit_linearadr_oldgoam_n16_32_64.slurm
sbatch application/ADR_quadp/submit_linearadr_oldgoam_n128.slurm
sbatch application/ADR_quadp/submit_linearadr_oldgoam_n256.slurm
```

Shared worker script:

```bash
application/ADR_quadp/submit_linearadr_oldgoam.sh
```

## Current smoke test

A small smoke test has been submitted with:

```text
job id: 3079498
ntrain: 16
ntest: 104
latent rank: 12
max iterations: 2
time integrator: rk4
latent embedding mode: qoi_augmentation
```

As of the last check it was still pending with reason `Priority`.
