# Navier-Stokes Rerun Cases

## Clean oldGOAM baseline

Run oldGOAM/direct-A GOATTM with:

```text
latent ranks: 12, 14, 16
dynamic form: AHBc
decoder form: V1V2v
optimizer: bfgs
max iterations: 20000
decoder/observable regularization reg_f: 1e-7
dynamics regularization reg_g: 1e-9
spectral/eigenvalue regularization: 0
```

Default data size:

```text
train: 896
test: 104
```

Source HDF5:

```bash
$WORK/GOAM_clean/goam_clean/Example/Navier_Stokes_Re=100/dataset/latent_data_1000_Re=100.hdf5
```

## Why rerun

- This gives a clean GOATTM oldGOAM/direct-A baseline in the same library used for the ADR rerun.
- The previous NS discussion included missing or questionable points, especially around `r = 16` and dataset-size/eigenvalue-penalty sweeps.
- We should first establish the no-spectral-penalty baseline before deciding whether to rerun the eigenvalue-penalty curves.

## Follow-up candidates after the baseline

If the clean baseline looks reasonable, rerun the NS eigenvalue-penalty sensitivity grid:

```text
dataset sizes: 112, 224, 448, 896
primary rank: 16
spectral/eigenvalue penalty: sweep values to be chosen before submission
```

Reason: the previous NS plot/data sweep had missing `r = 16` data and possible non-monotone/outlier behavior.

## Submission order

```bash
cd $WORK/GOATTM
sbatch application/Navier_Stokes/submit_convert_hdf5_to_npz.slurm
sbatch application/Navier_Stokes/submit_preprocess_navier_stokes.slurm
sbatch application/Navier_Stokes/submit_rerun_oldgoam.slurm
```
