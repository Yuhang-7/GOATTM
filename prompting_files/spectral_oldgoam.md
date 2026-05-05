### Mode GOAM implementation

Here, we write a prompt for a special mode in goattm, named "oldGOAM mode", which is designed to replicate the original GOAM implementation as closely as possible. This mode will be used for comparison purposes, to understand the differences between the new GOAM implementation and the original one.

### Calling of this Mode

Add a flag `--oldgoam` to the command line when running goattm to activate this mode. 


### What this Mode does

This mode changes the way of penalizing the linear part of the latent dynamics. 

In the original node, we use the parametrization $A = -SS^T +W$, where S is a lower triangular matrix and W is a skew-symmetric matrix. We penalize on the Frobenius norm of S and W.

Now, we parametrize A directly by the elements of A. We add two types of penalization on A:
    - Frobenius norm of A, which is the same as the original one.
    - Spectral penalization, basically \lambda(A+A^T) / 2, the largest eigenvalue of the symmetric part of A. To define this spectral penalization, you need to define the derivative of the largest eigenvalue with respect to A. Note that A is a very small matrix (e.g. 20*20), so the computation of the largest eigenvalue and its derivative is not expensive.

### Other pipeline should be done

- Finish the implementation of this mode. By finish it, write the corresponding unit test and run it with the demo. Run the demo with oldGOAM mode, with 100 BFGS iterations.
- Rerun the training of GOAM after the implementation with this mode. To do this, you need to do specified data preprocess for the different dataset. We are interested in the training for ADR_quadp and Navier Stokes. Please create subdir ADR_quadp and Navier_Stokes under GOATTM/application, and preprocessed the data accordingly.
- The original datafiles can be found in the original GOAM implementation in h5 format. You need to change it into csv format. Also, you need to make sure that we can select the number of training dataset and testing dataset.


For the specific training hyperparameters, you will be told later.


### Stampede3 preprocess implementation status

Implementation target: `$WORK/GOATTM` on Stampede3. This is intentionally separate from this prompt file, which currently lives on ccgo1 at:

```bash
/storage/yuhang/Myresearch/GOATTM/prompting_files/spectral_oldgoam.md
```

The original GOAM data used here is already available as chunked `.npz` files under:

```bash
$WORK/GOAM_clean/goam_clean
```

The new GOATTM converter reads GOAM chunk files with:

```text
QoI_list[dq, nt, nsample]
bc_datas_list[dp, nt, nsample]
Tlist[nt]
```

and writes GOATTM-compatible per-sample files plus:

```text
manifest.npz
train_manifest.npz
test_manifest.npz
summary.json
```

This makes the train/test dataset size selectable through command-line arguments rather than hard-coded in the data interface.

### Files added on Stampede3

```bash
$WORK/GOATTM/src/goattm/preprocess/goam_chunk_npz.py
$WORK/GOATTM/application/ADR_quadp/codes/preprocess_adr_quadp_npz_dataset.py
$WORK/GOATTM/application/ADR_quadp/codes/run_preprocess_adr_quadp.sh
$WORK/GOATTM/application/ADR_quadp/README.md
$WORK/GOATTM/application/ADR_quadp/data/README.md
$WORK/GOATTM/application/Navier_Stokes/codes/preprocess_navier_stokes_npz_dataset.py
$WORK/GOATTM/application/Navier_Stokes/codes/run_preprocess_navier_stokes.sh
$WORK/GOATTM/application/Navier_Stokes/README.md
$WORK/GOATTM/application/Navier_Stokes/data/README.md
```

### ADR_quadp preprocess command

```bash
cd $WORK/GOATTM
./application/ADR_quadp/codes/run_preprocess_adr_quadp.sh \
  --train-count 112 \
  --test-count 112 \
  --qoi-stride 1 \
  --output-root application/ADR_quadp/data/processed_data/train112_test112
```

Use `--train-root` and `--test-root` to select a different ADR_quadp rerun directory.

### Navier-Stokes preprocess command

```bash
cd $WORK/GOATTM
./application/Navier_Stokes/codes/run_preprocess_navier_stokes.sh \
  --train-count 896 \
  --test-count 104 \
  --qoi-stride 1 \
  --output-root application/Navier_Stokes/data/processed_data/train896_test104
```

The tiny smoke tests completed on Stampede3 with `--train-count 3 --test-count 2` for both ADR_quadp and Navier-Stokes. Full training should still be submitted as jobs, not run on a login node.


### Preprocess submit scripts

The preprocess jobs are prepared on Stampede3 and can be checked/submitted from `$WORK/GOATTM`.

ADR_quadp:

```bash
cd $WORK/GOATTM
sbatch application/ADR_quadp/submit_preprocess_adr_quadp.slurm
```

Navier-Stokes:

```bash
cd $WORK/GOATTM
sbatch application/Navier_Stokes/submit_preprocess_navier_stokes.slurm
```

Useful overrides:

```bash
TRAIN_COUNT=64 TEST_COUNT=112 QOI_STRIDE=1 \
OUTPUT_ROOT=$WORK/GOATTM/application/ADR_quadp/data/processed_data/train64_test112 \
sbatch application/ADR_quadp/submit_preprocess_adr_quadp.slurm

TRAIN_COUNT=448 TEST_COUNT=104 QOI_STRIDE=1 \
OUTPUT_ROOT=$WORK/GOATTM/application/Navier_Stokes/data/processed_data/train448_test104 \
sbatch application/Navier_Stokes/submit_preprocess_navier_stokes.slurm
```

Both submit scripts use the GOATTM311 Python environment and the chunked GOAM `.npz` data. They do not rely on `h5py`.


### Corrected HDF5 to NPZ pipeline

The intended data path is two-stage:

```text
original GOAM .hdf5
  -> raw GOAM-style .npz chunk directory
  -> GOATTM NpzQoiSample manifest directory
```

This matters because the GOATTM runtime environment does not provide `h5py`. The first conversion script therefore loads the Stampede3 HDF5 module and uses `h5dump` as a fallback when `h5py` is unavailable. The second stage only reads `.npz` files.

ADR HDF5 to raw NPZ:

```bash
cd $WORK/GOATTM
sbatch application/ADR_quadp/submit_convert_hdf5_to_npz.slurm
```

Default ADR source and output:

```bash
$WORK/GOAM_clean/goam_clean/Example/ADR_LinearQoI/dataset/ADR_LinearQoI.hdf5
$WORK/GOATTM/application/ADR_quadp/data/raw_npz/train112_test20
```

ADR raw NPZ to GOATTM manifest:

```bash
sbatch application/ADR_quadp/submit_preprocess_adr_quadp.slurm
```

Navier-Stokes HDF5 to raw NPZ:

```bash
cd $WORK/GOATTM
sbatch application/Navier_Stokes/submit_convert_hdf5_to_npz.slurm
```

Default NS source and output:

```bash
$WORK/GOAM_clean/goam_clean/Example/Navier_Stokes_Re=100/dataset/latent_data_1000_Re=100.hdf5
$WORK/GOATTM/application/Navier_Stokes/data/raw_npz/train896_test104
```

NS raw NPZ to GOATTM manifest:

```bash
sbatch application/Navier_Stokes/submit_preprocess_navier_stokes.slurm
```

Dataset size can be changed at either stage with environment variables, for example:

```bash
TRAIN_COUNT=64 TEST_COUNT=20 \
OUTPUT_ROOT=$WORK/GOATTM/application/ADR_quadp/data/raw_npz/train64_test20 \
sbatch application/ADR_quadp/submit_convert_hdf5_to_npz.slurm

TRAIN_COUNT=64 TEST_COUNT=20 \
TRAIN_ROOT=$WORK/GOATTM/application/ADR_quadp/data/raw_npz/train64_test20 \
TEST_ROOT=$WORK/GOATTM/application/ADR_quadp/data/raw_npz/train64_test20 \
OUTPUT_ROOT=$WORK/GOATTM/application/ADR_quadp/data/processed_data/train64_test20 \
sbatch application/ADR_quadp/submit_preprocess_adr_quadp.slurm
```

Smoke validation completed for both ADR and Navier-Stokes with `3 train + 2 test` through the full chain `HDF5 -> raw_npz -> manifest`.
