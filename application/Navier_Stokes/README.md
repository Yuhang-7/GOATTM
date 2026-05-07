# GOATTM Navier-Stokes preprocess

This application converts GOAM Stampede3 chunked `.npz` files into the GOATTM `NpzQoiSample` manifest format.

Default source root:

```bash
$WORK/GOAM_clean/goam_clean/Example/Navier_Stokes_Re=100/tmpdirectory/Navier_Stokes_Re_100_trainsize_896_threads_112
```

The same directory contains numeric training chunks and `test.npz`; the converter uses both by default. Each GOAM file is expected to contain `QoI_list[dq, nt, nsample]`, `bc_datas_list[dp, nt, nsample]`, and `Tlist[nt]`.

Basic usage:

```bash
cd $WORK/GOATTM
./application/Navier_Stokes/codes/run_preprocess_navier_stokes.sh \
  --train-count 896 \
  --test-count 104 \
  --qoi-stride 1 \
  --output-root application/Navier_Stokes/data/processed_data/train896_test104
```

To change dataset size, change `--train-count` and `--test-count`. To use a different Re or rerun directory, pass `--train-root` explicitly. If the test data is in a separate directory, also pass `--test-root`; otherwise `test.npz` under `--train-root` is used automatically.

For a quick login-node smoke test, use tiny counts and a larger stride:

```bash
./application/Navier_Stokes/codes/run_preprocess_navier_stokes.sh \
  --train-count 3 \
  --test-count 2 \
  --qoi-stride 20 \
  --output-root application/Navier_Stokes/data/smoke_processed_data
```
