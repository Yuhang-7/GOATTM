# GOATTM ADR_quadp preprocess

This application converts GOAM Stampede3 chunked `.npz` files into the GOATTM `NpzQoiSample` manifest format.

Default source roots:

```bash
$WORK/GOAM_clean/goam_clean/Example/ADR_LinearQoI/tmpdirectory/ADR_LinearQoI_quadp_opinf_trainsize_112_threads_112
$WORK/GOAM_clean/goam_clean/Example/ADR_LinearQoI/tmpdirectory/ADR_LinearQoI_quadp_validate_trainsize_112_threads_112
```

Each GOAM file is expected to contain `QoI_list[dq, nt, nsample]`, `bc_datas_list[dp, nt, nsample]`, and `Tlist[nt]`. The converter writes one GOATTM sample per source trajectory and creates `manifest.npz`, `train_manifest.npz`, `test_manifest.npz`, and `summary.json`.

Basic usage:

```bash
cd $WORK/GOATTM
./application/ADR_quadp/codes/run_preprocess_adr_quadp.sh \
  --train-count 112 \
  --test-count 112 \
  --qoi-stride 1 \
  --output-root application/ADR_quadp/data/processed_data/train112_test112
```

To change dataset size, change `--train-count` and `--test-count`. To point at a different rerun directory, pass `--train-root` and `--test-root` explicitly.

For a quick login-node smoke test, use tiny counts and a large stride:

```bash
./application/ADR_quadp/codes/run_preprocess_adr_quadp.sh \
  --train-count 3 \
  --test-count 2 \
  --qoi-stride 40 \
  --output-root application/ADR_quadp/data/smoke_processed_data
```
