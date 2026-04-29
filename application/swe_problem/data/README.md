# SWE Data Workspace

This folder is the local data workspace for `application/swe_problem`.

Set the original SWE dataset location through an environment variable:

```bash
export SWE_ORIGINAL_DATA_ROOT=/storage/yuhang/swedata/originaldata/swe_data_2026_510510
```

The preprocessing script reads samples from:

```text
$SWE_ORIGINAL_DATA_ROOT/sample_*/sample_*.npz
```

and writes GOATTM-compatible data to:

```text
application/swe_problem/data/processed_data/
```

By default it keeps SWE QoI snapshots at physical times `5, 10, ..., 1500`
and prepends a zero QoI observation at `t=0`. Each processed sample therefore
has 301 observation times:

```text
0, 5, 10, ..., 1500
```

When the demo pipeline later uses `time_rescale_to_unit_interval=True`, this
becomes `0, 1/300, 2/300, ..., 1`, so `max_dt=1/600` gives two integration
steps per QoI interval.

Smoke run:

```bash
cd /storage/yuhang/Myresearch/GOATTM
python application/swe_problem/codes/preprocess_swe_npz_dataset.py --limit 40 --output-root application/swe_problem/data/processed_data_smoke
```

Full run:

```bash
cd /storage/yuhang/Myresearch/GOATTM
python application/swe_problem/codes/preprocess_swe_npz_dataset.py
```

Run the 64-sample rank-8 RK4 optimization demo:

```bash
cd /storage/yuhang/Myresearch/GOATTM
bash application/swe_problem/codes/run_swe_rank8_demo.sh
```

For a very short trial:

```bash
MAX_ITERATIONS=2 bash application/swe_problem/codes/run_swe_rank8_demo.sh
```
