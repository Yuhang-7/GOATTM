# GOATTM Application: Navier-Stokes Validation Backfill

Submit scripts live under `submit/`.

The current backfill job does **not** retrain. It calls the existing GOAM Navier-Stokes validation workflow to compute missing `validation_convergence_r15.csv` files for repaired/rerun badpoint directories.

Cases are listed in `manifests/ns_badpoint_validation_cases.csv`.
