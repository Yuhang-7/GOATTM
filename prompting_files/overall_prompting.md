# GOATTM Prompting Overview

Use this repository as a library-oriented codebase, not just as a temporary experiment branch.

At a high level, `GOATTM` provides:

- quadratic latent dynamics models,
- quadratic decoders for QoI prediction,
- `.npz` dataset and manifest utilities,
- centered max-absolute preprocessing normalization and `OpInf` initialization,
- rollout/loss/adjoint logic for `implicit_midpoint`, `explicit_euler`, and `rk4`,
- homotopy fallback for difficult implicit-midpoint nonlinear solves,
- reduced best-response training workflows,
- root-led L-BFGS with MPI-aware sample-parallel dataset evaluation,
- collective-safe forward rollout failure reporting and run logging.

When describing the library in prompts, prefer wording that matches the current implementation:

`GOATTM is a reduced-order modeling and reduced-QoI training library for quadratic latent dynamics,
with structured parameterizations, multiple time integrators, exact discrete derivatives,
decoder best-response optimization, bounded preprocessing normalization, root-led distributed
optimization, and homotopy-guarded implicit nonlinear solves.`

Useful companion documents:

- `prompting_files/description.md`: implementation-aligned library description
- `prompting_files/nonlinear_solve_revisit.md`: nonlinear-solve stability notes and current homotopy status
- `documents/high_level_flyer.md`: short, user-facing overview
- `documents/detailed_documentation.md`: more detailed architecture and workflow guide
