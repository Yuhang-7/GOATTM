# ADR_quadp Problem Specification

This document records the ADR dataset and reduced-order-modeling convention used for the GOATTM ADR application. `LinearADR` is not part of this comparison path and should not be used for the RBniCS/DEIM-facing ADR experiments.

## Data Source

The canonical GOAM data source for the current GOATTM ADR reruns is:

```text
/storage/yuhang/Myresearch/GOAM_clean/Example/ADR/dataset/ADR_quadp/ADR_quadp_trainsize=256_testsize=200.hdf5
```

The file contains 456 samples: the first 256 are training samples and the final 200 are testing samples. The source arrays are stored as:

```text
QoI_list       [11, 1001, 456]
bc_datas_list  [2, 1001, 456]
Tlist          [1001]
```

The physical time grid is `0, 0.002, ..., 2.0`. GOATTM preprocessing keeps time indices `0,2,4,...,1000`, giving 501 time samples on `0, 0.004, ..., 2.0` unless a different `--qoi-stride` is requested.

Other GOAM `ADR_quadp` files exist for dataset sensitivity studies with train sizes `0, 16, 32, 64, 128, 256, 512`, always with `testsize=200`. The `256/200` split is the default for this application.

## Mathematical Problem

The underlying full-order problem is a one-dimensional nonlinear advection-diffusion-reaction equation on `x in [0,1]`, `t in [0,2]`:

```math
u_t + b u_x - D u_{xx} + K u^3 = 0.
```

The coefficients are:

```text
b = 1.0
D = 0.01
K = 0.01
```

The initial condition is zero. The left boundary is driven by a time-dependent Dirichlet signal:

```math
u(0,t) = g(t;\mu),
```

RBniCS marks only this left boundary for the non-homogeneous Dirichlet condition. There is no second Dirichlet input at the right boundary in the RBniCS workflow; the right side is left as the natural/outflow boundary implied by the weak form.

where

```math
g(t;\mu)=\sum_{i=1}^{40}
\frac{a_i}{\sigma_i\sqrt{2\pi}}
\exp\left(-\frac{(t-m_i)^2}{2\sigma_i^2}\right).
```

The parameter vector has 120 entries:

```text
mu = (m_1, sigma_1, a_1, ..., m_40, sigma_40, a_40)
```

with historical GOAM sampling bounds:

```text
m_i     in [0.0, 2.0]
sigma_i in [0.01, 0.04]
a_i     in [-0.3, 0.3]
```

## Inputs And Outputs

`ADR_quadp` exposes two input channels:

```text
bc_datas_list[0] = g(t)
bc_datas_list[1] = g(t)^2
```

The QoI vector has 11 channels. The first five are point observations of the state, the next five are point observations of the spatial derivative, and the last channel is the energy:

```math
q_i(t)=u(x_i,t),\quad x_i=i/6,\quad i=1,\dots,5,
```

```math
q_{5+i}(t)=u_x(x_i,t),\quad i=1,\dots,5,
```

```math
q_{11}(t)=\int_0^1 u(x,t)^2\,dx.
```

GOATTM stores these labels as:

```text
density_1, density_2, density_3, density_4, density_5,
flux_1, flux_2, flux_3, flux_4, flux_5,
energy
```

## Reduced-Basis / RBniCS-DEIM Reference

The RBniCS reference builds a POD-Galerkin reduced basis for the nonlinear parabolic ADR problem. In the DEIM workflow, the cubic reaction contribution `(u^3, v)` is approximated online by DEIM, while the reduced trajectory is still obtained by solving the reduced nonlinear time-dependent system.

For GOATTM comparisons, the target is the input-to-output map:

```math
[g(t), g(t)^2]_{t=0}^{T} \longmapsto [q_1(t),\dots,q_{11}(t)]_{t=0}^{T}.
```

Therefore GOATTM runs must use the same QoI definition and compatible time grid as the RBniCS/DEIM comparison. Runs based on `LinearADR.hdf5` have only 8 QoI channels, one input channel, and 501 source time points; they are not comparable and should be treated as deprecated for this experiment.

## GOATTM Preprocess Contract

The GOATTM preprocessor is:

```text
application/ADR/codes/preprocess_adr_hdf5_dataset.py
```

The default conversion source is the `ADR_quadp_trainsize=256_testsize=200.hdf5` file. The default output is:

```text
application/ADR/data/processed_data/manifest.npz
application/ADR/data/processed_data/samples/*.npz
```

Each sample stores:

```text
observation_times [501]
qoi_observations  [501, 11]
input_times       [501]
input_values      [501, 2]
u0                [11]
```

By default, the preprocessor does not divide by `Ps_normalize_constant` or `Qs_normalize_constant`; this preserves the GOAM ADR data scale. Normalization can be enabled explicitly with `--normalize-input-from-file` or `--normalize-qoi-from-file` for controlled experiments.

