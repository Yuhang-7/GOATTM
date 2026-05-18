# SWE KL Input Pipeline Correction

## Critical bug report

The previous GOATTM SWE runs accidentally used the legacy processed input

```text
p(t) = [xi, yi, Ti, sigma_i, Hi] in R^25
```

as the latent-ODE forcing.  This is not the intended scientific problem.
The wrong dataset is:

```text
/work2/08667/yuuuhang/stampede3/GOATTM/application/swe_problem/data/processed_data/manifest.npz
```

Direct checks showed:

```text
input_values.shape = (1500, 25)
B.shape = (r, 25)
input_values[0] == input_values[-1]
```

Those runs must not be used as the final SWE input-to-output comparison.

## Correct workflow

For each original SWE sample, the PDE data contains five Gaussian uplift
sources:

```text
xi, yi, Ti, sigma_i, Hi,  i = 1,...,5.
```

The seafloor uplift has the form

```math
u(x,y,t;\mu)
= \sum_{i=1}^{5}
  H_i \exp\left(
    -\frac{(x-x_i)^2 + (y-y_i)^2}{2\sigma_i^2}
  \right)
  \phi(t;T_i),
```

where `phi(t;T_i)` is the quintic temporal mollifier from the SWE data
generator.

The corrected GOATTM input pipeline is:

1. For every original sample, assemble the final total uplift field

```math
u_\infty(x,y;\mu)
= \sum_{i=1}^{5}
  H_i \exp\left(
    -\frac{(x-x_i)^2 + (y-y_i)^2}{2\sigma_i^2}
  \right).
```

2. Restrict the KL spatial domain to the far-ocean source region:

```text
0 <= x <= 40 km, 0 <= y <= 100 km.
```

3. Learn a centered KL basis from the ensemble of final total uplift fields.
Use the first `m = 200` KL modes.

4. For each sample, project each individual Gaussian source field onto that
final-uplift KL basis.  Then build the time-dependent latent input

```math
m(t;\mu)
= \sum_{i=1}^{5} \phi(t;T_i)\, c_i(\mu) - \bar c,
```

where `c_i(mu)` is the KL projection of the `i`th Gaussian source field and
`\bar c` is the centered-KL mean coefficient.

5. Store the paired data as GOATTM samples:

```text
qoi_observations.shape = (1501, 30)
input_values.shape = (1501, 200)
```

The first time is prepended at `t = 0`, so the stored time grid is

```text
0, 1, 2, ..., 1500.
```

## Required checks before training

Before submitting any optimizer job, verify:

```text
processed_data_kl_m200/summary.json:
  input_mode == kl_projected_time_dependent_gaussian_uplift
  input_parameter_dimension == 200
  kl_x_max == 40.0

sample_000000.npz:
  input_values.shape == (1501, 200)
  qoi_observations.shape == (1501, 30)
  max(abs(input_values - input_values[0])) > 0

optimizer checkpoint:
  b_matrix.shape == (latent_rank, 200)
```

If `B.shape[1] == 25`, the run is wrong.

## Rerun targets

Use the corrected manifest:

```text
/work2/08667/yuuuhang/stampede3/GOATTM/application/swe_problem/data/processed_data_kl_m200/manifest.npz
```

Rerun:

```text
r = 60, skewCP rank R = 120
r = 60, skewCP rank R = 80
```

Use the quadratic decoder and the same train/test split:

```text
ntrain = 1344
ntest = 384
sample_count = 1728
```
