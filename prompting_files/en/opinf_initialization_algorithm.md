# GOATTM Preprocess and OpInf Initialization Algorithm

## Purpose

This note serves two purposes:

1. define what the preprocess layer in `GOATTM` should do for raw datasets;
2. provide a clean `OpInf` initialization algorithm that matches the current `GOATTM` parametrization.

This document deliberately does **not** copy the old `GOAM` initializer structure.  
We only keep the mathematically meaningful objectives and rewrite them using the current `GOATTM` layering and distributed abstractions.


## 1. Responsibilities of the preprocess layer

Going forward, the user should only need to provide **raw, unprocessed data**, and the library should handle preprocessing internally.

At minimum, the preprocess layer should be responsible for:

1. train/test splitting;
2. training-statistics normalization;
3. construction of the intermediate objects required by initialization.

This information should later be reflected in the run outputs:

- which raw manifest was used;
- how the train/test split was produced;
- whether normalization was applied;
- which normalization statistics were used;
- whether initialization was random or `OpInf`-based.


## 2. Normalization

### 2.1 Goal

We normalize each observed training-data dimension so that, on the training set,

\[
\mu_j = 0, \qquad \sigma_j = 1.
\]

At the moment this should be applied to:

- `q(t)`, stored as `qoi_observations`;
- `p(t)`, stored as `input_values`.

This should not be applied to:

- `observation_times`;
- `input_times`;
- `u0`.

### 2.2 Statistics

For the training manifest, aggregate across all samples and all time points.

For the `j`-th component of `q`,

\[
\bar q_j = \frac{1}{N_q}\sum_{s,n} q_{s,n,j},
\qquad
\sigma_{q,j}^2
=
\frac{1}{N_q}\sum_{s,n} q_{s,n,j}^2 - \bar q_j^2.
\]

For the `j`-th component of `p`,

\[
\bar p_j = \frac{1}{N_p}\sum_{s,m} p_{s,m,j},
\qquad
\sigma_{p,j}^2
=
\frac{1}{N_p}\sum_{s,m} p_{s,m,j}^2 - \bar p_j^2.
\]

If a standard deviation is too small, replace it by `1` to avoid division by zero.

### 2.3 Application

Both training and test data use the **same statistics computed from the training data**:

\[
\widetilde q_{s,n,j} = \frac{q_{s,n,j}-\bar q_j}{\sigma_{q,j}},
\qquad
\widetilde p_{s,m,j} = \frac{p_{s,m,j}-\bar p_j}{\sigma_{p,j}}.
\]

### 2.4 Distributed organization

The preprocess layer should not require one rank to load all samples.

Recommended workflow:

1. each rank reads only its local subset of the training samples;
2. each rank accumulates local `sum / sumsq / count`;
3. global mean and variance are obtained via `allreduce`;
4. each rank normalizes and writes only its local output samples;
5. rank 0 writes the normalization stats and the normalized manifests.


## 3. Objective of the OpInf initialization

We want to construct an initial dynamics parameter

\[
\mu_g = (S, W, \mu_H, B, c),
\]

and an initial decoder

\[
\mu_f = (V_1, V_2, v_0).
\]

The desired initialization route is:

1. build the best rank-`r` linear subspace from `q(t)`;
2. define the latent state
   \[
   u(t) = V_1^\top q(t);
   \]
3. compute \(\dot u(t)\) by finite differences;
4. fit
   \[
   \dot u = W u + H(u,u) + B p + c,
   \]
   under the constraints
   - `S = 0`,
   - `W` skew-symmetric,
   - `H` energy-preserving;
5. evolve the latent dynamics and then solve for the best decoder
   \[
   (V_1, V_2, v_0).
   \]


## 4. Step 1: initial V1 from q

Let the training snapshots be assembled into

\[
Q_{\mathrm{train}}
=
\begin{bmatrix}
q^{(1)}(t_0) & q^{(1)}(t_1) & \cdots &
q^{(N_s)}(t_{N_T})
\end{bmatrix}
\in \mathbb{R}^{d_q \times N_{\mathrm{snap}}}.
\]

Take the rank-`r` truncated SVD:

\[
Q_{\mathrm{train}} \approx U_r \Sigma_r V_r^\top.
\]

Then define

\[
V_1^{(0)} = U_r^\top \in \mathbb{R}^{r \times d_q},
\]

or equivalently store it in the decoder convention used by the codebase.

This is the best rank-`r` linear reconstruction subspace for `q`.

### Distributed implementation suggestion

Avoid gathering all snapshots to one rank.

A cleaner route is:

1. each rank forms its local snapshot block;
2. build a randomized range finder or local Gram contribution;
3. reduce a small global matrix;
4. rank 0 computes the final SVD and broadcasts `V1`.


## 5. Step 2: latent trajectories and time derivatives

For each training sample,

\[
u^{(s)}(t_n) = V_1^\top q^{(s)}(t_n).
\]

Then approximate time derivatives by finite differences:

\[
\dot u^{(s)}(t_n) \approx D_t u^{(s)}(t_n).
\]

Recommended choice:

- centered differences at interior points;
- one-sided differences at boundaries;
- if the time grid is nonuniform, use explicit nonuniform finite-difference formulas.


## 6. Step 3: fit W, H, B, c

We fit

\[
\dot u = W u + H(u,u) + B p + c
\]

subject to:

- \(W^\top = -W\),
- \(u^\top H(u,u) = 0\),
- \(S = 0\).

### 6.1 Feature matrices

Define

\[
\phi_H(u) = \zeta(u) \in \mathbb{R}^{s},
\qquad
s = \frac{r(r+1)}{2}.
\]

After stacking all samples and all time points, let

\[
U = [u_k] \in \mathbb{R}^{r \times N},
\qquad
Z = [\zeta(u_k)] \in \mathbb{R}^{s \times N},
\qquad
P = [p_k] \in \mathbb{R}^{d_p \times N},
\qquad
\dot U = [\dot u_k] \in \mathbb{R}^{r \times N}.
\]

Then the regression target is

\[
\dot U \approx WU + HZ + BP + c \mathbf{1}^\top.
\]

### 6.2 Parametrization of W

We do not infer a full unconstrained `A`, and we do not introduce `S` during initialization.

Instead, as requested,

\[
S = 0, \qquad A = W, \qquad W^\top = -W.
\]

So write `W` in the skew basis induced by `w_params`:

\[
\mathrm{vec}(W) = \mathcal{B}_W \theta_W.
\]

### 6.3 Parametrization of H

`GOATTM` already has an energy-preserving `\mu_H` parametrization, so we use it directly:

\[
\mathrm{vec}(H) = \mathcal{B}_H \theta_H,
\]

where `\theta_H = \mu_H` and `\mathcal{B}_H` is the basis induced by `mu_h_to_compressed_h`.

This means the energy-preserving constraint is already absorbed into the basis representation; no separate KKT system is needed for that constraint.


## 7. Constrained matrix least squares

Collect all unknowns into

\[
\theta =
\begin{bmatrix}
\theta_W \\
\theta_H \\
\mathrm{vec}(B) \\
c
\end{bmatrix}.
\]

Equivalently, define a global basis

\[
\mathrm{vec}(X) = \mathcal{B}\theta,
\]

with

\[
X = \begin{bmatrix} W & H & B & c \end{bmatrix}.
\]

Let the global regressor be

\[
R =
\begin{bmatrix}
U \\
Z \\
P \\
\mathbf{1}^\top
\end{bmatrix}.
\]

Then the least-squares problem becomes

\[
\min_\theta
\frac12 \|X R - \dot U\|_F^2
 \frac{\lambda}{2}\|\theta\|_2^2,
\qquad
\mathrm{vec}(X)=\mathcal{B}\theta.
\]

After vectorization:

\[
\min_\theta
\frac12 \|K\theta - y\|_2^2
 \frac{\lambda}{2}\|\theta\|_2^2,
\]

where

\[
K = (R^\top \otimes I_r)\mathcal{B},
\qquad
y = \mathrm{vec}(\dot U).
\]

The normal equation is

\[
(K^\top K + \lambda I)\theta = K^\top y.
\]

This is the core constrained matrix least-squares problem needed for the future `OpInf` initializer.


## 8. Step 4: recompute the decoder from the initialized dynamics

Once we have an initialized dynamics parameter `\mu_g^{(0)}`, we should not stop at the first linear `V1`.

A cleaner initialization is:

1. rollout the latent dynamics with `\mu_g^{(0)}`;
2. feed the resulting `u(t)` into the decoder normal equation;
3. solve for the optimal decoder
   \[
   (V_1, V_2, v_0).
   \]

This makes the decoder initialization consistent with the initialized dynamics.


## 9. Recommended workflow integration

Eventually this should become an explicit workflow:

1. `raw manifest`
2. `train/test split`
3. `normalization materialization`
4. `initialization`
   - `random`
   - `opinf`
5. `trainer`

And the run directory should contain provenance such as:

- `preprocess_config.json`
- `normalization_stats.npz`
- `normalized_train_manifest.npz`
- `normalized_test_manifest.npz`
- `initialization_summary.json`

so that the logs answer:

- whether the data was preprocessed;
- which normalization was used;
- where the initialization came from.


## 10. Current status

Already completed:

- training-statistics normalization;
- normalized train/test dataset materialization;
- basis-parameterized constrained matrix least-squares for energy-preserving `H`;
- corresponding tests.

Not yet completed:

- the full distributed `OpInf` initializer;
- formal integration into the training workflow and logger.
