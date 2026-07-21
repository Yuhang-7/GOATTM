# HGNVP GPU Optimization Log - 2026-07-21

## Scope

This log records the recent GPU optimization work for reduced Gauss-Newton / HGNVP actions in
`quad_goattm`, focused on the SWE reduced-order VarPro workflow with latent rank 120 and
EnergyTuckerTT quadratic dynamics.

Primary files touched:

- `quadrode_gpu_goattm/reduced_gn.py`
- `quadrode_gpu_goattm/adjoint.py`

## Baseline and Current Runtime

The timings below are single-GPU timings on Perlmutter A100 nodes, using the packed Cascadia
training data and batched HGNVP directions. Here `k` is the number of probing/sketch directions,
not the latent dimension.

| Case | Earlier optimized path | Current path | Speedup |
| --- | ---: | ---: | ---: |
| 1024 samples, k=16 | ~20.3 s | ~16.6 s | ~1.22x |
| 2048 samples, k=16 | ~40.8 s | ~33.3 s | ~1.22x |

Relative to the older adjoint path, the current implementation is close to 2x faster:

| Case | Old path | Current path | Speedup |
| --- | ---: | ---: | ---: |
| 1024 samples, k=16 | ~32.8-36.3 s | ~16.6 s | ~2.0x |
| 2048 samples, k=16 | ~65.6-72.8 s | ~33.3 s | ~2.0x |

Representative current breakdown for `1024 samples, k=16`:

- incremental tangent propagation: ~4.7 s
- decoder GN cotangent: ~7.1 s
- batched adjoint: ~4.8 s
- decoder normal Cholesky/factor: ~0.006 s

## GEMM/BMM Rewrite

The main source-level cleanup was to remove `torch.einsum` from the two HGNVP implementation
files and replace high-frequency contractions with explicit GEMM/BMM-style operations.

Current status:

- `reduced_gn.py`: 0 `einsum`
- `adjoint.py`: 0 `einsum`

Important rewrites:

- EnergyTucker reduced matrix construction now uses flattened tensor contractions and matmul.
- EnergyTucker parameter-core directional action now uses larger GEMMs instead of many small contractions.
- Linear/source tangent and adjoint VJPs now use matmul/batched matmul.
- Decoder dense/fallback Schur contractions now use matmul helpers.
- EnergyTuckerTT core gradient flush was rewritten as GEMM sequences.

The last EnergyTuckerTT reconstruction rewrite was numerically checked against the old einsum form:

- `k=16`: old 0.353 ms, new 0.279 ms, ~1.27x
- `k=32`: old 0.524 ms, new 0.539 ms, approximately neutral
- relative error: ~3.5e-16

That specific cache construction is not a major runtime contributor; the large speedups came from
rewriting contractions inside sample/time-step loops and decoder masked-cross paths.

## Decoder Masked-Cross Optimization

The largest decoder improvement came from replacing an output-loop/scatter formulation with a
direct quadratic-feature chunk GEMM:

```text
quad_dot = du_i * u_j + u_i * du_j
d_pred += quad_dot @ coeff_quad.T
```

This avoids materializing a large `sample x output_chunk x quadratic_feature` gradient tensor and
reduces scatter overhead.

Observed effect for `1024 samples, k=16`:

- decoder GN cotangent before this rewrite: ~10.1 s
- decoder GN cotangent after this rewrite: ~7.1 s

The remaining decoder bottleneck is still the masked-cross gather/scatter pattern:

- `_masked_cross_prediction_tangent_batched`
- `_masked_cross_output_cotangent_state_grad_batched`
- `_add_masked_cross_tangent_residual_terms_`

These are the best candidates for a fused CUDA/Triton kernel.

## Decoder Normal Solve

The decoder normal solve is already factorized and reused in HGNVP:

- `_decoder_normal_cholesky(...)` computes/caches the Cholesky factor.
- `_solve_decoder_normal(...)` uses `torch.cholesky_solve` when the factor is available.

Measured sizes for `1024 samples, k=16`:

- normal matrix: ~0.051 GB
- normal Cholesky: ~0.051 GB
- factor time: ~0.006 s

Conclusion: decoder normal factorization/solve is not the current bottleneck.

## Large Intermediate Tensors

The main peak-memory contributors are trajectory-shaped batched tangent/cotangent buffers.

Measured for `1024 samples, k=16, latent_dim=120`:

- `states_dot`: ~7.92 GB
- `state_cot`: ~7.92 GB
- rollout states: ~0.50 GB
- Picard iterates: ~1.48 GB

The `hessian_batch_flat` path now enables trajectory-buffer reuse by default. The decoder cotangent
routine overwrites the `states_dot` buffer with `state_cot` after each block's tangent values are no
longer needed. This reduces peak memory by roughly one trajectory buffer. Set
`GOATTM_REUSE_STATE_DOT_BUFFER=0` to disable this behavior for debugging.

## Candidate Fused Kernel

The next meaningful optimization is likely not another plain GEMM rewrite, but a fused masked-cross
decoder kernel that combines:

1. gather of `u_i, u_j, du_i, du_j`
2. computation of `du_i * u_j + u_i * du_j`
3. multiplication/accumulation into output tangent or state cotangent

The target is to reduce:

- temporary `quad_dot` materialization
- repeated `index_select`
- repeated `scatter_add_`
- many small kernel launches across feature chunks

The expected benefit is mostly in `decoder_gn_cotangent`, currently around 7 s per
`1024 samples, k=16`.

## Validation Performed

Representative checks:

- HVP old/new relative error: typically `1e-17` to `1e-18`
- local Gram relative error: typically `1e-17`
- EnergyTuckerTT reconstruction rewrite relative error: ~`1e-16`
- Dense fallback VJP rewrite relative error: ~`2e-16`

The current stable path uses `GOATTM_REUSE_STATE_DOT_BUFFER=1` semantics by default and keeps `0` as a debug fallback.

## Masked-Cross Shared Tangent/Residual Pass

A follow-up optimization merged two masked-cross decoder passes in the direct HGNVP cotangent path.
Previously, for each sample chunk the code computed the same quadratic tangent rows

```text
du_i * u_j + u_i * du_j
```

twice:

1. once to form the fixed decoder prediction tangent `d_pred_fixed`, and
2. once to accumulate the normal-equation right-hand side term `-dF.T @ weighted_residual`.

The new helper `_masked_cross_prediction_and_residual_terms_batched(...)` computes each masked-cross
feature chunk once, using it for both accumulations before moving to the next chunk. This does not
change the algebra; it removes one large gather/multiply/matmul traversal in the first decoder GN
cotangent stage.

Validation after the change:

- `python -m py_compile quadrode_gpu_goattm/reduced_gn.py`
- `tools/test_varpro_schur_components.py`
- `tools/test_gauss_newton_hvp.py`

Single-GPU timings on `nid001180` with packed Cascadia accum-displacement data, latent rank 120,
`k=16`, and `sample_chunk_size=1024`:

| Case | Previous current path | After shared pass | Speedup |
| --- | ---: | ---: | ---: |
| 1024 samples, k=16 | ~16.6 s | 12.94 s | ~1.28x |
| 2048 samples, k=16 | ~33.3 s | 25.37 s | ~1.31x |

Breakdown after the shared pass:

| Case | Incremental tangent | Decoder GN cotangent | Batched adjoint | Peak memory |
| --- | ---: | ---: | ---: | ---: |
| 1024 samples, k=16 | 3.89 s | 4.64 s | 4.36 s | 20.5 GiB |
| 2048 samples, k=16 | 7.68 s | 9.24 s | 8.44 s | 32.5 GiB |

The 2048 case now scales almost exactly as two 1024-sample chunks. The remaining dominant costs are
roughly balanced between incremental tangent propagation, decoder cotangent application, and the
batched adjoint. Decoder cotangent is still the best target for a future fused kernel because it
continues to perform masked gathers and scatter-adds in the state-cotangent stage.

## Default Buffer Reuse And Auto `d_pred_fixed` Cache

A second follow-up made two runtime policies explicit in `hessian_batch_flat`:

1. `GOATTM_REUSE_STATE_DOT_BUFFER` now defaults to enabled. This is a pure memory optimization: after
   decoder cotangents for a chunk are formed, `state_cot` may reuse the storage formerly occupied by
   `states_dot`.
2. `GOATTM_CACHE_D_PRED_FIXED` now defaults to `auto`. In the masked-cross direct path, the fixed
   prediction tangent `d_pred_fixed` computed during the normal-equation RHS pass can be cached and
   reused in the state-cotangent pass. The auto policy estimates the cache size and enables it only
   when it is at most half of currently free CUDA memory. Set `GOATTM_CACHE_D_PRED_FIXED=0` to force
   recomputation, or `1` to force caching.

Validation:

- buffer reuse same-model check: HVP relative error `3.2e-20`, qform relative error `0`
- `d_pred_fixed` cache same-model check: HVP relative error `2.3e-21`, qform relative error `0`

Additional timings on `nid001180`:

| Case | Policy | HGNVP action | Decoder GN cotangent | Peak memory |
| --- | --- | ---: | ---: | ---: |
| 1024 samples, k=16 | reuse on, no `d_pred` cache | 12.65 s | 4.62 s | 12.6 GiB |
| 1024 samples, k=16 | reuse on, `d_pred` cache on | 11.30 s | 3.25 s | 15.8 GiB |
| 2048 samples, k=16, chunk=2048 | reuse on, no `d_pred` cache | 22.98 s | 9.23 s | 23.9 GiB |
| 2048 samples, k=16, chunk=1024 | reuse on, `d_pred` cache on | 22.67 s | 6.49 s | 27.8 GiB |
| 1024 samples, k=32 | reuse on, no `d_pred` cache | 21.19 s | 8.65 s | 21.7 GiB |
| 1024 samples, k=32 | reuse on, auto `d_pred` cache | 18.51 s | 6.01 s | 28.1 GiB |

Practical policy from these measurements:

- For `k=16`, one GPU can process 2048 samples in one chunk comfortably.
- For `k=32`, use 1024-sample chunks; auto `d_pred` caching gives a useful speedup while staying well
  below 40 GB.
- For larger sketch ranks such as `k=64`, prefer smaller chunks unless the auto policy confirms that
  the cache fits.
