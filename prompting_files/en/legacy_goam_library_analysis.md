## Legacy GOAM Library Analysis

### Purpose

This document summarizes the legacy `GOAM` library located at:

`/storage/yuhang/Myresearch/GOAM_clean/GOAM`

The goal of this document is not to preserve the old implementation as a design target, but to record:

- what the library was trying to do,
- what parts of the mathematical and software design are worth keeping,
- what parts are confusing, inconsistent, brittle, or partially broken,
- and what a rewrite should treat as "conceptual intent" rather than "code to port directly".


### High-Level Purpose of the Legacy Library

The old library is a research codebase for **goal-oriented operator inference**.

Its main intent is:

1. Learn a reduced latent state evolution model `u(t)`.
2. Learn a decoder that maps the latent state `u(t)` to a quantity of interest `q(t)`.
3. Fit the model using **QoI mismatch** rather than full-state mismatch.
4. Support adjoint-based gradients and Hessian-vector products.
5. Run training in parallel across samples using MPI.

This means the library is not simply a standard reduced-order model package. It is more specifically a framework for:

- fitting a latent dynamical system,
- fitting a QoI decoder,
- and optimizing parameters using first- or second-order methods,
- with the optimization objective focused on target observables.


### Core Mathematical Structure

The old library separates the learned model into two pieces.

#### 1. Decoder

The decoder maps latent state `u(t)` to QoI `q(t)`:

`q_hat(u) = V1 u + V2 vecsym(u u^T) + v`

So the decoder has:

- a linear term `V1`,
- an optional quadratic term `V2`,
- and an affine offset `v`.

This part is implemented in:

- `baselibrary/functionlibrary.py`
- `baselibrary/staticfunction.py`

#### 2. Reduced Dynamics

The latent state evolves by:

`du/dt = A u + H(u \otimes u) + B p + c`

where:

- `u` is the reduced latent variable,
- `p(t)` is the input or control,
- `A` is a linear reduced operator,
- `H` is a quadratic reduced operator,
- `B` maps input into latent dynamics,
- `c` is an affine offset.

There are two variants:

- `general`: directly learns a general quadratic operator `H`
- `stablized`: uses a constrained parameterization for the quadratic term to encode a structural stability-related property


### What the Legacy Library Was Actually Doing in Practice

At the workflow level, the main behavior of the old code is:

1. Read a multi-sample dataset.
2. Split the dataset into training and validation subsets.
3. Shard the dataset across MPI ranks.
4. For each candidate dynamic parameter:
   - solve the reduced forward system for all samples,
   - solve a least-squares normal equation for the decoder,
   - compute QoI loss,
   - solve adjoint equations,
   - compute gradients,
   - optionally compute Hessian-vector products via incremental equations.
5. Optimize the dynamic parameter using SciPy or custom optimizers.

This is an important conceptual point:

The old library is **not mainly doing joint unconstrained optimization over both decoder and dynamics at every step**.
Instead, the dominant pattern is:

- treat the dynamic parameter as the main optimization variable,
- then solve for the decoder from a normal equation inside the loss evaluation.

So the software is conceptually closest to a **variable projection / separable least-squares** structure.

That idea is one of the strongest parts of the old design and is likely worth preserving in the rewrite.


### Effective Main Modules

The old repository is organized into several folders, but only some of them are real working layers.

#### `solver/cross_validation_problem.py`

This is effectively the top-level user-facing orchestration class.

Responsibilities:

- preprocess dataset,
- create train and test problems,
- drive optimization,
- save logs and checkpoints,
- expose validation methods.

This is the closest thing the old library has to a "main API".

#### `solver/multiple.py`

This contains the `opinf_problem` class.

Responsibilities:

- hold the multi-sample training problem,
- coordinate MPI parallelism,
- solve forward / adjoint / normal equations across all local samples,
- evaluate total loss,
- compute gradient and Hessian-vector products,
- manage regularization and initialization.

This is the actual core engine of the library.

#### `solver/single_problem_family.py`

This contains the single-sample problem.

Responsibilities:

- solve forward problem,
- solve adjoint problem,
- compute single-sample loss,
- compute single-sample gradients,
- compute incremental forward/adjoint solutions,
- compute Hessian actions.

This is the mathematical core at sample level.

#### `solver/preprocess.py`

Responsibilities:

- read the original dataset,
- split train/test,
- normalize data,
- write one per-rank `.npz` file for MPI training.

This is the data-loading boundary, though in practice it is inconsistent.

#### `baselibrary/functionlibrary.py` and `baselibrary/staticfunction.py`

These hold the parameterized model classes and many low-level static formulas:

- decoder evaluation,
- reduced dynamics evaluation,
- gradients with respect to state and parameters,
- second derivatives,
- Hessian-related terms.

Conceptually, this is the model-and-derivative layer.

#### `solver/implicit_solver.py`

This provides the Crank-Nicolson-based forward/adjoint/integral routines.

It is the main implementation for the CN path.

#### `initialization/opinf_initialize.py`

This is the initialization layer.

Its main role is:

- use full-order data,
- compute reduced basis via randomized SVD,
- regress reduced operators,
- produce an initial dynamic model.

This file reflects the original "operator inference initialization" idea.


### Data Contract the Legacy Library Appears to Want

The library appears to want a dataset with:

- `bc_datas_list`: input trajectories, shape `(dp, Nt+1, Ns)`
- `QoI_list`: quantity-of-interest trajectories, shape `(dq, Nt+1, Ns)`
- `Tlist`: time grid, shape `(Nt+1,)`
- `param_num`: number of samples `Ns`

Optional:

- `QoI_dt_list`: time derivative of QoI
- `fulldata` or full state snapshots for initialization

However, one major source of confusion is that the codebase uses multiple inconsistent names for the full-state data:

- `full_solutions_list`
- `fulldata`
- `fullmodel`

These are not consistently aligned.


### Effective End-to-End Flow

The actual flow of the legacy code can be summarized as:

```text
original dataset
-> preprocess into train/test and MPI rank shards
-> create cross_validation_opinf_problem
-> create training opinf_problem
-> for each optimization step:
   -> update dynamic parameter
   -> forward solve latent dynamics
   -> assemble normal equation for decoder
   -> solve decoder
   -> compute QoI loss
   -> solve adjoint equations
   -> compute dynamic gradient
   -> optionally compute Hessian-vector products
-> validate on held-out data
-> save parameter snapshots and logs
```


### Good Ideas in the Legacy Library

These are the parts that are conceptually strong and may be worth preserving in the rewrite.

#### 1. Goal-oriented learning objective

The old library is explicitly built to fit QoIs rather than full states.

That is not accidental or superficial. It is the central modeling idea, and it distinguishes the library from standard reduced modeling code.

#### 2. Separation between dynamics and decoder

The software treats:

- latent evolution model `g`
- QoI decoder `f`

as separate components.

That separation is mathematically clean and should remain in the new design.

#### 3. Decoder elimination through normal equations

The old library repeatedly solves for the decoder inside the training loop rather than always optimizing everything jointly.

This is one of the most valuable structural ideas in the old code.

It reduces the outer optimization burden and makes the training problem more structured.

#### 4. Adjoint-based gradient and incremental Hessian logic

The code does not rely on naive finite differences for optimization.

It includes:

- adjoint equations,
- incremental forward systems,
- incremental adjoint systems,
- Hessian-vector products.

Even though some branches are incomplete, the presence of this machinery shows the intended algorithmic sophistication.

#### 5. MPI sample-parallel design

The library distributes samples across MPI ranks and accumulates global quantities on rank 0.

For the intended workload, this is a sensible scaling strategy.

#### 6. Operator inference initialization

The initialization logic using full-state data and reduced basis learning is useful.

Even if the implementation is messy, the conceptual intent is good:

- use full-order data for a structured initial reduced dynamic model,
- then refine with goal-oriented training.


### Main Categories of Confusion in the Legacy Library

The biggest problem is not that the code has one bug here or there. The bigger problem is that the library mixes:

- conceptual layers,
- historical experiments,
- old naming conventions,
- half-deprecated branches,
- and several incompatible assumptions about data and optimization.

Below are the main categories of confusion.


### 1. Documentation and Real Implementation Diverged

The README does not describe the current code accurately.

Examples:

- It references files that are not present.
- It describes interfaces that do not match the current implementation.
- It describes configuration choices using strings that do not match the actual code checks.

This means the library cannot be understood safely from documentation alone.


### 2. Data Format Logic Is Inconsistent

This is one of the most serious design problems.

The library uses inconsistent names and assumptions for full-order data:

- some places expect `full_solutions_list`,
- some places expect `fulldata`,
- some places refer to `fullmodel`.

At the same time:

- some interfaces talk about `.npz` input,
- some real preprocessing code opens HDF5 through `h5py`,
- and the comments imply multiple incompatible dataset organizations.

So the "true" data contract is not centralized anywhere.


### 3. The Boundary Between Research Branches and Core Functionality Is Blurry

The code contains:

- real production-like paths,
- testing utilities,
- numerical checks,
- half-finished alternative optimizers,
- and older ideas that were never fully removed.

These are all mixed in the same modules.

As a result, it is hard to tell:

- which path is the intended main path,
- which path is experimental,
- which path is obsolete,
- and which path is simply broken.


### 4. Option Naming Is Confusing and Sometimes Wrong

Several options use names that do not match their meaning cleanly.

Examples:

- `dynamic_type` and `dynamicform` are different concepts, but the code occasionally mixes them in conditionals.
- regularization names in README do not match the strings actually used in code.
- `stablized` is misspelled throughout and has become part of the interface.

This creates avoidable ambiguity for both users and developers.


### 5. Partial Features Are Mixed with Fully Implemented Features

Some features look like supported options because:

- they appear in README,
- they appear in argument lists,
- they appear in branch conditions,
- or they have helper functions.

But several of these are not fully connected end-to-end.

This is especially true for:

- some optimizer modes,
- some Hessian-related branches,
- some initialization branches,
- some parsing/task orchestration features,
- some time-derivative branches.


### 6. Orchestration Logic Is Spread Across Too Many Layers

The library has no crisp separation between:

- mathematical formulation,
- sample-level PDE/ODE logic,
- MPI distribution,
- logging,
- parameter saving,
- optimization method dispatch,
- and initialization strategy.

Instead, these concerns bleed across classes.

That makes it hard to modify one area without understanding several others.


### Concrete Weak Points and Broken or Suspicious Logic

Below is a more direct list of specific weaknesses or suspicious points.

#### A. `taskparsing` is effectively not implemented

The task parsing layer exists in directory structure, but functionally it is not a real subsystem.

- `taskparsing/parsetask.py` contains a placeholder function with `pass`
- `taskparsing/validate_problem.py` is empty

So this part should not be considered real functionality.

#### B. README describes a main workflow that no longer exists cleanly

The README points to example entry files that are absent and gives a picture of the library that does not match the current repository state.

This is a major source of false confidence.

#### C. Preprocessing contains at least one real data-writing bug

The test-set `QoI_dt_list` is written incorrectly in one path.

This means the preprocessing layer is not only confusing, but also unsafe to trust blindly.

#### D. Train/test split is deterministic despite `randomseed`

There is a `randomseed` argument, but the actual split logic uses the first samples for train and the last samples for test.

So the API suggests random splitting, but the implementation does not actually do that.

#### E. Initialization interface is inconsistent

The OpInf initialization path is conceptually important, but the call signatures and usage are inconsistent enough that the path is not reliable as a clean public feature.

#### F. Full Hessian and Gauss-Newton support are not consistently wired

The code includes substantial Hessian-action machinery, but some higher-level orchestration paths refer to missing task names or inconsistent function names.

That strongly suggests some second-order paths were renamed or refactored incompletely.

#### G. Some options are present as strings, but not fully supported as a stable workflow

Examples include:

- `LM`
- `Gauss_Newton`
- `LRSFN`
- `exact_LRSFN`

These appear in code, but the actual optimization driver is not organized as a clean, uniformly supported dispatcher for them.

#### H. RK4 and CN support are asymmetric

The code suggests both are solver choices, but some extended features, especially involving QoI time-derivative misfit, are only properly implemented for the CN path.

So "solver_type is selectable" is only partly true if one expects all downstream features to behave the same.

#### I. MPI object lifecycle is fragile

Some constructors can exit early if thread count and MPI size do not match, leaving partially initialized objects behind.

This is not a safe design for a reusable library API.


### What the Old Code Is Best Understood As

The cleanest way to interpret the old code is:

It is a **research prototype with a strong mathematical core and a weak software boundary**.

More specifically:

- the central optimization idea is meaningful,
- the model decomposition is meaningful,
- the adjoint/incremental formulation is meaningful,
- the MPI sample-parallel strategy is meaningful,
- but the code organization around those ideas is unstable and historically layered.

So a rewrite should preserve the conceptual backbone, not the exact module boundaries.


### Recommended Interpretation for the Rewrite

The rewrite should treat the old library as a source of:

- mathematical intent,
- workflow intent,
- and algorithmic ingredients,

not as a source of software architecture to mirror.

The old code is most useful as a reference for:

1. What model family was intended.
2. What loss terms were intended.
3. What derivative information was intended.
4. What train-time workflow was intended.

It is much less useful as a reference for:

1. public API design,
2. data schema design,
3. option naming,
4. module boundaries,
5. logging and orchestration structure.


### Suggested "Keep vs Rewrite" View

#### Keep conceptually

- goal-oriented QoI fitting
- separate dynamic and decoder models
- optional quadratic decoder
- optional quadratic reduced dynamics
- structured/stabilized reduced dynamics option
- decoder elimination via normal equations
- adjoint-based gradients
- incremental Hessian-vector products
- MPI sample parallelism
- OpInf-style initialization from full-state data

#### Rewrite aggressively

- dataset schema
- preprocessing
- file naming and metadata conventions
- public API
- optimizer dispatch
- configuration system
- naming conventions
- documentation
- MPI orchestration boundary
- logging and checkpointing
- validation and experiment management


### Final Summary

The legacy GOAM library was trying to be a **goal-oriented reduced-model training framework with second-order optimization support**.

Its best ideas are:

- fit QoIs rather than full states,
- separate latent dynamics from output decoder,
- solve decoder by least squares inside the outer loop,
- compute exact-ish derivative information through adjoint and incremental systems,
- and scale sample-wise through MPI.

Its biggest problems are:

- inconsistent data contracts,
- stale documentation,
- incomplete feature wiring,
- unclear public entry points,
- mixed experimental and core code paths,
- and many signs of historical drift.

So the rewrite should preserve the old library's **mathematical intent** while replacing most of its **software structure**.
