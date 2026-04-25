## GOATTM Rewrite Scope Prompt

### Purpose

This prompt records the current rewrite direction for `GOATTM`.

The goal is **not** to finish everything at once.
The goal is to record the current shared understanding so that future implementation work stays aligned with:

- the original mathematical intent of the old `GOAM` library,
- the practical performance requirements,
- the distributed training requirements,
- and the desire to avoid carrying forward old structural confusion.


### Core Intent of the Rewrite

The rewrite should preserve the strongest ideas from the legacy library while replacing most of its software structure.

The new library should still be centered around:

- goal-oriented reduced modeling,
- learning latent reduced dynamics,
- learning a QoI decoder,
- fitting QoI trajectories rather than full-state trajectories,
- solving the decoder through a structured least-squares / normal-equation step inside training,
- and supporting high-performance execution.

The rewrite should **not** be treated as a direct code port.
It should be treated as a reimplementation of the core mathematical and computational ideas.


### Important Legacy Strengths That Must Not Be Ignored

#### 1. Performance was a first-class concern

The old library did not merely implement the math in a naive way.
Many low-level kernels for matrix, polynomial, quadratic-form, gradient, and second-order operations were accelerated using `numba`.

This was done to make the training process fast enough to be practical.

That means:

- performance considerations are part of the problem definition,
- low-level kernels should continue to be treated as optimization targets,
- and a rewrite that is conceptually clean but significantly slower would be incomplete.

#### 2. Distributed execution was a first-class concern

The old library was also designed with distributed-node execution in mind.
MPI-based sample-parallel execution was introduced because the training workload needed to run on distributed systems.

Even if the old MPI implementation is not ideal, the intent behind it is important and valid.

That means:

- distributed training is not an optional afterthought,
- sample-parallel execution should remain part of the design space,
- and the rewrite should leave room for scalable parallel execution.


### First-Phase Rewrite Strategy

The rewrite should proceed in a deliberately reduced scope.

We do **not** need to reproduce every feature from the old library immediately.
Instead, the first phase should establish a clean and reliable core.

The first phase should prioritize:

- clean data contracts,
- clean model contracts,
- clean solver interfaces,
- clear separation of concerns,
- preservation of performance hooks,
- preservation of distributed execution hooks,
- and a stable main training path.


### Explicit Non-Goals for the First Phase

The following items are currently considered acceptable to leave out of the first implementation phase.

#### 1. `QoIdt`-related functionality

The rewrite does not need to support:

- `QoIdt` loss terms,
- `QoIdt` gradient terms,
- `QoIdt` Hessian-related terms,
- or any training path that depends on QoI time-derivative mismatch.

This should be treated as explicitly deferrable functionality.

#### 2. Legacy numerical solver details that are known to be temporary

The numerical solver layer will be modified later.
Therefore the rewrite should not hard-wire itself around the exact old solver structure.

We should preserve interfaces and solver slots, but not assume the old numerical methods are final.

#### 3. Full feature parity with the old library

The first phase does not need to reproduce:

- every experimental optimizer branch,
- every second-order feature branch,
- every old command/task parsing path,
- every historical compatibility path,
- or every legacy file naming convention.

#### 4. HDF5-based data loading

The rewrite should not continue using the awkward old HDF5-centered data access path as the primary data interface.


### Data Layer Direction

The new library should move away from the old HDF5-based workflow.

The preferred direction is:

- use a collection of `.npz` files,
- define a clean and explicit schema,
- and make the data format naturally compatible with sample-parallel or shard-parallel execution.

The old HDF5 route existed partly as a workaround for server-side constraints and was awkward in practice.

The rewrite should therefore aim for:

- straightforward `.npz`-based storage,
- consistent field names,
- clean train/validation/test indexing,
- and a simple manifest or indexing strategy if multiple files are used.

The new implementation should avoid legacy naming ambiguity such as:

- `fulldata`
- `fullmodel`
- `full_solutions_list`

The data contract should be singular, explicit, and documented.


### Model Structure to Preserve

The rewrite should preserve the separation between:

- the latent reduced dynamics model,
- and the QoI decoder.

The rewrite should also preserve the core training idea:

- treat the dynamic model as the outer optimization object,
- solve the decoder in a structured way from the current latent trajectories,
- then evaluate the QoI-based loss.

This structural idea is one of the most valuable aspects of the old library and should remain central.


### Solver Layer Direction

The solver layer should be redesigned as a replaceable module boundary.

The new library should avoid embedding solver-specific logic too deeply into every layer.

Instead, the rewrite should aim for solver interfaces such as:

- forward solve interface,
- adjoint solve interface,
- incremental solve interface,
- and possibly Hessian-action interface later.

This is important because the numerical solvers are expected to change later.
The architecture should therefore allow those changes without forcing a redesign of the entire library.


### Performance Requirements

The rewrite must explicitly respect performance constraints.

It is not enough for the new library to be mathematically correct and structurally clean.
It should also be written so that high-frequency kernels can be accelerated.

The design should therefore preserve room for:

- `numba`-accelerated low-level kernels,
- minimized Python overhead in inner loops,
- preallocation and structured linear algebra where appropriate,
- and separation between high-level orchestration code and low-level compute kernels.

The old code's `numba` usage reflects a real need and should be treated as a design requirement, not an implementation accident.


### Distributed Execution Requirements

The rewrite should preserve the ability to support distributed execution.

At minimum, the architecture should leave a clean path for:

- MPI-based sample parallelism,
- rank-local data ownership,
- global reductions for loss and gradient aggregation,
- and future scaling on distributed nodes.

Even if the first implementation is simpler than the old MPI code, it should not close the door on distributed training.

The long-term goal is:

- cleaner distributed orchestration than the old library,
- while preserving the practicality of running on multi-node systems.


### Recommended Architectural Separation

The rewrite should move toward a clearer separation of layers.

Suggested conceptual layers:

#### 1. Kernel / math layer

Contains:

- quadratic and polynomial kernels,
- decoder evaluation kernels,
- dynamic evaluation kernels,
- derivative kernels,
- normal-equation assembly kernels,
- and other hotspots suitable for `numba`.

#### 2. Model / problem layer

Contains:

- decoder model definition,
- reduced dynamics model definition,
- loss definition,
- adjoint problem definition,
- and training problem structure.

This layer should describe math, not files or MPI.

#### 3. Runtime / execution layer

Contains:

- data loading,
- sample iteration,
- distributed aggregation,
- logging,
- checkpointing,
- optimizer drivers,
- and execution orchestration.

This separation is important because it allows:

- kernel optimization without disturbing orchestration,
- solver replacement without redesigning data loading,
- and distributed execution changes without rewriting the mathematical core.


### Configuration Direction

The rewrite should avoid the old pattern of unclear or historically inconsistent string options.

The new configuration style should aim for:

- explicit config objects or structured settings,
- unified naming,
- consistent terminology,
- and minimal reliance on magic string branches.

In particular, the rewrite should avoid carrying forward confusing interface artifacts from the old code where possible.


### First-Phase Success Criteria

The first phase of the rewrite should be considered successful if it provides a clean main path that can:

1. Read the new `.npz`-based dataset format.
2. Represent the reduced dynamics model and QoI decoder cleanly.
3. Run the main training workflow without `QoIdt`.
4. Solve the decoder through the intended structured inner solve.
5. Compute the main QoI-based loss.
6. Compute adjoint-based gradients.
7. Preserve a path for `numba` acceleration.
8. Preserve a path for distributed sample-parallel execution.

This is enough for the first stage.
Broader feature recovery can come later.


### Summary

The current rewrite direction is:

- preserve the old library's mathematical backbone,
- preserve its performance intent,
- preserve its distributed execution intent,
- reduce the scope for the first implementation,
- remove awkward HDF5-centric data handling,
- defer `QoIdt`,
- and build a cleaner architecture that can accept future solver changes.

This prompt is meant to act as a stable reference while the rewrite proceeds incrementally.
