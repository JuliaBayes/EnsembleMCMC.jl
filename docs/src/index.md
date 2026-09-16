# EnsembleMCMC.jl

EnsembleMCMC samples a log density with coupled walkers. It provides Stretch,
differential-evolution (DE), and snooker moves, fixed mixtures, and threaded
evaluation. Julia 1.10 or later is required. The package is not registered yet.

## Installation

Until registration, install from the repository with an account that can access it:

```julia
using Pkg
Pkg.add(url="https://github.com/JuliaBayes/EnsembleMCMC.jl")
```

## Sample a target

Supply an unnormalized log density and either a coordinate-by-walker matrix or
an initial vector of coordinate vectors. The following example discards warmup
and collects two consecutive sample blocks.

```jldoctest quickstart
using EnsembleMCMC, Random, Random123

rng = Philox4x((42, 1));

initial = [randn(rng, 2) for _ in 1:24];

logdensity(x) = -sum(abs2, x) / 2;

state = initialize(rng, logdensity, initial; move=StretchMove());

step!(state, 100);

draws = sample!(state, 200); more = sample!(state, 50);

println((size(draws.positions), size(more.positions), current_state(state).sweep_count))

# output

((2, 24, 200), (2, 24, 50), 350)
```

This checks usage, not convergence. Choose warmup and run length for your target.

`step!(state, n)` performs `n` complete sweeps without collecting history.
`current_state(state)` returns `positions` as a vector of coordinate vectors,
along with `logdensities`, `accepted`, `walker_ids`, `attempts`, `acceptances`,
`move_index`, and `sweep_count`. Its arrays are mutable but read-only by
contract, borrowed until the next mutation; scalar metadata is captured. Do not
inspect the state concurrently with a step. A failed state rejects both
`current_state` and `snapshot` and cannot resume. Use `snapshot(state)` when a
stable, independent copy is needed:

```jldoctest state_views
using EnsembleMCMC, Random, Random123

rng = Philox4x((42, 3)); initial = [randn(rng, 2) for _ in 1:24];

state = initialize(rng, x -> -sum(abs2, x) / 2, initial);

step!(state, 10);

saved = snapshot(state);

saved_positions = deepcopy(saved.positions);

step!(state);

println((saved.sweep_count, current_state(state).sweep_count, saved.positions == saved_positions))

# output

(10, 11, true)
```

Snapshots are owned observations, not restart checkpoints. They include
the same fields as `current_state`; `positions` is a vector of coordinate
vectors, `acceptances` and `attempts` are counts per move, and `move_index` is
`0` before the first sweep.

## Moves and threads

```jldoctest mixture
using EnsembleMCMC, Random, Random123

rng = Philox4x((42, 2)); initial = [randn(rng, 2) for _ in 1:24];

moves = MoveMixture((StretchMove(), DEMove(), DESnookerMove()), [4, 2, 1]; schedule=:cycle);

state = initialize(rng, x -> -sum(abs2, x) / 2, initial;
    move=moves, executor=ThreadedExecutor());

println(sample!(state, 7).move_indices == [1, 1, 1, 1, 2, 2, 3])

# output

true
```

With the default `schedule=:random`, integer and floating-point weights define
fixed random selection probabilities. `schedule=:cycle` requires integer-valued
nonnegative weights and defines a repeating cycle. One move is selected for each
complete sweep; weights do not adapt. The default executor is
[`SerialExecutor`](@ref).

Start Julia with multiple threads, such as `julia --threads=4`, to use
[`ThreadedExecutor`](@ref). The target must support concurrent calls and must not
mutate its input. Groups update in order with frozen complements. Seeded results
do not depend on thread scheduling.
The executor uses the first few groups to choose a task size for each move.
Cheap groups stay serial when threading would cost more than it saves.

## Batched log densities

Use [`BatchedLogDensity`](@ref) to evaluate candidates together, one per column:

```jldoctest batched
using EnsembleMCMC, Random123

scalar(x) = -sum(abs2, x) / 2
batch!(values, positions) = (values .= scalar.(eachcol(positions)))
target = BatchedLogDensity(scalar, batch!)
initial = [-1.0 0 1 0; 0 -1 0 1]
state = initialize(Philox4x((42, 4)), target, initial)
println(size(sample!(state, 10).positions))

# output

(2, 4, 10)
```

The broadcast illustrates the API. Use a faster batched computation when available.
Initialization uses `scalar`. Each nonempty group calls `batch!` once, omitting
degenerate proposals. Fill every output, keep positions read-only, and retain
neither borrowed array. Both callbacks must compute the same log density.
Exceptions or invalid outputs invalidate the state.

With `SerialExecutor` or `ThreadedExecutor`, storage remains on the CPU.
`ThreadedExecutor` parallelizes proposals. The callback owns evaluation
parallelism. Cheap scalar targets may run faster without batching.

## CUDA sampling

Use [`KernelExecutor`](@ref) with a CUDA matrix and a device batch callback:

```julia
using CUDA, EnsembleMCMC, Random, Random123

CUDA.allowscalar(false)
rng = Philox4x((42, 5))
initial = CuArray(randn(rng, Float32, 8, 32))
scalar(x) = -sum(abs2, x) / 2
batch!(values, positions) = (values .= vec(-sum(abs2, positions; dims=1) / 2))
target = BatchedLogDensity(scalar, batch!)
state = initialize(rng, target, initial; move=DEMove(), executor=KernelExecutor())
step!(state, 100)
draws = sample!(state, 200)
host_positions = Array(draws.positions)
```

Initialization copies coordinates to the host for validation and calls `scalar`
with CPU vectors. Supply separate host and device target data when needed.
During sampling, proposals, log densities, acceptance flags, snapshots and history
stay on the input device. `current_state(state).positions` is a host vector of
device column views. Walker IDs, move indices and counts remain on the host.

The current Random123 backend generates controls on the host. Each group transfers
those controls to the device and returns small status/count records. Successful
sweeps transfer no coordinates or log densities to the host. Transfers such as
`Array(draws.positions)` are explicit.

The callback must use the current task's CUDA stream, or synchronize its own work
before returning. It must fill every output and leave input positions unchanged.
Sampler operations run on the input device and complete before returning.
CPU and GPU floating-point arithmetic may differ, so trajectories need not match
bit for bit. `KernelExecutor` also accepts CPU matrices for testing and supports
the three built-in moves and their mixtures.

GPU sampling suits expensive, parallel batch targets. Small targets can be slower
because kernel launches and group synchronization dominate.

## Inputs and outputs

Initial coordinates must be finite and span their dimension. Initial log densities
must be finite. Stretch requires at least `2d` walkers. DE and snooker require at
least `max(2d, 4)`. A target may return `-Inf` for proposals outside its support.
NaN and `+Inf` are errors.

| Field returned by `sample!` | Meaning |
| --- | --- |
| `positions` | coordinate × walker × sweep |
| `logdensities` | walker × sweep |
| `accepted` | walker × sweep |
| `move_indices` | selected move per sweep |
| `walker_ids` | logical IDs in storage order |

Rejections repeat the current state. Returned arrays own their storage. Repeated
calls to [`sample!`](@ref) continue the same ensemble. Initialization copies the
input coordinates and RNG. Reusing an unchanged RNG produces the same run. Use
independent seeds or streams for independent ensembles.

Random123 `Philox4x{UInt64}` and `Threefry4x{UInt64}` are supported. Do not mutate
state fields. If a target throws during a sweep, that state cannot resume. Correct
the target and initialize a fresh state.

## Scope

One state is one coupled ensemble. Its walkers are not independent chains.
The package preserves sweep and walker axes but does not compute ESS or test
convergence. Acceptance rate alone does not establish convergence.

Stretch and DE are affine-equivariant. Snooker is not generally affine-equivariant.
Affine equivariance does not solve multimodality.

Coordinate transforms, automatic initialization, and diagnostic integration
remain outside this package.
AbstractMCMC and LogDensityProblems adapters are not included. The interface is
experimental during the initial 0.0 series.

## License

EnsembleMCMC.jl is licensed under Apache 2.0. See the repository's
[`LICENSE.md`](https://github.com/JuliaBayes/EnsembleMCMC.jl/blob/main/LICENSE.md).
