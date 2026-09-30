# EnsembleMCMC.jl

```@meta
CurrentModule = EnsembleMCMC
```

EnsembleMCMC samples a log density with coupled walkers. It provides Stretch,
differential-evolution (DE), snooker, and Gaussian replacement moves, fixed mixtures, and threaded
evaluation. Julia 1.10 or later is required.

## Installation

Install the registered release and Random123. `initialize` takes a Random123
`Philox4x` or `Threefry4x` RNG, so `using Random123` must work in your environment:

```julia
using Pkg
Pkg.add(["EnsembleMCMC", "Random123"])
```

For features not yet in the registered release, install the development version:

```julia
using Pkg
Pkg.add(url="https://github.com/JuliaBayes/EnsembleMCMC.jl")
```

## Importing the verbs

The generic verbs `initialize`, `step!`, `sample!`, `current_state`, `snapshot`,
`synchronize!`, `validate_positions` and `acceptance_rate` are not exported. The
package is mainly used through adapters such as BAT.jl, and these names clash with
StatsBase and AbstractMCMC. Import them, or call them qualified:

```julia
using EnsembleMCMC: initialize, step!, sample!, current_state, snapshot
EnsembleMCMC.acceptance_rate(state)
```

On Julia 1.11 and later they are declared `public`. The types and moves are
exported.

## Sample a target

Supply an unnormalized log density and either a coordinate-by-walker matrix or
an initial vector of coordinate vectors. The following example discards warmup
and collects two consecutive sample blocks.

```jldoctest quickstart
using EnsembleMCMC, Random, Random123
using EnsembleMCMC: initialize, step!, sample!, current_state, snapshot

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

`sample!(state, n; thin=1)` runs `n * thin` sweeps and stores every `thin`-th one.
`acceptance_rate(state)` returns the cumulative acceptance rate of each move.
The history of `sample!` is allocated up front and returned only on success. If
the target throws during a sweep, the state becomes invalid and the collected
history is lost. For full control, call `step!` in a loop and keep `snapshot`s.

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
using EnsembleMCMC: initialize, step!, sample!, current_state, snapshot

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

Snapshots are owned observations, not restart checkpoints. Sampling is
continuable in process, not resumable from disk. They include
the same fields as `current_state`; `positions` is a vector of coordinate
vectors, `acceptances` and `attempts` are counts per move, and `move_index` is
`0` before the first sweep.

## External sampler integration

`initialize(...; logdensities=cached_values)` reuses initial log densities
without calling the target. Values are copied, retain their own precision, and
must match the target and walker order. NaN and `+Inf` remain invalid.

`current_state` also exposes the last candidates, their log densities, and actual
acceptance probabilities. These fields include rejected transitions and follow
the same borrowing rules as positions.

[`synchronize!`](@ref) copies external positions and matching cached densities
into a valid state. It preserves walker IDs, RNG state, mixture phase, and counts.
The caller owns density consistency and affine rank. Inputs must not alias the
state's borrowed arrays. This operation clears the last-transition metadata.
Use [`validate_positions`](@ref) to check finite coordinates and affine rank after
retry initialization. Initialization uses the same validation. Ordinary
synchronization does not repeat the rank check.

`step!(state, rng; proposal_index=1)` accepts an externally addressed RNG of the
same type used at initialization. Reserve two remaining partition levels for
purposes and walkers. Inner mixture selection uses purpose 2, leaving purpose 1
for an outer mixture. Ordinary `step!(state)` retains its standalone RNG law.

## Moves and threads

```jldoctest mixture
using EnsembleMCMC, Random, Random123
using EnsembleMCMC: initialize, step!, sample!, current_state, snapshot

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
Each group uses balanced, contiguous chunks, with at most one task per default-pool
thread and no more tasks than walkers in the group. There is no timing-based
calibration or cost-based serial fallback.

`ThreadedExecutor` spawns tasks for every group, which costs a few microseconds per
task. For cheap targets `SerialExecutor()` is faster. Example timings for d = 10
and 100 walkers: with a very cheap target, a serial sweep takes 8 µs and a threaded
sweep 29 µs. With a target of about 0.6 µs per evaluation, the threaded sweep is
about twice as fast. `ThreadedExecutor(; min_chunk=k)` gives each task at least `k`
walkers, which limits the task count for cheap targets. Results do not depend on
`min_chunk`. An error in the target is rethrown as the original exception.

### Executors and reproducibility

`SerialExecutor`, `ThreadedExecutor` and `KernelExecutor` on the CPU give bitwise
identical trajectories for `StretchMove` and `DEMove`. `DESnookerMove` and
`GaussianReplacementMove` differ at rounding level, and a GPU can differ further
from the CPU. Do not compare trajectories across executors for those moves. The
samples are statistically equivalent.

### Gaussian replacement

[`GaussianReplacementMove`](@ref) fits a Gaussian to the frozen complement before
updating each group. It proposes independent replacements with the exact Hastings
correction. This specializes the replacement move in
[Goodman and Weare (2010), equation (12)](https://msp.org/camcos/2010/5-1/camcos-v5-n1-p04-p.pdf).

```julia
move = MoveMixture((DEMove(), GaussianReplacementMove()), [1, 1])
state = initialize(rng, logdensity, initial; move)
```

The default `shrinkage=:auto` uses the proposal covariance `(1 + d/m) C`, where
`C` is the sample covariance of the `m` complement walkers. This is affine
invariant, so the acceptance rate does not depend on the conditioning of the
target. If `m <= d` or the fit fails, the group falls back to shrinkage `1/2`.
Use at least `4d` walkers. On near-isotropic targets with few walkers, an explicit
`shrinkage` such as `0.5` can accept more. Explicit real values keep their meaning:
a blend of the sample covariance with an isotropic covariance of the same trace.
Shrinkage `0` needs at least `2(d + 1)` walkers. A failed covariance fit leaves
that group unchanged without evaluating the target. Repeated failures can stall
the move.

Use Gaussian replacement for roughly elliptical targets. The independence proposal
has lighter tails than heavy-tailed targets and can miss curved regions, such as
a banana or a funnel. Mix it with `DEMove` or `StretchMove` for such targets. Check
tail estimates and independent runs, not acceptance or ESS alone. The default move
remains `StretchMove`.

## Batched log densities

Use [`BatchedLogDensity`](@ref) to evaluate candidates together, one per column:

```jldoctest batched
using EnsembleMCMC, Random123
using EnsembleMCMC: initialize, step!, sample!, current_state, snapshot

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
Initialization uses `scalar` unless cached `logdensities` are supplied.
Each nonempty group calls `batch!` once, omitting
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
using EnsembleMCMC: initialize, step!, sample!, current_state, snapshot

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
all built-in moves and their mixtures. Gaussian replacement also keeps its
complement fit on the device, returning only scalar fit checks to the host.

GPU sampling suits expensive, parallel batch targets. Small targets can be slower
because kernel launches and group synchronization dominate.

## Inputs and outputs

Initial coordinates must be finite and span their dimension. `initialize` warns if
the ensemble is nearly degenerate (smallest singular value ratio below `sqrt(eps)`).
Initial log densities may be `-Inf`, but `initialize` warns about them. Stretch, DE
and snooker move along lines through other walkers, so a start far outside the
support may never recover. Start inside the support when possible. Non-finite
initial values (NaN, `+Inf`) throw an error that names the walker.

Stretch requires at least `2d` walkers. DE and snooker require at
least `max(2d, 4)`. A target may return `-Inf` or `NaN` for proposals outside its
support. Both are rejected, as in Stan. A `+Inf` candidate throws an error that
names the walker and the sweep, and invalidates the state.

| Field returned by `sample!` | Meaning |
| --- | --- |
| `positions` | coordinate × walker × sweep |
| `logdensities` | walker × sweep |
| `accepted` | walker × sweep |
| `move_indices` | selected move per sweep |
| `walker_ids` | logical IDs in storage order |

Rejections repeat the current state. Returned arrays own their storage. Repeated
calls to [`sample!`](@ref) continue the same ensemble. Initialization copies the
input coordinates and RNG. Reusing an unchanged RNG produces the same run.

For independent chains, give each chain its own Random123 key, for example
`Philox4x((seed, chain))` for `chain = 1:4`. Reusing one RNG object for several
states gives identical chains.

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
