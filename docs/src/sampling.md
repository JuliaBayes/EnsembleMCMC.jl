# Sampling

```@meta
CurrentModule = EnsembleMCMC
```

## Choose starting points and walker counts

Put one walker in each column of a parameter-by-walker matrix. A vector of
parameter vectors also works. Use floating-point coordinates.

Choose distinct starting points spread across a plausible part of the target.
Every coordinate must be finite, and the points must span all parameter
directions. Identical points or points on a lower-dimensional plane cannot form
a valid ensemble. With exactly four walkers and only snooker moves, at least
one pair of opposite edges must have a nonzero dot product.

For a model with `d` parameters, the default Stretch move needs at least `2d`
walkers. The minimum depends on the move:

```jldoctest
using EnsembleMCMC

EnsembleMCMC.minimum_walkers(StretchMove(), 3)

# output

6
```

This is the smallest valid population, not a recommended sample size. More
walkers provide more proposal partners, but each sweep costs more target
evaluations. Compare the accuracy of the summaries you need at a fixed run time.

With two walkers in one dimension, Stretch keeps their initial order in exact
arithmetic. In the stationary law for that order, the two labels have the
marginals of the smaller and larger of two target draws. Their equally weighted
pool has the target marginal. Use at least three walkers to allow labels to pass.

Start inside the target's support. A walker far outside the support may never
recover because the moves use other walkers to construct proposals. The sampler
warns about initial log densities of `-Inf` and nearly degenerate ensembles.

The target may return `-Inf` for a proposed point outside its support. `NaN`
proposals are also rejected. An initial `NaN` or `+Inf`, or a proposed `+Inf`, is
an error.

## Choose a move

| Move | Proposal | Minimum walkers |
| --- | --- | --- |
| [`StretchMove`](@ref) | Stretch the distance to another walker. This is the default. | `2d` |
| [`DEMove`](@ref) | Add a scaled difference between two other walkers. | `max(2d, 4)` |
| [`DESnookerMove`](@ref) | Project a difference along a direction set by another walker. | `max(2d, 4)` |
| [`GaussianReplacementMove`](@ref) | Draw from a Gaussian fitted to the other group. | `max(2d, 4)`, or `2(d + 1)` with `shrinkage=0` |

Stretch and DE preserve their form under affine changes of coordinates. Snooker
does not have that property. None of these properties guarantees that a run
finds every mode of a distribution.

With integer-valued starting coordinates, `DEMove(gamma0=1, sigma=0)` can
stay on the integer grid because it adds integer differences. In one dimension,
`DESnookerMove(scale=1)` can also stay on the grid because projecting onto
the single axis leaves the difference unchanged. For a continuous target,
use jitter large enough to change the rounded coordinates, or include
`StretchMove()` with positive mixture weight. For example,
`DEMove(gamma0=1, sigma=0.01)` can leave a unit-spaced grid near zero.

### Mix moves

A mixture chooses one move for each complete sweep. With the default random
schedule, weights set fixed selection probabilities:

```jldoctest mixture
using EnsembleMCMC
using PureRNGs: Philox4x64, randn_next, subrng

rng = Philox4x64(42);
initial = first(randn_next(subrng(rng, 1), 2, 24));
move = MoveMixture((StretchMove(), DEMove()), [2, 1]);
draws = EnsembleMCMC.sample(rng, x -> -sum(abs2, x) / 2, initial, 10; move);
size(draws.positions)

# output

(2, 24, 10)
```

Here Stretch has probability `2/3` and DE has probability `1/3` at each sweep.
For `Float16`, `Float32`, and `Float64` weights, selection uses a `Float64`
uniform on a grid spaced by `2^-53`. Every positive component must have
probability at least `2^-53`. The constructor rejects smaller probabilities.

Use `schedule=:cycle` for a repeating sequence. Weights `[2, 1]` then mean Stretch,
Stretch, DE, repeated. A cyclic schedule requires integer weights. Zero-weight
moves never run and do not raise the minimum walker count. Weights do not adapt.

### Gaussian replacement

Gaussian replacement suits roughly elliptical targets. Its fitted Gaussian can
miss heavy tails or curved regions, so mix it with Stretch or DE for those
targets. Inspect tail estimates across separate runs.

The default `shrinkage=:auto` uses the covariance `(1 + d/m) C`, where `C` is
the sample covariance of the `m` walkers in the other group. This proposal is
affine invariant when the fit succeeds. If the group is too small or the fit
fails, it uses shrinkage `1/2` instead. That fallback is not affine invariant.

More walkers give the fit more points, at the cost of more target evaluations.
An explicit `shrinkage` blends the sample covariance with an isotropic covariance
of the same trace. Positive shrinkage preserves a common change of units.
Changing units on just one axis can change the proposal and its acceptance.
`shrinkage=0` uses the sample covariance unchanged. A fit that still fails
leaves the group unchanged. Repeated failures can stall the move.

The proposal uses the independence-proposal Hastings correction, following the
replacement move in [Goodman and Weare, equation (12)](https://msp.org/camcos/2010/5-1/camcos-v5-n1-p04-p.pdf).

The Hastings correction evaluates the continuous Gaussian density at the stored
coordinates. When the fitted spread covers only a few floating-point steps,
rounding changes the proposal probabilities. Near ten million, adjacent
`Float32` values are one unit apart. For a Gaussian with mean 10,000,000.25
and standard deviation 0.5, the draw rounds to 10,000,001 about 0.484 times as
often as to 10,000,000. The continuous density ratio is 0.368. For a narrow
distribution far from zero, express the model in centered coordinates before
converting to `Float32`, or store the walkers in `Float64`.

## Warmup and thinning

`warmup` is the number of sweeps to discard before collecting samples. Its
default is zero. It gives walkers time to move away from their starts, but it
does not tune proposals or establish convergence.

`thin=1` keeps every sweep. With `thin=k`, the sampler runs `k` sweeps for each
retained sample. Thinning reduces stored output, not target evaluations. It does
not make the retained samples independent.

## Checking a run

Inspect parameter traces, compare summaries from separate ensembles, and check
the quantities that matter for your analysis. High acceptance alone does not
show that the ensemble explored the target. Low acceptance or walkers that stop
moving can reveal a poor starting region or an unsuitable move.

Derive a separate key for each ensemble with `subrng`. Across runs, choose
starting points that span plausible regions of the target. Fresh starts in one
region can all miss the same mode.

For a target uniform on `[-2, -1]` and `[1, 2]`, the right interval has probability
one half. With the default stretch move, an ensemble confined to one interval
stays there. Four ensembles started on the left would all estimate zero.
Starting on both sides reveals the discrepancy.

```jldoctest independent
using EnsembleMCMC
using PureRNGs: Philox4x64, rand_next, subrng
using Statistics: mean

rng = Philox4x64(19);
logdensity(x) = 1 <= abs(only(x)) <= 2 ? 0.0 : -Inf;
initial_points(key, side) = side .* (1 .+ first(rand_next(key, Float64, 1, 24)));
runs = [EnsembleMCMC.sample(subrng(rng, i, 1), logdensity,
    initial_points(subrng(rng, i, 2), side), 200; warmup=100)
    for (i, side) in enumerate((-1, -1, 1, 1))];
positive_fraction(run) = vec(mean(run.positions[1, :, :] .> 0; dims=1));
Tuple(mean(positive_fraction(run)) for run in runs)

# output

(0.0, 0.0, 1.0, 1.0)
```

Here the pooled fraction is fixed by the number of ensembles started on each side.

Walkers within one ensemble are dependent. For an estimate using all walkers,
average the quantity of interest over walkers at each sweep, as above. Use the
time series from separate ensembles as independent chains for diagnostics, and
account for autocorrelation when estimating Monte Carlo error. A fixed-walker
diagnostic follows one trajectory from each ensemble.

## Reproducibility

Equal seeds and inputs reproduce a run with a deterministic target in the same
software and hardware environment. Both `sample` and `initialize` require a
PureRNGs key. Reusing it replays the same random streams
even if you advance its position between calls. Use `subrng(root, run_id)`
with distinct run IDs for independent ensembles. See
[Continuing a run](continuation.md).

Thread scheduling and `ThreadedExecutor`'s `min_chunk` do not change the random
streams assigned to walkers. Different executors or hardware can round arithmetic
differently and produce different trajectories. Bitwise replay across software
or hardware changes is not guaranteed.
