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
a valid ensemble.

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

### Mix moves

A mixture chooses one move for each complete sweep. With the default random
schedule, weights set fixed selection probabilities:

```jldoctest mixture
using EnsembleMCMC, Random

rng = Xoshiro(42);
initial = randn(rng, 2, 24);
move = MoveMixture((StretchMove(), DEMove()), [2, 1]);
draws = EnsembleMCMC.sample(rng, x -> -sum(abs2, x) / 2, initial, 10; move);
size(draws.positions)

# output

(2, 24, 10)
```

Here Stretch has probability `2/3` and DE has probability `1/3` at each sweep.
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
of the same trace. `shrinkage=0` uses the sample covariance unchanged. A fit that
still fails leaves the group unchanged. Repeated failures can stall the move.

The proposal uses the independence-proposal Hastings correction, following the
replacement move in [Goodman and Weare, equation (12)](https://msp.org/camcos/2010/5-1/camcos-v5-n1-p04-p.pdf).

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

Each call to `sample` advances the supplied RNG and starts a new ensemble. Use
fresh starting points for each run:

```jldoctest independent
using EnsembleMCMC, Random

rng = Xoshiro(19);
logdensity(x) = -sum(abs2, x) / 2;
runs = [EnsembleMCMC.sample(rng, logdensity, randn(rng, 2, 24), 200;
    warmup=100) for _ in 1:4];
length(runs)

# output

4
```

For a chain-based diagnostic, compare a fixed walker across separate ensembles.
For example, `[run.positions[:, 1, :] for run in runs]` contains one trajectory
from each ensemble. Walkers within one ensemble are dependent. Do not label them
as independent chains or add their effective sample sizes.

## Reproducibility

Equal seeds and inputs reproduce a run with a deterministic target in the same
software and hardware environment. Reusing an advancing RNG starts fresh runs.
The lower-level `initialize` method instead copies its RNG without advancing it.
See [Continuing a run](continuation.md).

Thread scheduling and `ThreadedExecutor`'s `min_chunk` do not change the random
streams assigned to walkers. Different executors or hardware can round arithmetic
differently and produce different trajectories. Bitwise replay across software
or hardware changes is not guaranteed.
