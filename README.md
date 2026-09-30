# EnsembleMCMC.jl

[![CI](https://github.com/JuliaBayes/EnsembleMCMC.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaBayes/EnsembleMCMC.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/JuliaBayes/EnsembleMCMC.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaBayes/EnsembleMCMC.jl)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliabayes.org/EnsembleMCMC.jl/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE.md)

Ensemble MCMC with Stretch, differential-evolution, snooker, and Gaussian replacement moves.
Supports fixed move mixtures, threaded evaluation, and in-process continuable sampling.

Install the package and Random123. `initialize` takes a Random123 `Philox4x` or `Threefry4x` RNG:

```julia
using Pkg
Pkg.add(["EnsembleMCMC", "Random123"])
```

The generic verbs are not exported. The package is mainly used through adapters such as BAT.jl,
and names like `sample!` and `initialize` clash with StatsBase and AbstractMCMC. Import them
explicitly, or call them qualified (`EnsembleMCMC.sample!`). On Julia 1.11 and later they are `public`.

```julia
using EnsembleMCMC, Random, Random123
using EnsembleMCMC: initialize, step!, sample!, current_state, acceptance_rate

# Sample a two-dimensional standard normal target.
# Only the log density is needed, up to an additive constant.
logdensity(x) = -(x[1]^2 + x[2]^2) / 2

rng = Philox4x((42, 1))
nwalkers = 24
initial = randn(rng, 2, nwalkers)  # one walker per column
state = initialize(rng, logdensity, initial; move=StretchMove())

step!(state, 100)                 # discard 100 warmup sweeps
draws = sample!(state, 1_000)     # retain 1,000 further sweeps
size(draws.positions)            # (2, 24, 1000): coordinates × walkers × sweeps
draws.positions[:, 1, 1]         # first walker's first retained position
acceptance_rate(state)           # cumulative acceptance rate per move
```

Each sweep updates every walker. Walkers interact and are not independent chains.
For independent chains, give each chain its own Random123 key, for example `Philox4x((seed, chain))`.
Reusing one RNG object gives identical chains.

`sample!(state, n; thin=1)` stores every `thin`-th sweep. `snapshot(state)` is an in-process copy, not a checkpoint.

[Documentation](https://juliabayes.org/EnsembleMCMC.jl/) · [Apache 2.0 license](LICENSE.md)
