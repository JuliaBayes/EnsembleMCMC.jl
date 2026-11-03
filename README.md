# EnsembleMCMC.jl

[![CI](https://github.com/JuliaBayes/EnsembleMCMC.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaBayes/EnsembleMCMC.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/JuliaBayes/EnsembleMCMC.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaBayes/EnsembleMCMC.jl)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliabayes.org/EnsembleMCMC.jl/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE.md)

Ensemble MCMC with Stretch, differential-evolution, snooker, and Gaussian replacement moves.
Supports fixed move mixtures, threaded evaluation, and in-process continuable sampling.

```julia
using EnsembleMCMC, Random

# Sample a two-dimensional standard normal target.
# Only the log density is needed, up to an additive constant.
logdensity(x) = -(x[1]^2 + x[2]^2) / 2

rng = Xoshiro(42)
nwalkers = 24
initial = randn(rng, 2, nwalkers)  # one walker per column
draws = EnsembleMCMC.sample(rng, logdensity, initial, 1_000; warmup=100)
size(draws.positions)            # (2, 24, 1000): coordinates × walkers × sweeps
draws.positions[:, 1, 1]         # first walker's first retained position
```

This one-call API is on `dev` for 0.1.0. Install it with
`Pkg.add(url="https://github.com/JuliaBayes/EnsembleMCMC.jl", rev="dev")`.

Each sweep updates every walker. Walkers interact and are not independent chains.
Each call to `sample` advances the supplied RNG to seed a new ensemble.
Choose warmup and run length for the target; sampling does not test convergence.

For in-process continuation, use `initialize`, `step!` and `sample!`. `initialize`
requires and copies a Random123 `Philox4x` or `Threefry4x` RNG without advancing it.
Use a distinct RNG key for each independent state. `sample!(state, n; thin=1)`
stores every `thin`-th sweep. `snapshot(state)` is an in-process copy, not a checkpoint.

[Documentation](https://juliabayes.org/EnsembleMCMC.jl/) · [Apache 2.0 license](LICENSE.md)
