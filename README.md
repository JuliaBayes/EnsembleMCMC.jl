# EnsembleMCMC.jl

[![CI](https://github.com/JuliaBayes/EnsembleMCMC.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaBayes/EnsembleMCMC.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/JuliaBayes/EnsembleMCMC.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaBayes/EnsembleMCMC.jl)
[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliabayes.org/EnsembleMCMC.jl/stable/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE.md)

Sample a log density with an ensemble of walkers. EnsembleMCMC provides Stretch,
differential-evolution, snooker, and Gaussian replacement moves, with threads and
CUDA support. It needs no gradients.

Install the package with `using Pkg; Pkg.add("EnsembleMCMC")`.

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

Each sweep updates every walker. Walkers interact and are not independent chains.
Each call to `sample` advances the supplied RNG to seed a new ensemble.
Choose warmup and run length for the target. Sampling does not test convergence.

Use `initialize`, `step!`, and `sample!` to continue a run or store samples in blocks.
The [guide](https://juliabayes.org/EnsembleMCMC.jl/stable/) covers a complete inference
example, starting points, move choice, and interpreting results.

[Stable docs](https://juliabayes.org/EnsembleMCMC.jl/stable/) ·
[Development docs](https://juliabayes.org/EnsembleMCMC.jl/dev/) ·
[Apache 2.0 license](LICENSE.md)
