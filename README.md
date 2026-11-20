# EnsembleMCMC.jl

[![CI](https://github.com/JuliaBayes/EnsembleMCMC.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaBayes/EnsembleMCMC.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/JuliaBayes/EnsembleMCMC.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaBayes/EnsembleMCMC.jl)
[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliabayes.org/EnsembleMCMC.jl/stable/) <a href="https://juliabayes.org/EnsembleMCMC.jl/stable/"><img src="assets/logo.svg" align="right" alt="EnsembleMCMC.jl logo" width="128" height="128"></a>
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE.md)

Sample a log density with an ensemble of walkers. EnsembleMCMC provides Stretch,
differential-evolution, snooker, and Gaussian replacement moves, with threads and
CUDA support. It needs no gradients.

For the stable guide, run `import Pkg` followed by
`Pkg.add(["EnsembleMCMC", "PureRNGs"])`.

This example follows `dev`.

```julia
using Pkg
Pkg.add(url="https://github.com/JuliaBayes/EnsembleMCMC.jl", rev="dev")
Pkg.add("PureRNGs")
```

```julia
using EnsembleMCMC
using PureRNGs: Philox4x64, randn_next, subrng

# Sample a two-dimensional standard normal target.
# Only the log density is needed, up to an additive constant.
logdensity(x) = -(x[1]^2 + x[2]^2) / 2

rng = Philox4x64(42)
nwalkers = 24
initial = first(randn_next(subrng(rng, 1), 2, nwalkers))  # one walker per column
draws = EnsembleMCMC.sample(rng, logdensity, initial, 1_000; warmup=100)
size(draws.positions)            # (2, 24, 1000): coordinates × walkers × sweeps
draws.positions[:, 1, 1]         # first walker's first retained position
```

Each sweep updates every walker. Walkers interact and are not independent chains.
Reuse the PureRNGs key to replay the ensemble in the same environment.
Choose warmup and run length for the target. Sampling does not test convergence.

Use `initialize`, `step!`, and `sample!` to continue a run or store samples in blocks.
The [development guide](https://juliabayes.org/EnsembleMCMC.jl/dev/) covers a complete
inference example, starting points, move choice, and interpreting results.

[Stable docs](https://juliabayes.org/EnsembleMCMC.jl/stable/) ·
[Development docs](https://juliabayes.org/EnsembleMCMC.jl/dev/) ·
[Apache 2.0 license](LICENSE.md)
