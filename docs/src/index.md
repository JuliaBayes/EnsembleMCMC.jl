# EnsembleMCMC.jl

EnsembleMCMC samples a probability distribution from its log density. It needs
no gradients. You provide the log density and starting points, then choose how
many samples to keep.

An **ensemble** is a set of **walkers**, each holding one point in parameter space.
A **sweep** updates every walker using the other walkers to propose moves.
Walkers interact, so they are not independent chains.

## Install

```julia
using Pkg
Pkg.add("EnsembleMCMC")
```

## Infer a mean from measurements

Suppose these measurements have a Normal distribution with an unknown mean
`mu` and a known standard deviation of `0.5`. Give `mu` a Normal prior with mean
zero and standard deviation `2`.

The log posterior is the log likelihood plus the log prior. Terms that do not
depend on `mu` can be omitted.

```@example inference
using EnsembleMCMC, Random, Statistics

measurements = [1.1, 1.9, 1.4, 0.9, 1.7]

function logdensity(x)
    mu = x[1]
    logprior = -mu^2 / (2 * 2.0^2)
    loglikelihood = -sum(y -> (y - mu)^2, measurements) / (2 * 0.5^2)
    return logprior + loglikelihood
end

rng = Xoshiro(42)
initial = 2 .* randn(rng, 1, 24)  # one parameter, 24 walkers
draws = EnsembleMCMC.sample(rng, logdensity, initial, 2_000; warmup=500)
nothing # hide
```

Each column of `initial` holds one walker's parameters. Here the starting points
come from the prior. The sampler discards `500` warmup sweeps, then keeps `2_000`
sweeps. Warmup does not tune the move or check convergence.

The default move is `StretchMove()`. Sampling functions use qualified names,
such as `EnsembleMCMC.sample`, to avoid clashes with other packages.

## Read the samples

```@example inference
size(draws.positions)
```

The axes are **parameter × walker × retained sweep**. In this example,
`draws.positions[1, 3, :]` is the sequence of values for `mu` from walker 3.
Rejected proposals repeat the walker's previous value.

Pool the retained values to estimate the posterior mean and a central credible
interval:

```@example inference
mu_samples = vec(draws.positions[1, :, :])
(
    posterior_mean=mean(mu_samples),
    credible_interval=quantile(mu_samples, [0.025, 0.975]),
)
```

The interval contains the central 95% of the sampled posterior. It describes
uncertainty about `mu` under the model. It is not an error bar on the Monte Carlo
estimate. The retained values are correlated.

This example has a known Normal posterior, so there is a direct check:

```@example inference
posterior_variance = inv(1 / 2.0^2 + length(measurements) / 0.5^2)
posterior_mean = posterior_variance * sum(measurements) / 0.5^2
(posterior_mean=posterior_mean, posterior_sd=sqrt(posterior_variance))
```

The sampled mean should be close to the analytic mean. The posterior standard
deviation is about `0.22`, so the interval should extend roughly `0.44` on either
side of the mean.

For a model without an analytic answer, compare separate ensembles and inspect
their traces and summaries. Pooling values is useful for posterior summaries,
but treating walkers as independent chains gives misleading convergence
diagnostics. See [Checking a run](sampling.md#Checking-a-run).

## Choose the next step

- [Sampling](sampling.md) explains starting points, walker counts, moves, warmup,
  thinning, and separate ensembles.
- [Continuing a run](continuation.md) covers longer runs, saved samples, and progress.
- [Threads and GPUs](parallel.md) covers parallel targets and CUDA arrays.
- [Writing an adapter](integration.md) covers cached densities, borrowed arrays,
  and external RNG addresses.
- [API](api.md) lists constructors, arguments, and return values.

EnsembleMCMC works with fixed-dimensional vectors of real parameters. Coordinate
transforms and model-specific starting points belong to the caller or a model
adapter. The package does not compute effective sample sizes or test convergence.

## License

EnsembleMCMC.jl uses the [Apache 2.0 license](https://github.com/JuliaBayes/EnsembleMCMC.jl/blob/main/LICENSE.md).
