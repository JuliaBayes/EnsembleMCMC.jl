# API

```@meta
CurrentModule = EnsembleMCMC
```

Call sampling functions with qualified names, such as `EnsembleMCMC.sample`.
Move and executor types are exported. Individual functions can also be imported,
for example with `using EnsembleMCMC: initialize, step!, sample!`.

## Sampling

```@docs
sample
initialize
step!
sample!
current_state
snapshot
acceptance_rate
```

## Moves

```@docs
StretchMove
DEMove
DESnookerMove
GaussianReplacementMove
MoveMixture
minimum_walkers
```

## Execution

```@docs
BatchedLogDensity
SerialExecutor
ThreadedExecutor
KernelExecutor
```

## Adapter support

```@docs
synchronize!
validate_positions
```
