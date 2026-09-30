# API

```@meta
CurrentModule = EnsembleMCMC
```

These functions are not exported. Import them with `using EnsembleMCMC: initialize,
step!, sample!, current_state, snapshot, synchronize!, validate_positions,
acceptance_rate`, or call them qualified. On Julia 1.11 and later they are `public`.

## Sampling

```@docs
initialize
step!
sample!
current_state
snapshot
synchronize!
validate_positions
acceptance_rate
```

## Moves

```@docs
StretchMove
DEMove
DESnookerMove
GaussianReplacementMove
MoveMixture
```

## Execution

```@docs
BatchedLogDensity
SerialExecutor
ThreadedExecutor
KernelExecutor
```
