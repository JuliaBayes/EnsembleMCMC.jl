# API

```@meta
CurrentModule = EnsembleMCMC
```

## Sampling

```@docs
initialize
step!
sample!
current_state
snapshot
synchronize!
validate_positions
```

## Moves

```@docs
StretchMove
DEMove
DESnookerMove
MoveMixture
```

## Execution

```@docs
BatchedLogDensity
SerialExecutor
ThreadedExecutor
KernelExecutor
```
