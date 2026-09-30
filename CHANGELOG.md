# Changelog

## Unreleased

### Breaking

- Stop exporting the generic verbs `initialize`, `step!`, `sample!`, `current_state`, `snapshot`, `synchronize!` and `validate_positions`. Use `using EnsembleMCMC: ...` or qualified calls. They are `public` on Julia 1.11 and later. The types stay exported.
- A NaN candidate log density is now a rejection, as in Stan. A `+Inf` candidate and a non-finite initial value still throw.
- `GaussianReplacementMove` defaults to `shrinkage=:auto`, which changes its results. The proposal covariance is `(1 + d/m) C` from the `m` complement walkers. It is affine invariant. It falls back to shrinkage `1/2` when `m <= d` or the fit fails.
- Threefry4x streams changed: `set_rng!` now resets the buffer position, so every Threefry4x trajectory differs from 0.0.2. Philox4x trajectories are unchanged, except for `GaussianReplacementMove` (new default).
- `ThreadedExecutor` no longer calibrates or falls back to serial execution (8eca227).

### Added

- `acceptance_rate(state)`.
- `sample!(state, n; thin=1)`.
- `ThreadedExecutor(; min_chunk=1)` limits the task count for cheap targets.
- Warnings for `-Inf` initial log densities and for nearly degenerate ensembles.
- BigFloat positions.
- `KernelExecutor` accepts wrapped CuArrays (view, transpose, adjoint, `PermutedDimsArray`).
- Aqua tests, API tests, a banana validation test, and more Gaussian tests.
- Precompile workload for threaded runs and mixtures.

### Changed

- Errors name the walker and the sweep.
- `ArgumentError` for an unsupported RNG, an invalid move, unknown keywords, and an RNG partition depth too high for `initialize` and `step!(state, rng)`.
- `ThreadedExecutor` rethrows the original target exception, not a `CompositeException`.
- `MoveMixture` accepts tuple weights.
- Docs: install `Random123`, independent chains, and executor reproducibility.

### Fixed

- Float32 Gaussian fit precision for ensembles far from the origin.
- `KernelExecutor` on wrapped CuArrays raised `MethodError`.
- Docstrings of `initialize` and `validate_positions`.

### Performance

- Allocation per sweep drops from 21 KB to 1.9 KB (Stretch, 100 walkers). A sweep takes 8.3 µs instead of 15.0 µs.
- Kernel bookkeeping runs in parallel, with fewer host syncs per sweep. Stretch, DE and snooker are 1.3-1.7x faster on an A100.
- The Gaussian reject kernel runs in parallel.

## 0.0.2 — 2026-09-22

- Add Gaussian replacement moves and device-resident sampling through KernelAbstractions and the CUDA extension.
- Add cached initial densities, transition details, state synchronization, and externally addressed sweeps for sampler adapters.
- Improve proposal throughput and workload-aware thread scheduling, and precompile common sampling workloads.
- Preserve rejected-transition metadata for degenerate proposals and run synchronization on the state's owning CUDA device.

Breaking: `current_state` and `snapshot` now include candidate positions, candidate log densities, and acceptance probabilities. Code that assumes the old record shape must change.

## 0.0.1

- Add Stretch, DE, and snooker ensemble moves.
- Add fixed mixtures, sequential/threaded sweeps, and resumable sample collection.
- Preserve logical walker RNG streams and retain independent sample storage.
- Add contract tests, a Documenter guide and API reference, and CI.
- License the package under Apache 2.0.

The initial interface is experimental. Diagnostics remain separate work.
