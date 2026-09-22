# Changelog

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
