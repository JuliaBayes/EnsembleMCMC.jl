# Development

Run package tests from the repository root:

```sh
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Build the docs and run doctests:

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

`docs/make.jl` runs doctests (`doctest = true`) and fails on any warning. The docs CI job
therefore runs the doctests. They are not part of `Pkg.test()`, which has no Documenter dependency.

Generated HTML is in `docs/build`. Serve that directory with a local HTTP server
to view it. CI uploads the same build as a `documentation` artifact. Successful
tests and docs builds on `main` deploy it to
<https://juliabayes.org/EnsembleMCMC.jl/> through GitHub Pages.

CI checks Julia 1.10 (1 and 4 threads), current stable Julia, a pre-release Julia
(allowed to fail), serial and threaded execution, Linux/macOS/Windows, and strict
documentation builds. CI has no GPU job. Each state represents a
coupled ensemble. Tests must preserve that statistical contract.

The coverage job (current Julia, Linux) collects source coverage, retains `lcov.info` as a
`coverage` artifact, and uploads it to Codecov using GitHub OIDC. Coverage measures
executed lines, not statistical correctness. No coverage percentage target is set.

## GPU tests

`Pkg.test()` does not run the CUDA tests. Run them on a machine with a CUDA GPU,
from the repository root:

```sh
julia -e 'using Pkg; Pkg.activate(temp=true); Pkg.develop(path="."); Pkg.add(["CUDA", "Random123", "Test"]); include("test/cuda.jl")'
```

The tests use the current device. Call `CUDA.device!(i)` before the include to
choose another one. The file exits without a functional GPU. The multi-device block
is skipped with an `@info` if no second device can open a context.

## Before a release

- Confirm every CI job passes on the exact release commit.
- Review the exported interface and document breaking changes in the experimental 0.0.x series.
- Update the changelog from unreleased to the chosen version and date.
- Confirm `Project.toml` matches that version.
- Confirm repository visibility and documentation hosting before registration.
- Obtain maintainer approval before tagging, publishing, or registering.

ESS and convergence checks are not implemented package features.
