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

Generated HTML is in `docs/build`. Serve that directory with a local HTTP server
to view it. CI uploads the same build as a `documentation` artifact. Successful
tests and docs builds on `main` deploy it to
<https://juliabayes.org/EnsembleMCMC.jl/> through GitHub Pages.

CI checks Julia 1.10 and current stable Julia, serial and threaded execution,
Linux/macOS/Windows, and strict documentation builds. Each state represents a
coupled ensemble. Tests must preserve that statistical contract.

The current Julia/Linux job collects source coverage, retains `lcov.info` as a
`coverage` artifact, and uploads it to Codecov using GitHub OIDC. Coverage measures
executed lines, not statistical correctness. No coverage percentage target is set.

## Before a release

- Confirm every CI job passes on the exact release commit.
- Review the exported interface and document breaking changes in the experimental 0.0.x series.
- Update the changelog from unreleased to the chosen version and date.
- Confirm `Project.toml` matches that version.
- Confirm repository visibility and documentation hosting before registration.
- Obtain maintainer approval before tagging, publishing, or registering.

ESS and convergence checks are not implemented package features.
