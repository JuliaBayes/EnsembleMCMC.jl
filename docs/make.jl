using Documenter
using EnsembleMCMC

DocMeta.setdocmeta!(EnsembleMCMC, :DocTestSetup, :(using EnsembleMCMC); recursive=true)

makedocs(
    root = @__DIR__,
    sitename = "EnsembleMCMC.jl",
    repo = Documenter.Remotes.GitHub("JuliaBayes", "EnsembleMCMC.jl"),
    modules = [EnsembleMCMC],
    checkdocs = :public,
    doctest = true,
    warnonly = false,
    format = Documenter.HTML(
        prettyurls=true,
        edit_link="main",
        canonical="https://juliabayes.org/EnsembleMCMC.jl/",
    ),
    pages = ["Getting started" => "index.md", "API" => "api.md"],
)
