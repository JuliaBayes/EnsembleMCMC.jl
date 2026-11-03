using Documenter
using DocumenterCodeBlocks
using EnsembleMCMC

DocMeta.setdocmeta!(EnsembleMCMC, :DocTestSetup, :(using EnsembleMCMC); recursive = true)

makedocs(
    root = @__DIR__,
    sitename = "EnsembleMCMC.jl",
    repo = Documenter.Remotes.GitHub("JuliaBayes", "EnsembleMCMC.jl"),
    modules = [EnsembleMCMC],
    checkdocs = :public,
    doctest = true,
    warnonly = false,
    format = Documenter.HTML(
        prettyurls = true,
        edit_link = "dev",
        canonical = "https://juliabayes.org/EnsembleMCMC.jl/stable/",
    ),
    pages = [
        "Getting started" => "index.md",
        "Guides" => [
            "Sampling" => "sampling.md",
            "Continuing a run" => "continuation.md",
            "Threads and GPUs" => "parallel.md",
        ],
        "Reference" => ["Writing an adapter" => "integration.md", "API" => "api.md"],
    ],
    plugins = [CodeBlocks()],
)
