using Documenter
using DocumenterCodeBlocks
using EnsembleMCMC

DocMeta.setdocmeta!(EnsembleMCMC, :DocTestSetup, :(using EnsembleMCMC); recursive = true)

# Documenter discovers the shared logo in its source assets directory.
mkpath(joinpath(@__DIR__, "src", "assets"))
cp(joinpath(@__DIR__, "..", "assets", "logo.svg"),
   joinpath(@__DIR__, "src", "assets", "logo.svg"); force=true)

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
        assets = ["assets/logo.css"],
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
