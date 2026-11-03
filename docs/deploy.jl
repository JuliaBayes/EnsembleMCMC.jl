using Documenter

deploydocs(
    root = @__DIR__,
    repo = "github.com/JuliaBayes/EnsembleMCMC.jl.git",
    devbranch = "dev",
    versions = ["stable" => "v^", "v#.#.#", "dev" => "dev"],
)
