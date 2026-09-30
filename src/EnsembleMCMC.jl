module EnsembleMCMC

using LinearAlgebra
using Random
using Random123: Philox4x, Threefry4x
using PrecompileTools: @setup_workload, @compile_workload
import KernelAbstractions as KA
using KernelAbstractions: @index

export StretchMove, DEMove, DESnookerMove, MoveMixture
export GaussianReplacementMove
export SerialExecutor, ThreadedExecutor, KernelExecutor
export BatchedLogDensity

# The generic verbs stay unexported so they do not clash with adapter packages.
@static if VERSION >= v"1.11.0-DEV.469"
    eval(Expr(:public, :initialize, :step!, :sample!, :current_state, :snapshot,
        :synchronize!, :validate_positions, :acceptance_rate))
end

include("rng.jl")
include("moves.jl")
include("batched.jl")
include("sampling.jl")
include("kernels.jl")
include("backend.jl")
include("gaussian.jl")
include("gaussian_backend.jl")
include("precompile.jl")

end
