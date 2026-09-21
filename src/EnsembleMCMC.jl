module EnsembleMCMC

using LinearAlgebra
using Random
using Random123: Philox4x, Threefry4x
using PrecompileTools: @setup_workload, @compile_workload
import KernelAbstractions as KA
using KernelAbstractions: @index

export StretchMove, DEMove, DESnookerMove, MoveMixture
export GaussianReplacementMove
export SerialExecutor, ThreadedExecutor, initialize, step!, sample!, current_state, snapshot
export BatchedLogDensity
export KernelExecutor

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
