module EnsembleMCMCReactantExt

import EnsembleMCMC
import Reactant
using LinearAlgebra: diag, tril

function EnsembleMCMC._check_kernel_array(
    initial::Reactant.ConcretePJRTArray{T,N,1},
) where {T,N}
    Reactant.XLA.platform_name(Reactant.XLA.client(initial)) == "cuda" ||
        throw(ArgumentError("KernelExecutor requires a CUDA device for Reactant arrays"))
    return nothing
end

_owner(array::Reactant.ConcretePJRTArray) = array
_owner(array) = _owner(parent(array))
_wait(array) = wait(_owner(array))

function EnsembleMCMC._kernel_synchronize(array::Reactant.AnyConcretePJRTArray, arrays...)
    _wait(array)
    foreach(_wait, arrays)
    return nothing
end

function EnsembleMCMC._kernel_fill!(array::Reactant.ConcretePJRTArray, value)
    compiled = Reactant.Compiler.compile(
        fill!,
        (array, value);
        client = Reactant.XLA.client(array),
    )
    return compiled(array, value)
end

_copyto!(dest, src) = (dest .= reshape(src, size(dest)))

function EnsembleMCMC._kernel_copyto!(dest::Reactant.AnyConcretePJRTArray, src)
    source = Array{eltype(dest)}(src)
    compiled = Reactant.Compiler.compile(
        _copyto!,
        (dest, source);
        client = Reactant.XLA.client(_owner(dest)),
    )
    return compiled(dest, source)
end

function EnsembleMCMC._kernel_copyto!(
    dest::Reactant.AnyConcretePJRTArray,
    src::Reactant.AnyConcretePJRTArray,
)
    compiled = Reactant.Compiler.compile(
        _copyto!,
        (dest, src);
        client = Reactant.XLA.client(_owner(dest)),
    )
    return compiled(dest, src)
end

_all_converted(f, ::Type{T}, array) where {T} =
    all(f, convert(Reactant.TracedRArray{T,ndims(array)}, array))

function EnsembleMCMC._kernel_all(
    f,
    ::Type{T},
    array::Reactant.AnyConcretePJRTArray,
) where {T}
    compiled = Reactant.Compiler.compile(
        _all_converted,
        (f, T, array);
        client = Reactant.XLA.client(_owner(array)),
    )
    return Bool(compiled(f, T, array))
end

struct ReactantExecution{C,D}
    client::C
    device::D
    programs::Dict{Any,Any}
    history::Base.RefValue{Any}
end

function EnsembleMCMC._kernel_execution(initial::Reactant.ConcretePJRTArray)
    return ReactantExecution(
        Reactant.XLA.client(initial),
        Reactant.XLA.device(initial),
        Dict{Any,Any}(),
        Ref{Any}(nothing),
    )
end

# Only static metadata enters the key. Array values change at every sweep.
_signature(x) = (typeof(x), x)
_signature(x::Reactant.ConcretePJRTArray) = (typeof(x), size(x))
_signature(x::Reactant.ConcretePJRTNumber) = typeof(x)
_signature(x::SubArray) =
    (typeof(x), _signature(parent(x)), map(_signature, parentindices(x)))
_signature(x::AbstractRange) = (typeof(x), first(x), step(x), length(x))
_signature(x::Base.Slice) = (typeof(x), _signature(x.indices))

function EnsembleMCMC._kernel_execute(execution::ReactantExecution, f, args...)
    key = (f, map(_signature, args))
    compiled = get!(execution.programs, key) do
        Reactant.Compiler.compile(f, args; client = execution.client)
    end
    return compiled(args...)
end

# A state can retain histories of different lengths. Keep only the latest program.
function EnsembleMCMC._kernel_execute(
    execution::ReactantExecution,
    f::typeof(EnsembleMCMC._copy_kernel_history!),
    args...,
)
    signature = map(_signature, args)
    cached = execution.history[]
    if isnothing(cached) || first(cached) != signature
        cached = (signature, Reactant.Compiler.compile(f, args; client = execution.client))
        execution.history[] = cached
    end
    return last(cached)(args...)
end

EnsembleMCMC._kernel_scalar(execution::ReactantExecution, value) =
    Reactant.ConcretePJRTNumber(value; execution.client, execution.device)
EnsembleMCMC._host_scalar(value::Reactant.ConcretePJRTNumber{T}) where {T} = T(value)

EnsembleMCMC._gaussian_trace(factor::Reactant.TracedRArray) = sum(diag(factor))

function EnsembleMCMC._gaussian_covariance!(
    factor::Reactant.TracedRArray,
    centered,
    n,
    inflation,
)
    # Reactant's mul! evaluates beta * factor even for beta=0. A failed previous
    # factorization can leave NaNs, which must not contaminate the next covariance.
    factor .= (inflation / (n - 1)) .* (centered * transpose(centered))
    return factor
end

function EnsembleMCMC._gaussian_cholesky!(factor::Reactant.TracedRArray)
    # Reactant's cholesky! wrapper allocates upper factors. The move needs an
    # in-place lower factor and a validity flag, which this device operation supplies.
    finite = all(isfinite, factor)
    factor .= tril(Reactant.Ops.cholesky(factor; lower = true))
    return finite & all(isfinite, factor) & all(>(0), diag(factor))
end

end
