"""
    BatchedLogDensity(scalar, batch!; batch_size=nothing)

Wrap scalar and batched log-density callbacks. Initialization uses `scalar(x)`
unless cached `logdensities` are supplied.
During sampling, `batch!(values, positions)` receives borrowed `SubArray` views
of output storage and a coordinate-by-candidate matrix containing only valid
proposals. The callback must treat `positions` as read-only, fill every entry of
`values` with the same result as `scalar`, and must not retain either array. It
is called once per nonempty frozen proposal group and controls any parallelism
used to evaluate that group.

Set `batch_size` for a callback compiled for a fixed number of columns. It must
fit the largest active proposal group. The callback then borrows the full sampler-owned
buffers, padded with copies of the first valid proposal. Padding does not
contribute to the Markov chain. Empty groups still skip the callback.

With [`KernelExecutor`](@ref), invalid outputs (`+Inf` or unwritten entries) are
reported at sweep completion, so later group callbacks may run. A thrown callback
stops the sweep immediately. Either failure invalidates the state.
"""
struct BatchedLogDensity{F,B,S}
    scalar::F
    batch!::B
    batch_size::S
end

function BatchedLogDensity(scalar, batch!; batch_size::Union{Nothing,Integer}=nothing)
    isnothing(batch_size) || batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    return BatchedLogDensity(scalar, batch!, isnothing(batch_size) ? nothing : Int(batch_size))
end

(target::BatchedLogDensity)(x) = target.scalar(x)

struct BatchWorkspace{T,L}
    positions::Matrix{T}
    values::Vector{L}
    log_hastings::Vector{T}
end

function _batch_capacity(target::BatchedLogDensity, n, moves, weights)
    capacity = something(target.batch_size, n)
    required = maximum(cld(n, group_count(moves[i])) for i in eachindex(moves)
        if isnothing(weights) || weights[i] > 0)
    capacity >= required ||
        throw(ArgumentError("batch_size must fit the largest proposal group"))
    return capacity
end

_batch_workspace(logdensity, positions, logdensities, moves, weights) = nothing
function _batch_workspace(target::BatchedLogDensity, positions, logdensities, moves, weights)
    n = length(positions)
    capacity = _batch_capacity(target, n, moves, weights)
    d = length(first(positions))
    return BatchWorkspace(
        Matrix{eltype(first(positions))}(undef, d, capacity),
        Vector{eltype(logdensities)}(undef, capacity),
        Vector{eltype(first(positions))}(undef, n),
    )
end
