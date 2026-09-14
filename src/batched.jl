"""
    BatchedLogDensity(scalar, batch!)

Wrap scalar and batched log-density callbacks. Initialization uses `scalar(x)`
unless cached `logdensities` are supplied.
During sampling, `batch!(values, positions)` receives borrowed `SubArray` views
of output storage and a coordinate-by-candidate matrix containing only valid
proposals. The callback must treat `positions` as read-only, fill every entry of
`values` with the same result as `scalar`, and must not retain either array. It
is called once per nonempty frozen proposal group and controls any parallelism
used to evaluate that group.
"""
struct BatchedLogDensity{F,B}
    scalar::F
    batch!::B
end

(target::BatchedLogDensity)(x) = target.scalar(x)

struct BatchWorkspace{T,L}
    positions::Matrix{T}
    values::Vector{L}
    log_hastings::Vector{T}
end

_batch_workspace(logdensity, positions, logdensities) = nothing
function _batch_workspace(::BatchedLogDensity, positions, logdensities)
    n = length(positions)
    d = length(first(positions))
    return BatchWorkspace(
        Matrix{eltype(first(positions))}(undef, d, n),
        Vector{eltype(logdensities)}(undef, n),
        Vector{eltype(first(positions))}(undef, n),
    )
end
