"""
    GaussianReplacementMove(; shrinkage=0.5)

Replace a walker with an independent Gaussian draw fitted to the frozen
complement, with the independence-proposal Hastings correction. The unbiased
covariance is shrunk toward `tr(C)/d * I` by finite `shrinkage` in `[0, 1]`.
Uses two groups and requires at least `max(2d, 4)` walkers, or `2(d+1)` when
shrinkage is zero. A non-positive or non-finite fit rejects the whole group.
"""
struct GaussianReplacementMove{S<:Real} <: AbstractEnsembleMove
    shrinkage::S

    function GaussianReplacementMove(shrinkage::S) where {S<:Real}
        isfinite(shrinkage) && zero(shrinkage) <= shrinkage <= one(shrinkage) ||
            throw(ArgumentError("Gaussian shrinkage must be finite and in [0, 1]"))
        return new{S}(shrinkage)
    end
end

GaussianReplacementMove(; shrinkage::Real=0.5) = GaussianReplacementMove(shrinkage)
group_count(::GaussianReplacementMove) = 2
minimum_walkers(move::GaussianReplacementMove, dimension::Integer) =
    iszero(move.shrinkage) ? 2 * (dimension + 1) : max(2 * dimension, 4)

function prepare_move(move::GaussianReplacementMove, ::Type{T}, ::Integer) where {T<:Real}
    return GaussianReplacementMove(convert(float(T), move.shrinkage))
end

struct _AllocatedGaussianMove{S,V,M,H} <: AbstractEnsembleMove
    shrinkage::S
    mean::V
    factor::M
    scratch::M
    host_scratch::H
end

struct _FittedGaussianMove{M,S} <: AbstractEnsembleMove
    move::M
    scale::S
    valid::Bool
end

group_count(::_AllocatedGaussianMove) = 2

function _allocate_move(move::GaussianReplacementMove, positions)
    d, n = length(first(positions)), length(positions)
    T = eltype(first(positions))
    return _AllocatedGaussianMove(move.shrinkage, Vector{T}(undef, d),
        Matrix{T}(undef, d, d), Matrix{T}(undef, d, n), nothing)
end

function _fit_gaussian!(mean, factor, scratch, n, shrinkage)
    T = eltype(mean)
    d = length(mean)
    centered = view(scratch, :, 1:n)
    scale = maximum(abs, centered)
    isfinite(scale) && scale > zero(T) || return scale, false
    anchor = view(factor, diagind(factor))
    @views @. anchor = centered[:, 1] / scale
    @. centered = centered / scale - anchor
    sum!(reshape(mean, :, 1), centered)
    mean ./= n
    centered .-= mean
    mean .+= anchor
    mul!(factor, centered, transpose(centered), inv(T(n - 1)), zero(T))
    isotropic = shrinkage * tr(factor) / d
    factor .*= one(T) - shrinkage
    view(factor, diagind(factor)) .+= isotropic
    all(isfinite, mean) && all(isfinite, factor) || return scale, false
    fit = cholesky!(Symmetric(factor, :L); check=false)
    return scale, isposdef(fit) && all(isfinite, factor)
end

function _prepare_gaussian_group!(workspace, move, state, indices)
    for (j, i) in enumerate(indices)
        copyto!(view(move.scratch, :, j), state.positions[i])
    end
    scale, valid = _fit_gaussian!(move.mean, move.factor, move.scratch,
        length(indices), move.shrinkage)
    return _FittedGaussianMove(move, scale, valid)
end

_prepare_group!(move::_AllocatedGaussianMove, state, group, complement) =
    _prepare_gaussian_group!(state.batch_workspace, move, state, only(complement))

function _gaussian_squared_distance!(scratch, position, move, scale)
    @. scratch = position / scale - move.mean
    ldiv!(LowerTriangular(move.factor), scratch)
    return sum(abs2, scratch)
end

function propose!(candidate, fitted::_FittedGaussianMove, current,
    walker_idx::Integer, complement_groups, rng::AbstractRNG,
    step_rngpart::RNGPartition, proposal_idx::Integer, walkerid::Integer)
    fitted.valid || return nothing
    move, scale = fitted.move, fitted.scale
    # Each walker owns a scratch column even when its proposal runs concurrently.
    scratch = view(move.scratch, :, walker_idx)
    set_rng!(rng, _walker_rngpart(step_rngpart, _SCALE_PURPOSE, proposal_idx), walkerid)
    randn!(rng, scratch)
    mul!(candidate, LowerTriangular(move.factor), scratch)
    @. candidate = scale * (move.mean + candidate)
    all(isfinite, candidate) || return nothing
    current_distance = _gaussian_squared_distance!(scratch, current[walker_idx], move, scale)
    # Use the rounded candidate, not the normal draw used to generate it.
    candidate_distance = _gaussian_squared_distance!(scratch, candidate, move, scale)
    return (candidate_distance - current_distance) / 2
end
