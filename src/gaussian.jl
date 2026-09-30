"""
    GaussianReplacementMove(; shrinkage=:auto)

Replace a walker with an independent Gaussian draw fitted to the frozen
complement, with the independence-proposal Hastings correction.

With the default `shrinkage=:auto`, the proposal covariance is `(1 + d/m) * C`,
where `C` is the unbiased sample covariance of the `m` complement walkers. This
is affine invariant, so acceptance does not depend on the scales or the
correlations of the target. The factor `1 + d/m` widens the directions that a
small complement underestimates. If `m <= d` or `C` is not positive definite,
the move uses `shrinkage=1/2` for that group.

A finite `shrinkage` in `[0, 1]` shrinks `C` toward `tr(C)/d * I` instead. This
is not scale invariant: on targets with very different scales or strong
correlations, a positive value makes the proposal too wide and acceptance
collapses. `shrinkage=0` uses `C` unchanged.

Uses two groups and requires at least `max(2d, 4)` walkers, or `2(d+1)` when
`shrinkage=0`. Use at least `4d` walkers for good acceptance.

A non-finite or non-positive-definite fit rejects every proposal of the group.
The move does not warn. Acceptances that stay at zero (`current_state(state).acceptances`)
indicate a stalled move, for example after the ensemble collapses.

The Gaussian proposal has light tails. For heavy-tailed or curved targets, mix it
with `DEMove` or `StretchMove` in a `MoveMixture`.
"""
struct GaussianReplacementMove{S<:Union{Real,Symbol}} <: AbstractEnsembleMove
    shrinkage::S

    function GaussianReplacementMove(shrinkage::Symbol)
        shrinkage === :auto ||
            throw(ArgumentError("Gaussian shrinkage must be :auto or a finite value in [0, 1]"))
        return new{Symbol}(shrinkage)
    end
    function GaussianReplacementMove(shrinkage::S) where {S<:Real}
        isfinite(shrinkage) && zero(shrinkage) <= shrinkage <= one(shrinkage) ||
            throw(ArgumentError("Gaussian shrinkage must be :auto or a finite value in [0, 1]"))
        return new{S}(shrinkage)
    end
end

GaussianReplacementMove(; shrinkage::Union{Real,Symbol}=:auto) = GaussianReplacementMove(shrinkage)
group_count(::GaussianReplacementMove) = 2
minimum_walkers(move::GaussianReplacementMove, dimension::Integer) =
    move.shrinkage !== :auto && iszero(move.shrinkage) ? 2 * (dimension + 1) : max(2 * dimension, 4)

prepare_move(move::GaussianReplacementMove{Symbol}, ::Type{<:Real}, ::Integer) = move
function prepare_move(move::GaussianReplacementMove, ::Type{T}, ::Integer) where {T<:Real}
    return GaussianReplacementMove(convert(float(T), move.shrinkage))
end

struct _AllocatedGaussianMove{S,V,M,H} <: AbstractEnsembleMove
    shrinkage::S
    anchor::V
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
    return _AllocatedGaussianMove(move.shrinkage, Vector{T}(undef, d), Vector{T}(undef, d),
        Matrix{T}(undef, d, d), Matrix{T}(undef, d, n), nothing)
end

# The fit is `anchor + scale * N(mean, L L')` with `L` in the lower triangle of `factor`.
function _fit_gaussian!(anchor, mean, factor, scratch, n, shrinkage)
    T = eltype(mean)
    d = length(mean)
    centered = view(scratch, :, 1:n)
    # Subtract before scaling: `x / scale` rounds at ulp(x), far from the origin.
    anchor .= view(centered, :, 1)
    centered .-= anchor
    scale = maximum(abs, centered)
    isfinite(scale) && scale > zero(T) || return scale, false
    centered ./= scale
    sum!(reshape(mean, :, 1), centered)
    mean ./= n
    centered .-= mean
    all(isfinite, mean) || return scale, false
    if shrinkage isa Symbol
        n > d && _gaussian_cholesky!(factor, centered, n, 1 + T(d) / n) && return scale, true
    end
    s = _explicit_shrinkage(shrinkage, T)
    _gaussian_covariance!(factor, centered, n, one(T))
    isotropic = s * tr(factor) / d
    factor .*= one(T) - s
    view(factor, diagind(factor)) .+= isotropic
    return scale, _gaussian_cholesky!(factor)
end

_explicit_shrinkage(::Symbol, ::Type{T}) where {T} = inv(T(2))
_explicit_shrinkage(shrinkage::Real, ::Type) = shrinkage

_gaussian_covariance!(factor, centered, n, inflation) =
    mul!(factor, centered, transpose(centered), inflation / (n - 1), zero(eltype(factor)))

function _gaussian_cholesky!(factor, centered, n, inflation)
    _gaussian_covariance!(factor, centered, n, inflation)
    return _gaussian_cholesky!(factor)
end

function _gaussian_cholesky!(factor)
    all(isfinite, factor) || return false
    fit = cholesky!(Symmetric(factor, :L); check=false)
    return isposdef(fit) && all(isfinite, factor)
end

function _prepare_gaussian_group!(workspace, move, state, indices)
    for (j, i) in enumerate(indices)
        copyto!(view(move.scratch, :, j), state.positions[i])
    end
    scale, valid = _fit_gaussian!(move.anchor, move.mean, move.factor, move.scratch,
        length(indices), move.shrinkage)
    return _FittedGaussianMove(move, scale, valid)
end

_prepare_group!(move::_AllocatedGaussianMove, state, group, complement) =
    _prepare_gaussian_group!(state.batch_workspace, move, state, only(complement))

_gaussian_ldiv!(factor, scratch) = ldiv!(LowerTriangular(factor), scratch)
function _gaussian_ldiv!(factor::StridedMatrix{T}, scratch::StridedVector{T}) where {T<:Union{Float32,Float64}}
    length(scratch) > 128 && return ldiv!(LowerTriangular(factor), scratch)
    # Small forward solves avoid BLAS call overhead and visit contiguous columns.
    @inbounds for j in eachindex(scratch)
        x = scratch[j] / factor[j, j]
        scratch[j] = x
        @simd for i in (j + 1):length(scratch)
            scratch[i] -= factor[i, j] * x
        end
    end
    return scratch
end

function _gaussian_squared_distance!(scratch, position, move, scale)
    @. scratch = (position - move.anchor) / scale - move.mean
    _gaussian_ldiv!(move.factor, scratch)
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
    @. candidate = move.anchor + scale * (move.mean + candidate)
    all(isfinite, candidate) || return nothing
    current_distance = _gaussian_squared_distance!(scratch, current[walker_idx], move, scale)
    # Use the rounded candidate, not the normal draw used to generate it.
    candidate_distance = _gaussian_squared_distance!(scratch, candidate, move, scale)
    return (candidate_distance - current_distance) / 2
end
