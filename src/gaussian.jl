"""
    GaussianReplacementMove(; shrinkage=:auto)

Replace a walker with an independent Gaussian draw fitted to the frozen
complement, with the independence-proposal Hastings correction.

With the default `shrinkage=:auto`, the proposal covariance is `(1 + d/m) * C`,
where `C` is the unbiased sample covariance of the `m` complement walkers. This
is affine invariant when the fit succeeds. The factor `1 + d/m` widens the
directions that a small complement underestimates. If `m <= d` or `C` is not
positive definite, the move uses `shrinkage=1/2` for that group. This isotropic
fallback is not affine invariant.

A finite `shrinkage` in `[0, 1]` shrinks `C` toward `tr(C)/d * I` instead. This
is not scale invariant: on targets with very different scales or strong
correlations, a positive value makes the proposal too wide and acceptance
collapses. `shrinkage=0` uses `C` unchanged.

Uses two groups and requires at least `max(2d, 4)` walkers, or `2(d+1)` when
`shrinkage=0`. More walkers give the covariance fit more points. Acceptance
still depends on the target.

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
_minimum_walkers(move::GaussianReplacementMove, dimension::Integer) =
    move.shrinkage !== :auto && iszero(move.shrinkage) ?
    Base.checked_mul(2, Base.checked_add(dimension, 1)) : max(Base.checked_mul(2, dimension), 4)

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
    host::H
end

struct _FittedGaussianMove{M,S} <: AbstractEnsembleMove
    move::M
    scale::S
    valid::Bool
end

group_count(::_AllocatedGaussianMove) = 2
_companion_lanes(::_AllocatedGaussianMove) = 0
_scale_width(::_AllocatedGaussianMove, d) = d

function _allocate_move(move::GaussianReplacementMove, positions)
    d, n = length(first(positions)), length(positions)
    T = eltype(first(positions))
    return _AllocatedGaussianMove(move.shrinkage, Vector{T}(undef, d), Vector{T}(undef, d),
        Matrix{T}(undef, d, d), Matrix{T}(undef, d, n), (; log_hastings=Vector{T}(undef, n)))
end

# The fit is `anchor + scale * N(mean, L L')` with `L` in the lower triangle of `factor`.
function _fit_gaussian!(anchor, mean, factor, scratch, n, shrinkage, execution=nothing)
    T = eltype(mean)
    d = length(mean)
    centered = view(scratch, :, 1:n)
    scale = _kernel_execute(execution, _center_gaussian!, anchor, centered)
    host_scale = _host_scalar(scale)
    isfinite(host_scale) && host_scale > zero(T) || return scale, false
    valid = _kernel_execute(execution, _standardize_gaussian!, mean, centered, scale, n)
    _host_scalar(valid) || return scale, false
    if shrinkage isa Symbol
        n > d && _gaussian_cholesky!(factor, centered, n, 1 + T(d) / n, execution) && return scale, true
    end
    s = _explicit_shrinkage(shrinkage, T)
    _kernel_execute(execution, _gaussian_covariance!, factor, centered, n, one(T))
    _kernel_execute(execution, _shrink_gaussian!, factor, s)
    return scale, _host_scalar(_kernel_execute(execution, _gaussian_cholesky!, factor))
end

function _center_gaussian!(anchor, centered)
    # Subtract before scaling: `x / scale` rounds at ulp(x), far from the origin.
    anchor .= view(centered, :, 1)
    centered .-= anchor
    return maximum(abs, centered)
end

function _standardize_gaussian!(mean, centered, scale, n)
    centered ./= scale
    sum!(reshape(mean, :, 1), centered)
    mean ./= n
    centered .-= mean
    return all(isfinite, mean)
end

_gaussian_trace(factor) = tr(factor)

function _shrink_gaussian!(factor, shrinkage)
    isotropic = shrinkage * _gaussian_trace(factor) / size(factor, 1)
    factor .*= one(eltype(factor)) - shrinkage
    view(factor, diagind(factor)) .+= isotropic
    return nothing
end

_explicit_shrinkage(::Symbol, ::Type{T}) where {T} = inv(T(2))
_explicit_shrinkage(shrinkage::Real, ::Type) = shrinkage

_gaussian_covariance!(factor, centered, n, inflation) =
    mul!(factor, centered, transpose(centered), inflation / (n - 1), zero(eltype(factor)))

function _gaussian_cholesky!(factor, centered, n, inflation, execution=nothing)
    _kernel_execute(execution, _gaussian_covariance!, factor, centered, n, inflation)
    return _host_scalar(_kernel_execute(execution, _gaussian_cholesky!, factor))
end

function _gaussian_cholesky!(factor)
    all(isfinite, factor) || return false
    fit = cholesky!(Symmetric(factor, :L); check=false)
    return isposdef(fit) && all(isfinite, factor)
end

_prepare_group!(move::_AllocatedGaussianMove, state, group, complement, streams) =
    _prepare_gaussian_group!(state.batch_workspace, move, state, group, only(complement), streams)

function _prepare_gaussian_group!(workspace, move, state, group, complement, streams)
    for (j, i) in enumerate(complement)
        copyto!(view(move.scratch, :, j), state.positions[i])
    end
    scale, valid = _fit_gaussian!(move.anchor, move.mean, move.factor, move.scratch,
        length(complement), move.shrinkage)
    if valid
        # Store each coordinate contiguously across the active walkers.
        work = reshape(view(move.scratch, :, 1:length(group)), length(group), length(move.mean))
        _gaussian_proposals!(work, move, state, group, scale, streams)
    end
    return _FittedGaussianMove(move, scale, valid)
end

function _gaussian_proposals!(work, move, state, group, scale, streams)
    for (j, i) in enumerate(group)
        _normal_column!(streams.scale, state.draw_buffers.slots[i], view(work, j, :))
    end
    _gaussian_multiply!(move.factor, work)
    for (j, i) in enumerate(group)
        noise = view(work, j, :)
        candidate = state.candidates[i]
        @. candidate = move.anchor + scale * (move.mean + noise)
    end

    # The independence correction uses the rounded candidate coordinates.
    logh = move.host.log_hastings
    _gaussian_distances!(work, state.positions, group, move, scale) do i, distance
        logh[i] = -distance
    end
    _gaussian_distances!(work, state.candidates, group, move, scale) do i, distance
        logh[i] += distance
    end
    return nothing
end

# Multiply by the lower Cholesky factor, working backwards for an in-place product.
function _gaussian_multiply!(factor, work)
    for j in reverse(axes(work, 2))
        x = view(work, :, j)
        for i in (j + 1):size(work, 2)
            y = view(work, :, i)
            coefficient = factor[i, j]
            @. y += coefficient * x
        end
        x .*= factor[j, j]
    end
    return work
end

# Forward substitution uses the same factor for every walker in the group.
function _gaussian_solve!(factor, work)
    for j in axes(work, 2)
        x = view(work, :, j)
        x ./= factor[j, j]
        for i in (j + 1):size(work, 2)
            y = view(work, :, i)
            coefficient = factor[i, j]
            @. y -= coefficient * x
        end
    end
    return work
end

function _gaussian_distances!(store!, work, positions, group, move, scale)
    for (j, i) in enumerate(group)
        residual = view(work, j, :)
        position = positions[i]
        @. residual = (position - move.anchor) / scale - move.mean
    end
    _gaussian_solve!(move.factor, work)
    for (j, i) in enumerate(group)
        # The half-square can fit even when the full squared residual overflows.
        store!(i, sum(x -> (x / 2) * x, view(work, j, :)))
    end
    return nothing
end

function propose!(candidate, fitted::_FittedGaussianMove, current, walker_idx::Integer,
    complement, streams, walker_id::Integer)
    fitted.valid && all(isfinite, candidate) || return nothing
    return fitted.move.host.log_hastings[walker_idx]
end
