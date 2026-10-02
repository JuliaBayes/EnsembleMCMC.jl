abstract type AbstractEnsembleMove end

"""
    StretchMove(; scale=2)

Stretch a walker away from a companion in the frozen complement. The stretch
factor lies in `[1/scale, scale]`; `scale` must be finite and greater than one.
Uses two groups and requires at least `2d` walkers for dimension `d`.
"""
struct StretchMove{S<:Real} <: AbstractEnsembleMove
    scale::S

    function StretchMove(scale::S) where {S<:Real}
        isfinite(scale) && scale > one(scale) || throw(ArgumentError(
            "StretchMove scale must be finite and greater than 1",
        ))
        return new{S}(scale)
    end
end

StretchMove(; scale::Real = 2) = StretchMove(scale)

"""
    DEMove(; gamma0=nothing, sigma=1e-5)

Add a scaled difference of two distinct companions in the frozen complement.
The scale is `gamma0 * (1 + sigma * randn())`, with default
`gamma0 = 2.38 / sqrt(2d)` for dimension `d`. `gamma0` must be finite and
positive when supplied. `sigma` must be finite and nonnegative.
Uses two groups and requires at least `max(2d, 4)` walkers.
"""
struct DEMove{G<:Union{Nothing,Real},S<:Real} <: AbstractEnsembleMove
    gamma0::G
    sigma::S

    function DEMove(gamma0::G, sigma::S) where {G<:Union{Nothing,Real},S<:Real}
        (isnothing(gamma0) || isfinite(gamma0) && gamma0 > zero(gamma0)) ||
            throw(ArgumentError("DEMove gamma0 must be nothing or finite and positive"))
        isfinite(sigma) && sigma >= zero(sigma) || throw(ArgumentError(
            "DEMove sigma must be finite and nonnegative",
        ))
        return new{G,S}(gamma0, sigma)
    end
end

DEMove(; gamma0::Union{Nothing,Real} = nothing, sigma::Real = 1e-5) =
    DEMove(gamma0, sigma)

"""
    DESnookerMove(; scale=1.7)

Project a companion difference onto the line from a reference walker to the
active walker, then multiply that displacement by `scale`. The scale must be
finite and positive. Degenerate reference directions produce a rejection.
Uses four groups and requires at least `max(2d, 4)` walkers for dimension `d`.
This move is not generally affine-equivariant.
"""
struct DESnookerMove{S<:Real} <: AbstractEnsembleMove
    scale::S

    function DESnookerMove(scale::S) where {S<:Real}
        isfinite(scale) && scale > zero(scale) || throw(ArgumentError(
            "DESnookerMove scale must be finite and positive",
        ))
        return new{S}(scale)
    end
end

DESnookerMove(; scale::Real = 1.7) = DESnookerMove(scale)

group_count(::StretchMove) = 2
group_count(::DEMove) = 2
group_count(::DESnookerMove) = 4

minimum_walkers(::StretchMove, dimension::Integer) = 2 * dimension
minimum_walkers(::Union{DEMove,DESnookerMove}, dimension::Integer) =
    max(2 * dimension, 4)

function prepare_move(move::StretchMove, ::Type{T}, ::Integer) where {T<:Real}
    F = float(T)
    scale = convert(F, move.scale)
    isfinite(scale) && scale > one(scale) || throw(ArgumentError(
        "StretchMove scale must remain finite and greater than 1 after conversion to $(F)",
    ))
    return StretchMove(scale)
end

function prepare_move(move::DEMove, ::Type{T}, dimension::Integer) where {T<:Real}
    F = float(T)
    gamma0 = if isnothing(move.gamma0)
        convert(F, 2.38) / sqrt(convert(F, 2 * dimension))
    else
        convert(F, move.gamma0)
    end
    sigma = convert(F, move.sigma)
    isfinite(gamma0) && gamma0 > zero(gamma0) || throw(ArgumentError(
        "DEMove gamma0 must remain finite and positive after conversion to $(F)",
    ))
    isfinite(sigma) && sigma >= zero(sigma) || throw(ArgumentError(
        "DEMove sigma must remain finite and nonnegative after conversion to $(F)",
    ))
    return DEMove(gamma0, sigma)
end

function prepare_move(move::DESnookerMove, ::Type{T}, ::Integer) where {T<:Real}
    F = float(T)
    scale = convert(F, move.scale)
    isfinite(scale) && scale > zero(scale) || throw(ArgumentError(
        "DESnookerMove scale must remain finite and positive after conversion to $(F)",
    ))
    return DESnookerMove(scale)
end

function _de_companion_indices(
    rng::AbstractRNG,
    companion_indices::AbstractVector{<:Integer},
)
    Base.require_one_based_indexing(companion_indices)
    n_companions = length(companion_indices)
    n_companions >= 2 || throw(ArgumentError(
        "DEMove requires at least two walkers in each frozen complement",
    ))
    first_position = rand(rng, 1:n_companions)
    second_position = rand(rng, 1:(n_companions - 1))
    second_position >= first_position && (second_position += 1)
    return companion_indices[first_position], companion_indices[second_position]
end

function propose!(
    candidate,
    move::StretchMove,
    current,
    walker_idx::Integer,
    complement_groups,
    rng::AbstractRNG,
    step_rngpart::RNGPartition,
    proposal_idx::Integer,
    walkerid::Integer,
)
    companion_indices = only(complement_groups)
    companion_rngpart = _walker_rngpart(
        step_rngpart, _COMPANION_PURPOSE, proposal_idx,
    )
    scale_rngpart = _walker_rngpart(step_rngpart, _SCALE_PURPOSE, proposal_idx)

    set_rng!(rng, companion_rngpart, walkerid)
    companion_idx = rand(rng, companion_indices)

    set_rng!(rng, scale_rngpart, walkerid)
    T = float(eltype(current[walker_idx]))
    u = rand(rng, T)
    b = (move.scale - one(move.scale)) * u + one(move.scale)
    stretch = b * (b / move.scale)
    @. candidate = current[companion_idx] +
        stretch * (current[walker_idx] - current[companion_idx])
    return (length(candidate) - 1) * log(stretch)
end

function propose!(
    candidate,
    move::DEMove,
    current,
    walker_idx::Integer,
    complement_groups,
    rng::AbstractRNG,
    step_rngpart::RNGPartition,
    proposal_idx::Integer,
    walkerid::Integer,
)
    companion_indices = only(complement_groups)
    companion_rngpart = _walker_rngpart(
        step_rngpart, _COMPANION_PURPOSE, proposal_idx,
    )
    scale_rngpart = _walker_rngpart(step_rngpart, _SCALE_PURPOSE, proposal_idx)

    set_rng!(rng, companion_rngpart, walkerid)
    companion_a, companion_b = _de_companion_indices(rng, companion_indices)

    set_rng!(rng, scale_rngpart, walkerid)
    T = typeof(move.gamma0)
    gamma = move.gamma0 * (one(T) + move.sigma * randn(rng, T))
    @. candidate = current[walker_idx] +
        gamma * (current[companion_a] - current[companion_b])
    return zero(move.gamma0)
end

const _DE_SNOOKER_GROUP_ORDERS = (
    (1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1),
)

function _de_snooker_companion_indices(rng::AbstractRNG, complement_groups)
    length(complement_groups) == 3 || throw(ArgumentError(
        "DESnookerMove requires exactly three frozen complement groups",
    ))
    all(!isempty, complement_groups) || throw(ArgumentError(
        "DESnookerMove requires nonempty frozen complement groups",
    ))
    group_order = rand(rng, _DE_SNOOKER_GROUP_ORDERS)
    return (
        rand(rng, complement_groups[group_order[1]]),
        rand(rng, complement_groups[group_order[2]]),
        rand(rng, complement_groups[group_order[3]]),
    )
end

function _de_snooker_direction_norm(a, b)
    return norm((a[i] - b[i] for i in eachindex(a, b)))
end

@inline function _projected_difference(direction, a, b)
    difference = a - b
    # A finite projection can survive an overflowing companion difference.
    return ifelse(isfinite(difference), direction * difference, direction * a - direction * b)
end

function propose!(
    candidate,
    move::DESnookerMove,
    current,
    walker_idx::Integer,
    complement_groups,
    rng::AbstractRNG,
    step_rngpart::RNGPartition,
    proposal_idx::Integer,
    walkerid::Integer,
)
    companion_rngpart = _walker_rngpart(
        step_rngpart, _COMPANION_PURPOSE, proposal_idx,
    )
    set_rng!(rng, companion_rngpart, walkerid)
    reference_idx, companion_a_idx, companion_b_idx =
        _de_snooker_companion_indices(rng, complement_groups)

    current_walker = current[walker_idx]
    reference = current[reference_idx]
    direction_norm = _de_snooker_direction_norm(current_walker, reference)
    if iszero(direction_norm) || !isfinite(direction_norm)
        copyto!(candidate, current_walker)
        return nothing
    end

    @. candidate = (current_walker - reference) / direction_norm
    displacement = move.scale * sum(eachindex(candidate)) do i
        _projected_difference(candidate[i], current[companion_a_idx][i], current[companion_b_idx][i])
    end
    @. candidate = current_walker + candidate * displacement
    proposed_direction_norm = _de_snooker_direction_norm(candidate, reference)
    if iszero(proposed_direction_norm) || !isfinite(proposed_direction_norm)
        copyto!(candidate, current_walker)
        return nothing
    end
    return (length(current_walker) - 1) * (
        log(proposed_direction_norm) - log(direction_norm)
    )
end
