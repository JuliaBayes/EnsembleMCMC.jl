abstract type AbstractExecutor end
"""
    SerialExecutor()

Evaluate walker proposals serially within each group of a sweep.
"""
struct SerialExecutor <: AbstractExecutor end
"""
    ThreadedExecutor(; min_chunk=1)

Evaluate each proposal group in balanced, contiguous chunks using Julia threads.
The task count is the default-pool thread count, reduced so that each task gets
at least `min_chunk` walkers, and at least one. There is no timing-based
calibration. Groups still advance in order. The log-density function must support
concurrent calls. A deterministic target gives the same trajectory as
[`SerialExecutor`](@ref) for the same initial state, move, RNG, and walker IDs,
for any `min_chunk`. An error in the target is rethrown unwrapped, as with
[`SerialExecutor`](@ref), after all tasks of the group finish.

Each group costs a task spawn and join of a few microseconds per task. Use
[`SerialExecutor`](@ref) when one walker update costs less than about a
microsecond, or raise `min_chunk` so that each task does at least tens of
microseconds of work.
"""
struct ThreadedExecutor <: AbstractExecutor
    min_chunk::Int
    function ThreadedExecutor(; min_chunk::Integer=1)
        min_chunk >= 1 || throw(ArgumentError("min_chunk must be positive"))
        return new(min_chunk)
    end
end

"""
    KernelExecutor()

Execute proposals and acceptance with KernelAbstractions on the initial matrix's
backend. Requires [`BatchedLogDensity`](@ref). Initialization evaluates the scalar
callback on host vectors; sampling calls the batch callback with backend arrays.
The caller places target data explicitly. CPU and CUDA arrays are supported.
"""
struct KernelExecutor <: AbstractExecutor end

"""
    MoveMixture(moves, weights; schedule=:random)

Select one move per complete sweep. `weights` is a vector or tuple of reals.
`schedule=:random` gives fixed random
selection probabilities after normalization, regardless of weight types.
`schedule=:cycle` gives a repeating weighted cycle in component order and
requires integer-valued weights. Weights must be finite and nonnegative,
with at least one positive weight. The schedule and weights do not adapt.
"""
struct MoveMixture{M<:Tuple,W<:AbstractVector}
    moves::M
    weights::W
    function MoveMixture(moves, weights::Union{AbstractVector{<:Real},Tuple{Vararg{Real}}};
        schedule::Symbol=:random)
        weights isa Tuple && (weights = collect(_promoted_values(Real[weights...])))
        Base.require_one_based_indexing(weights)
        ms = Tuple(moves)
        !isempty(ms) && all(m -> m isa AbstractEnsembleMove, ms) ||
            throw(ArgumentError("A mixture requires ensemble moves"))
        length(ms) == length(weights) || throw(DimensionMismatch("Moves and weights must match"))
        all(w -> isfinite(w) && w >= 0, weights) && any(>(0), weights) ||
            throw(ArgumentError("Weights must be finite, nonnegative and have positive mass"))
        ws = if schedule == :cycle
            all(isinteger, weights) || throw(ArgumentError("Cycle weights must be integer-valued"))
            total = sum(BigInt, weights)
            total <= typemax(Int) || throw(ArgumentError("Mixture cycle is too long"))
            Int.(weights)
        elseif schedule == :random
            values = _promoted_values(map(float, weights))
            scaled = values ./ maximum(values)
            scaled ./ sum(scaled)
        else
            throw(ArgumentError("Mixture schedule must be :random or :cycle"))
        end
        new{typeof(ms),typeof(ws)}(ms, ws)
    end
end

const _PROPOSALS_PER_PURPOSE = typemax(Int16) - 2
const _COMPANION_PURPOSE = 5
const _SCALE_PURPOSE = 6

_stream_index(purpose, proposal) = (purpose - 1) * _PROPOSALS_PER_PURPOSE + proposal
_walker_rngpart(part, purpose, proposal) = rngpart_subpartition(
    part, _stream_index(purpose, proposal), Base.OneTo(typemax(Int32) - 2),
)

mutable struct EnsembleState{F,M,W,E,R,P,V,L,B,A,Q}
    logdensity::F
    moves::M
    weights::W
    executor::E
    rng::R
    cycle_partition::P
    walker_rngs::Vector{R}
    walker_ids::Vector{Int}
    walker_order::Vector{Int}
    positions::V
    candidates::V
    logdensities::L
    candidate_logdensities::L
    batch_workspace::B
    accepted::A
    acceptance_probabilities::Q
    attempts::Vector{Int}
    accepts::Vector{Int}
    step::Int
    active_index::Int
    valid::Bool
end

"""
    current_state(state)

Inspect a valid ensemble without copying arrays or collecting a history.
Return a named tuple with borrowed `positions` (a vector of coordinate vectors),
`logdensities`, `accepted` and `walker_ids` (one entry per walker), and `attempts`
and `acceptances` (cumulative counts per move). Scalar `move_index` names the
move selected for the latest sweep; `sweep_count` counts completed sweeps.
Before the first sweep, both scalars and all counts are zero, and `accepted`
is false for every walker. `candidates`, `candidate_logdensities`, and
`acceptance_probabilities` describe the last attempted transition, including
rejections; a NaN candidate density is recorded as -Inf. Before stepping or after
synchronization they describe the current points with zero acceptance
probabilities and `move_index == 0`.

Treat all borrowed arrays as read-only. Consume them before the next mutation
of the state. Scalar metadata is captured at query time, not updated live.
Do not inspect a state concurrently with stepping. Use [`snapshot`](@ref) to
retain an owned copy. A state invalidated by a failed sweep cannot be queried.
"""
function current_state(state::EnsembleState)
    state.valid || throw(ArgumentError("Cannot inspect a state after a failed sweep"))
    return (; positions=state.positions, logdensities=state.logdensities,
        candidates=state.candidates, candidate_logdensities=state.candidate_logdensities,
        accepted=state.accepted, acceptance_probabilities=state.acceptance_probabilities,
        walker_ids=state.walker_ids,
        attempts=state.attempts, acceptances=state.accepts,
        move_index=state.active_index, sweep_count=state.step)
end

"""
    acceptance_rate(state)

Return the cumulative acceptance rate of each move as a `Vector{Float64}`, in
mixture component order. A move with no attempts yet has rate `NaN`.
[`synchronize!`](@ref) does not reset the counts.
"""
function acceptance_rate(state::EnsembleState)
    current = current_state(state)
    return [a / n for (a, n) in zip(current.acceptances, current.attempts)]
end

"""
    snapshot(state)

Copy the fields returned by [`current_state`](@ref), including every nested
array. Later steps and edits to the snapshot cannot affect each other.
A snapshot contains sample data and counts, not a resumable sampler checkpoint.
"""
snapshot(state::EnsembleState) = _snapshot(state, state.batch_workspace)
function _snapshot(state, workspace)
    current = current_state(state)
    if isbitstype(eltype(first(current.positions))) && isbitstype(eltype(current.logdensities))
        return (; positions=map(copy, current.positions), logdensities=copy(current.logdensities),
            candidates=map(copy, current.candidates), candidate_logdensities=copy(current.candidate_logdensities),
            accepted=copy(current.accepted), acceptance_probabilities=copy(current.acceptance_probabilities),
            walker_ids=copy(current.walker_ids),
            attempts=copy(current.attempts), acceptances=copy(current.acceptances),
            current.move_index, current.sweep_count)
    end
    return deepcopy(current)
end

_coordinate_type(::AbstractVector{<:AbstractVector{T}}) where {T} = T
_coordinate_type(initial) = promote_type(map(eltype, initial)...)
_allocate_move(move, positions) = move
_prepare_group!(move, state, group, complement) = move
_promoted_values(values::AbstractVector{T}) where {T} =
    isconcretetype(T) ? values : collect(promote(values...))

function Base.show(io::IO, state::EnsembleState)
    print(io, "EnsembleState(", length(first(state.positions)), " dimensions, ",
        length(state.positions), " walkers, ", state.step, " sweeps, ",
        state.valid ? "valid" : "invalid", ")")
end

Base.show(io::IO, ::MIME"text/plain", state::EnsembleState) = show(io, state)

"""
    validate_positions(positions)

Check that a vector of coordinate vectors or a coordinate-by-walker matrix has
positive, matching dimensions, finite real coordinates, and full affine rank.
Return `nothing` without changing the input. Rank validation uses an owned host
matrix of the positions relative to the first walker, in the floating-point
coordinate type `T`. Singular values at or below `max(d, n) * eps(T)` times the
largest one count as zero, as in `LinearAlgebra.rank`. A full-rank ensemble whose
smallest-to-largest singular value ratio is below `sqrt(eps(T))` gives a warning,
because linear moves then stay close to a lower-dimensional subspace.
Types without an SVD, such as `BigFloat`, use the diagonal of a pivoted QR
factorization as the singular value estimate.
Call this after retry initialization, before synchronizing replacement positions.
Move-specific walker requirements are checked by [`initialize`](@ref).
"""
function validate_positions(positions::AbstractVector)
    Base.require_one_based_indexing(positions)
    isempty(positions) && throw(ArgumentError("An ensemble cannot be empty"))
    d = length(first(positions))
    d > 0 && all(x -> length(x) == d, positions) ||
        throw(DimensionMismatch("Walker dimensions must agree and be positive"))
    T = float(_coordinate_type(positions))
    T <: AbstractFloat || throw(ArgumentError("Coordinates must be real floating-point values"))
    centered = Matrix{T}(reduce(hcat, positions))
    all(isfinite, centered) || throw(ArgumentError("Coordinates must be finite"))
    centered .-= centered[:, 1]
    values = _singular_values(centered)
    tolerance = max(size(centered)...) * eps(T) * first(values)
    count(>(tolerance), values) == d || throw(ArgumentError("Walkers must have full affine rank"))
    ratio = values[d] / first(values)
    ratio < sqrt(eps(T)) && @warn string("Walkers are nearly degenerate: the singular value ratio ",
        Float64(ratio), " is below sqrt(eps($T)). Linear moves stay close to a subspace.")
    return nothing
end

_singular_values(x::Matrix{<:LinearAlgebra.BlasFloat}) = svdvals!(x)
_singular_values(x) = sort!(abs.(diag(qr!(x, ColumnNorm()).R)); rev=true)

function validate_positions(positions::AbstractMatrix)
    Base.require_one_based_indexing(positions)
    return validate_positions(eachcol(positions))
end

const _SUPPORTED_RNGS = "Random123 Philox4x or Threefry4x with UInt64 counters, such as Philox4x((seed, chain))"

"""
    initialize(rng, logdensity, initial; move=StretchMove(), executor=SerialExecutor(),
               walker_ids=1:nwalkers, logdensities=nothing)

Create one ensemble from a vector of finite coordinate vectors or a matrix
with axes (coordinate, walker). The coordinates must have full affine rank
and initial log densities other than NaN or +Inf. Initial values of -Inf are
allowed with a warning: stretch, DE and snooker walkers far outside the support
may never recover. The state owns copies of coordinates and RNG.

Supported RNGs are Random123 `Philox4x` and `Threefry4x` with UInt64 counters.
`initialize` copies the RNG and does not advance it, so two states made from the
same RNG object give identical chains. For independent ensembles, use distinct
keys, for example `Philox4x((seed, chain))` for `chain = 1, 2, ...`.
Standalone [`step!`](@ref) needs four free partition levels, so an RNG derived
through at most one partition level (depth 2 or less) is required.

Supply `logdensities` to reuse cached values without evaluating the target.
The vector must match walker order and contain real values other than NaN or
+Inf. Values are copied and promoted independently of coordinate precision.
The caller must ensure they equal the target at the initial coordinates.

Walker IDs default to the walker indices `1:nwalkers` and remain fixed throughout
the run. The density must not mutate its input. A scalar density must support
concurrent calls with ThreadedExecutor; a batched callback controls its own
parallelism. During sampling, a candidate log density of NaN rejects the
candidate like -Inf, and +Inf throws a `DomainError`. Do not mutate state fields.
Use [`current_state`](@ref) for borrowed inspection and [`snapshot`](@ref) for copies.
"""
function initialize(
    rng::Union{Philox4x{UInt64},Threefry4x{UInt64}}, logdensity, initial::AbstractVector;
    move = StretchMove(), executor::AbstractExecutor = SerialExecutor(),
    walker_ids = collect(eachindex(initial)),
    logdensities::Union{Nothing,AbstractVector} = nothing, kwargs...,
)
    isempty(kwargs) || throw(ArgumentError(
        "Unsupported keyword argument(s): $(join(keys(kwargs), ", "))"))
    executor isa KernelExecutor && throw(ArgumentError("KernelExecutor requires an initial matrix"))
    move isa Union{AbstractEnsembleMove,MoveMixture} || throw(ArgumentError(
        "move must be an ensemble move or a MoveMixture, not $(typeof(move))"))
    depth = rngpart_depth(rng)
    depth <= 2 || throw(ArgumentError(
        "RNG partition depth $depth is too deep; standalone stepping needs depth 2 or less"))
    validate_positions(initial)
    d = length(first(initial))
    T = float(_coordinate_type(initial))
    positions = [Vector{T}(x) for x in initial]
    moves = move isa MoveMixture ? map(m -> prepare_move(m, T, d), move.moves) :
        (prepare_move(move, T, d),)
    length(moves) <= _PROPOSALS_PER_PURPOSE || throw(ArgumentError("Too many mixture components"))
    n = length(positions)
    required = maximum(m -> minimum_walkers(m, d), moves)
    n >= required || throw(ArgumentError(
        "Too few walkers: $n given, the selected moves need $required in $d dimensions"))
    moves = map(m -> _allocate_move(m, positions), moves)
    length(walker_ids) == n && all(id -> id isa Integer && 1 <= id <= typemax(Int32) - 2, walker_ids) &&
        allunique(walker_ids) ||
        throw(ArgumentError("Walker IDs must be $n distinct integers in 1:$(typemax(Int32) - 2)"))
    ids = collect(Int, walker_ids)
    values = if isnothing(logdensities)
        [_checked_initial_logdensity(logdensity(x), i) for (i, x) in enumerate(positions)]
    else
        Base.require_one_based_indexing(logdensities)
        length(logdensities) == n || throw(DimensionMismatch("Walker and log-density counts must agree"))
        [_checked_initial_logdensity(v, i) for (i, v) in enumerate(logdensities)]
    end
    logds = _promoted_values(values)
    _warn_negative_infinity(logds)
    owned_rng = copy(rng)
    cycle_part = RNGPartition(owned_rng, 0:(typemax(Int16) - 2))
    weights = move isa MoveMixture ? copy(move.weights) : nothing
    return EnsembleState(
        logdensity, moves, weights, executor, owned_rng, cycle_part,
        [rngpart_createrng(typeof(rng)) for _ in 1:n], ids, sortperm(ids),
        positions, deepcopy(positions), logds, copy(logds),
        _batch_workspace(logdensity, positions, logds), fill(false, n), zeros(T, n),
        zeros(Int, length(moves)), zeros(Int, length(moves)), 0, 0, true,
    )
end

function initialize(rng::Union{Philox4x{UInt64},Threefry4x{UInt64}}, logdensity,
    initial::AbstractMatrix; executor=SerialExecutor(), kwargs...)
    Base.require_one_based_indexing(initial)
    executor isa KernelExecutor && return _initialize_kernel(rng, logdensity, initial; kwargs...)
    return initialize(rng, logdensity, eachcol(initial); executor, kwargs...)
end

function initialize(rng, logdensity, initial; kwargs...)
    rng isa Union{Philox4x{UInt64},Threefry4x{UInt64}} ||
        throw(ArgumentError("Unsupported RNG $(typeof(rng)); use $_SUPPORTED_RNGS"))
    throw(ArgumentError("Initial positions must be a vector of coordinate vectors or a coordinate-by-walker matrix"))
end

function _checked_initial_logdensity(value, i)
    value isa Real || throw(ArgumentError("Log density must return a real number, got $(typeof(value)) for walker $i"))
    (isnan(value) || value == Inf) &&
        throw(DomainError(value, "Invalid initial log density for walker $i: NaN and +Inf are not allowed"))
    return float(value)
end

function _warn_negative_infinity(logds)
    k = count(==(-Inf), logds)
    if k == length(logds)
        @warn "All $k initial log densities are -Inf. Walkers cannot move until a candidate has a finite density."
    elseif k > 0
        @warn string("$k of $(length(logds)) initial log densities are -Inf. Stretch, DE and snooker ",
            "walkers far outside the support may never recover.")
    end
    return nothing
end

Base.@inline function _accept_candidate!(state, i, log_hastings, logd, acceptance_part, batched=false)
    stored_logd = convert(eltype(state.logdensities), logd)
    if isnan(stored_logd)
        stored_logd = convert(eltype(state.logdensities), -Inf)
    elseif stored_logd == Inf
        _invalid_candidate(state, i, stored_logd, batched)
    end
    state.candidate_logdensities[i] = stored_logd
    T = eltype(state.positions[i])
    logratio = convert(T, log_hastings + stored_logd - state.logdensities[i])
    probability = isnan(logratio) ? zero(T) : clamp(exp(logratio), zero(T), one(T))
    rng = state.walker_rngs[i]
    set_rng!(rng, acceptance_part, state.walker_ids[i])
    state.acceptance_probabilities[i] = probability
    state.accepted[i] = rand(rng, T) < probability
    return nothing
end

@noinline function _invalid_candidate(state, i, value, batched)
    location = "walker $i (ID $(state.walker_ids[i])) in sweep $(state.step + 1)"
    throw(DomainError(value, batched ?
        "Invalid batched log density for $location: +Inf is not allowed and batch! must fill every value" :
        "Invalid log density for $location: candidate values must not be +Inf"))
end

_select_move(::Nothing, rng, step) = 1
function _select_move(weights::Vector{Int}, rng, step)
    offset = mod1(step, sum(weights))
    total = 0
    for i in eachindex(weights)
        total += weights[i]
        offset <= total && return i
    end
    error("Invalid mixture cycle")
end
function _select_move(weights::AbstractVector{T}, rng, step) where {T<:AbstractFloat}
    u = rand(rng, T) * sum(weights)
    total = zero(T)
    for i in eachindex(weights)
        total += weights[i]
        u < total && return i
    end
    return something(findlast(>(0), weights))
end

function _groups(rng, move, part, proposal_idx, order)
    rngpart_setfresh!(rng, part, _stream_index(4, proposal_idx))
    permutation = randperm(rng, length(order))
    ngroups = group_count(move)
    if ngroups == 2
        split = fld(length(order), 2)
        left = order[view(permutation, 1:split)]
        right = order[view(permutation, (split + 1):length(permutation))]
        return rand(rng, Bool) ? (left, right) : (right, left)
    end
    size, extra = divrem(length(order), ngroups)
    groups = Vector{Vector{Int}}(undef, ngroups)
    start = 1
    for i in eachindex(groups)
        stop = start + size - 1 + (i <= extra)
        groups[i] = order[permutation[start:stop]]
        start = stop + 1
    end
    shuffle!(rng, groups)
    return groups
end

_complement(groups::Tuple, i) = i == 1 ? (groups[2],) : (groups[1],)
_complement(groups::AbstractVector, i) = groups[eachindex(groups) .!= i]

function _evaluate!(state, move, part, proposal_idx, i, complement, acceptance_part)
    rng = state.walker_rngs[i]
    log_hastings = propose!(state.candidates[i], move, state.positions, i,
        complement, rng, part, proposal_idx, state.walker_ids[i])
    if isnothing(log_hastings)
        copyto!(state.candidates[i], state.positions[i])
        state.candidate_logdensities[i] = state.logdensities[i]
        state.acceptance_probabilities[i] = 0
        state.accepted[i] = false
        return nothing
    end
    return _evaluate_candidate!(state.batch_workspace, state, i, log_hastings, acceptance_part)
end

function _evaluate_group!(::SerialExecutor, state, move, part, move_index, proposal_index,
    group, complement, acceptance_part)
    for i in group
        _evaluate!(state, move, part, proposal_index, i, complement, acceptance_part)
    end
end
function _evaluate_chunk!(task, ntasks, state, move, part, idx, group, complement, acceptance_part)
    n = length(group)
    for k in (fld((task - 1) * n, ntasks) + 1):fld(task * n, ntasks)
        _evaluate!(state, move, part, idx, group[k], complement, acceptance_part)
    end
end

function _evaluate_group!(executor::ThreadedExecutor, state, move, part, move_index, proposal_index,
    group, complement, acceptance_part)
    ntasks = clamp(fld(length(group), executor.min_chunk), 1, Threads.nthreads(:default))
    ntasks == 1 && return _evaluate_chunk!(1, 1, state, move, part,
        proposal_index, group, complement, acceptance_part)
    tasks = map(Base.OneTo(ntasks)) do task
        Threads.@spawn _evaluate_chunk!(task, ntasks, state, move, part,
            proposal_index, group, complement, acceptance_part)
    end
    _wait_tasks(tasks)
    return nothing
end

# Wait for every task before throwing, and rethrow the first task error unwrapped
# so that callers see the same exception type as with SerialExecutor.
function _wait_tasks(tasks)
    failure = nothing
    for task in tasks
        try
            wait(task)
        catch err
            isnothing(failure) && (failure = err isa TaskFailedException ? err.task.exception : err)
        end
    end
    isnothing(failure) || throw(failure)
    return nothing
end

Base.@inline function _evaluate_candidate!(::Nothing, state, i, log_hastings, acceptance_part)
    logd = state.logdensity(state.candidates[i])
    logd isa Real || throw(ArgumentError(
        "Log density must return a real number, got $(typeof(logd)) for walker $i"))
    return _accept_candidate!(state, i, log_hastings, logd, acceptance_part)
end
function _evaluate_candidate!(workspace::BatchWorkspace, state, i, log_hastings, acceptance_part)
    workspace.log_hastings[i] = log_hastings
    state.accepted[i] = true # Marks a valid proposal until the batch is evaluated.
    return nothing
end

_evaluate_batch!(::Nothing, state, group, acceptance_part) = nothing
function _evaluate_batch!(workspace::BatchWorkspace, state, group, acceptance_part)
    count = 0
    for i in group
        if state.accepted[i]
            count += 1
            copyto!(view(workspace.positions, :, count), state.candidates[i])
        end
    end
    iszero(count) && return nothing
    values = view(workspace.values, 1:count)
    positions = view(workspace.positions, :, 1:count)
    # +Inf is invalid, so unfilled entries still raise an error while NaN rejects.
    fill!(values, Inf)
    state.logdensity.batch!(values, positions)
    k = 0
    for i in group
        state.accepted[i] || continue
        k += 1
        _accept_candidate!(state, i, workspace.log_hastings[i], values[k], acceptance_part, true)
    end
    return nothing
end

"""
    step!(state)
    step!(state, nsweeps)

Advance one full ensemble sweep, or `nsweeps` complete sweeps without storing
a history. Mutate and return the state. Zero sweeps leave a valid state unchanged.
Each sweep selects one move and updates every walker once, in ordered groups
against frozen complements. Use [`current_state`](@ref) to consume each sweep
without a copy, or [`snapshot`](@ref) to retain it.
A candidate log density of NaN rejects the candidate. An exception during a
sweep, including the `DomainError` for a candidate density of +Inf, invalidates
the state. Initialize a fresh state after correcting the target instead of
resuming a partially completed sweep. Argument and RNG-depth errors raised before
the sweep starts leave the state valid.
"""
step!(state::EnsembleState) = _step!(state, state.batch_workspace)
_step!(state, workspace, args...) = _step!(state, args...)
function _step!(state)
    state.valid || throw(ArgumentError("Cannot resume a state after a failed sweep"))
    state.step < typemax(Int32) - 2 || throw(ArgumentError("RNG step range exhausted"))
    set_rng!(state.rng, state.cycle_partition, 0)
    steps = RNGPartition(state.rng, 0:(typemax(Int32) - 2))
    set_rng!(state.rng, steps, state.step)
    part = RNGPartition(state.rng, Base.OneTo(6 * _PROPOSALS_PER_PURPOSE))
    return _sweep!(state, part, nothing)
end

"""
    step!(state, rng; proposal_index=1)

Advance one sweep using an externally addressed RNG without changing `rng`.
The RNG must match the initialization RNG type and leave two partition levels
for purposes and walkers, so its partition depth must be 4 or less.
`proposal_index` in `1:$(_PROPOSALS_PER_PURPOSE)` selects disjoint proposal streams.
Inner mixtures use purpose 2 for selection, leaving purpose 1 for an outer mixture.
The supplied address replaces the standalone cycle/step address for this sweep only.
The sweep still counts: `sweep_count` advances, so a later standalone
`step!(state)` uses the next step address, and a cyclic mixture moves to its
next phase.
"""
function step!(state::EnsembleState, rng::AbstractRNG; proposal_index::Integer=1)
    typeof(rng) === typeof(state.rng) || throw(ArgumentError("RNG type must match the ensemble state"))
    1 <= proposal_index <= _PROPOSALS_PER_PURPOSE ||
        throw(ArgumentError("Proposal index is outside the stream range"))
    depth = rngpart_depth(rng)
    depth <= 4 || throw(ArgumentError("RNG partition depth $depth is too deep; step!(state, rng) needs depth 4 or less"))
    return _step!(state, state.batch_workspace, rng, proposal_index)
end

function _step!(state, rng::AbstractRNG, proposal_index::Integer)
    part = RNGPartition(copy(rng), Base.OneTo(6 * _PROPOSALS_PER_PURPOSE))
    return _sweep!(state, part, proposal_index)
end

function _sweep!(state, part, proposal_index)
    state.valid || throw(ArgumentError("Cannot resume a state after a failed sweep"))
    state.step < typemax(Int32) - 2 || throw(ArgumentError("RNG step range exhausted"))
    part.depth + 2 <= length(part.partctrsbase) ||
        throw(ArgumentError("RNG partition is too deep for walker streams"))
    # The owned RNG is re-keyed at the start of each standalone sweep, so it can serve as scratch.
    scratch = state.rng
    rngpart_setfresh!(scratch, part, isnothing(proposal_index) ?
        _stream_index(1, 1) : _stream_index(2, proposal_index))
    idx = _select_move(state.weights, scratch, state.step + 1)
    stream_index = isnothing(proposal_index) ? idx : proposal_index
    move = state.moves[idx]
    groups = _groups(scratch, move, part, stream_index, state.walker_order)
    acceptance_part = _walker_rngpart(part, 3, stream_index)
    # Everything above leaves the ensemble untouched, so a failure there keeps the state valid.
    state.valid = false

    for active in eachindex(groups)
        group = groups[active]
        complement = _complement(groups, active)
        fitted_move = _prepare_group!(move, state, group, complement)
        _evaluate_group!(state.executor, state, fitted_move, part, idx, stream_index, group,
            complement, acceptance_part)
        _evaluate_batch!(state.batch_workspace, state, group, acceptance_part)
        _commit_group!(state, group, state.batch_workspace)
    end
    state.active_index = idx
    state.attempts[idx] += length(state.positions)
    state.accepts[idx] += _accepted_count(state, state.batch_workspace)
    state.step += 1
    state.valid = true
    return state
end

function _commit_group!(state, group, workspace)
    for i in group
        if state.accepted[i]
            copyto!(state.positions[i], state.candidates[i])
            state.logdensities[i] = state.candidate_logdensities[i]
        end
    end
end
_accepted_count(state, workspace) = count(state.accepted)

function step!(state::EnsembleState, nsweeps::Integer)
    nsweeps >= 0 || throw(ArgumentError("Sweep count must be nonnegative"))
    state.valid || throw(ArgumentError("Cannot resume a state after a failed sweep"))
    for _ in 1:nsweeps
        step!(state)
    end
    return state
end

"""
    sample!(state, nsamples; thin=1)

Advance `nsamples * thin` complete sweeps and store the state after every
`thin`-th sweep. Positions have axes (coordinate, walker, sample). Rejections
repeat the current state. Returned arrays own their storage. Each state
represents one coupled ensemble, not independent walker chains.

Return a named tuple with `positions`, `logdensities` and `accepted` (the latter
two indexed by walker and sample), `move_indices` (one move index per sample), and
`walker_ids` (one ID per walker). `accepted` and `move_indices` describe the
stored sweep only; the cumulative counts of [`current_state`](@ref) include
every sweep. The initial positions are not included.

The history is allocated up front and returned only on success. An exception
during a sweep invalidates the state and loses the collected history. For
checkpoints, progress output or custom storage, call [`step!`](@ref) in a loop
and keep [`snapshot`](@ref)s or read [`current_state`](@ref) after each sweep.
"""
function sample!(state::EnsembleState, nsamples::Integer; thin::Integer=1)
    nsamples >= 0 || throw(ArgumentError("Sample count must be nonnegative"))
    thin >= 1 || throw(ArgumentError("thin must be positive"))
    nsamples <= typemax(Int) ÷ thin || throw(ArgumentError("Sweep count overflows"))
    state.valid || throw(ArgumentError("Cannot resume a state after a failed sweep"))
    history = _allocate_history(state, nsamples, state.batch_workspace)
    for sample in 1:nsamples
        step!(state, thin)
        _store_history!(history, state, sample, state.batch_workspace)
    end
    return (; history..., walker_ids=copy(state.walker_ids))
end

function _allocate_history(state, nsweeps, workspace)
    n, d = length(state.positions), length(first(state.positions))
    return (; positions=Array{eltype(first(state.positions))}(undef, d, n, nsweeps),
        logdensities=Matrix{eltype(state.logdensities)}(undef, n, nsweeps),
        accepted=Matrix{Bool}(undef, n, nsweeps), move_indices=Vector{Int}(undef, nsweeps))
end
function _store_history!(history, state, sweep, workspace)
    d, n = size(history.positions, 1), length(state.positions)
    offset = (sweep - 1) * d * n
    for i in eachindex(state.positions)
        copyto!(history.positions, offset + (i - 1) * d + 1, state.positions[i], 1, d)
    end
    copyto!(history.logdensities, (sweep - 1) * n + 1, state.logdensities, 1, n)
    copyto!(history.accepted, (sweep - 1) * n + 1, state.accepted, 1, n)
    history.move_indices[sweep] = state.active_index
end

"""
    synchronize!(state, positions, logdensities)

Copy externally owned positions and their matching cached log densities into a
valid state. Inputs may be arrays borrowed from this state in their original
walker order; any other overlap with state storage, such as a reordered vector
of borrowed walkers, gives undefined results. Accept a
coordinate-by-walker matrix or a vector of coordinate vectors. Walker identities,
RNG, sweep count, mixture phase, and cumulative counters stay unchanged.
Clear the last-transition metadata. No target calls or affine-rank check run here.
The caller must preserve density consistency and affine rank. Use
[`validate_positions`](@ref) after replacing positions through initialization retries.
"""
function synchronize!(state::EnsembleState, positions::AbstractVector, logdensities::AbstractVector)
    return _synchronize!(state, positions, logdensities, state.batch_workspace)
end

synchronize!(state::EnsembleState, positions::AbstractMatrix, logdensities::AbstractVector) =
    synchronize!(state, eachcol(positions), logdensities)

function _synchronize!(state, positions, logdensities, workspace)
    return _synchronize!(state, positions, logdensities,
        eltype(first(state.positions)), eltype(state.logdensities))
end

function _synchronize!(state, positions, logdensities, ::Type{T}, ::Type{L}) where {T,L}
    state.valid || throw(ArgumentError("Cannot synchronize a state after a failed sweep"))
    Base.require_one_based_indexing(positions, logdensities)
    length(positions) == length(logdensities) == length(state.positions) ||
        throw(DimensionMismatch("Walker counts must agree"))
    d = length(first(state.positions))
    all(x -> length(x) == d && all(v -> isfinite(convert(T, v)), x), positions) ||
        throw(ArgumentError("Positions must have matching dimensions and finite coordinates"))
    all(v -> !isnan(convert(L, v)) && convert(L, v) != Inf, logdensities) ||
        throw(ArgumentError("Log densities must not contain NaN or +Inf"))
    for i in eachindex(positions)
        copyto!(state.positions[i], positions[i])
        copyto!(state.candidates[i], positions[i])
    end
    copyto!(state.logdensities, logdensities)
    copyto!(state.candidate_logdensities, logdensities)
    fill!(state.accepted, false)
    fill!(state.acceptance_probabilities, 0)
    state.active_index = 0
    return state
end
