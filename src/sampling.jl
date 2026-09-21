abstract type AbstractExecutor end
"""
    SerialExecutor()

Evaluate walker proposals serially within each group of a sweep.
"""
struct SerialExecutor <: AbstractExecutor end
"""
    ThreadedExecutor()

Use Julia threads when proposal groups contain enough work to benefit.
The first few groups calibrate task size and compare serial and threaded costs
for each move. Calibration uses normal proposals, without extra density calls.
Groups still advance in order. The log-density function must support concurrent
calls. A deterministic target gives the same trajectory as [`SerialExecutor`](@ref)
for the same initial state, move, RNG, and walker IDs.
"""
struct ThreadedExecutor <: AbstractExecutor end

mutable struct _GroupSchedule
    samples::Int
    ntasks::Int
    serial_ns::Float64
    threaded_ns::Float64
end
struct _ThreadedExecutor <: AbstractExecutor
    schedules::Vector{_GroupSchedule}
end

_prepare_executor(executor, nmoves) = executor
function _prepare_executor(::ThreadedExecutor, nmoves)
    samples = Threads.nthreads(:default) == 1 ? 8 : 0
    return _ThreadedExecutor([_GroupSchedule(samples, 1, Inf, Inf) for _ in 1:nmoves])
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

Select one move per complete sweep. `schedule=:random` gives fixed random
selection probabilities after normalization, regardless of weight types.
`schedule=:cycle` gives a repeating weighted cycle in component order and
requires integer-valued weights. Weights must be finite and nonnegative,
with at least one positive weight. The schedule and weights do not adapt.
"""
struct MoveMixture{M<:Tuple,W<:AbstractVector}
    moves::M
    weights::W
    function MoveMixture(moves, weights::AbstractVector{<:Real}; schedule::Symbol=:random)
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
_walker_rngpart(part, purpose, proposal) = RNGPartition(
    AbstractRNG(part, _stream_index(purpose, proposal)), Base.OneTo(typemax(Int32) - 2),
)

mutable struct EnsembleState{F,M,W,E,R,P,V,L,B,A}
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
is false for every walker.

Treat all borrowed arrays as read-only. Consume them before the next mutation
of the state. Scalar metadata is captured at query time, not updated live.
Do not inspect a state concurrently with stepping. Use [`snapshot`](@ref) to
retain an owned copy. A state invalidated by a failed sweep cannot be queried.
"""
function current_state(state::EnsembleState)
    state.valid || throw(ArgumentError("Cannot inspect a state after a failed sweep"))
    return (; positions=state.positions, logdensities=state.logdensities,
        accepted=state.accepted, walker_ids=state.walker_ids,
        attempts=state.attempts, acceptances=state.accepts,
        move_index=state.active_index, sweep_count=state.step)
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
            accepted=copy(current.accepted), walker_ids=copy(current.walker_ids),
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
    initialize(rng, logdensity, initial; move=StretchMove(), executor=SerialExecutor(),
               walker_ids=1:nwalkers)

Create one ensemble from a vector of finite coordinate vectors or a matrix
with axes (coordinate, walker). The coordinates must have full affine rank
and finite initial log densities. The state owns
copies of coordinates and RNG. Supported RNGs are Random123 Philox4x/Threefry4x
with UInt64 counters.
Use independent RNG seeds or streams for independent ensembles.

Walker IDs default to `1:nwalkers` and remain fixed throughout the run.
The density must not mutate its input. A scalar density must support concurrent
calls with ThreadedExecutor; a batched callback controls its own parallelism.
Do not mutate state fields.
Use [`current_state`](@ref) for borrowed inspection and [`snapshot`](@ref) for copies.
"""
function initialize(
    rng::Union{Philox4x{UInt64},Threefry4x{UInt64}}, logdensity, initial::AbstractVector;
    move = StretchMove(), executor::AbstractExecutor = SerialExecutor(),
    walker_ids = collect(eachindex(initial)),
)
    executor isa KernelExecutor && throw(ArgumentError("KernelExecutor requires an initial matrix"))
    isempty(initial) && throw(ArgumentError("An ensemble cannot be empty"))
    Base.require_one_based_indexing(initial)
    d = length(first(initial))
    d > 0 && all(x -> length(x) == d, initial) ||
        throw(DimensionMismatch("Walker dimensions must agree and be positive"))
    T = float(_coordinate_type(initial))
    T <: AbstractFloat || throw(ArgumentError("Coordinates must be real floating-point values"))
    positions = [Vector{T}(x) for x in initial]
    all(x -> all(isfinite, x), positions) || throw(ArgumentError("Coordinates must be finite"))
    moves = move isa MoveMixture ? map(m -> prepare_move(m, T, d), move.moves) :
        (prepare_move(move, T, d),)
    length(moves) <= _PROPOSALS_PER_PURPOSE || throw(ArgumentError("Too many mixture components"))
    n = length(positions)
    n >= maximum(m -> minimum_walkers(m, d), moves) ||
        throw(ArgumentError("Too few walkers for the dimension and selected moves"))
    moves = map(m -> _allocate_move(m, positions), moves)
    centered = reduce(hcat, [x .- first(positions) for x in positions])
    rank(centered; rtol = max(size(centered)...) * eps(T)) == d ||
        throw(ArgumentError("Initial walkers must have full affine rank"))
    ids = collect(Int, walker_ids)
    length(ids) == n && length(unique(ids)) == n &&
        all(id -> 1 <= id <= typemax(Int32) - 2, ids) ||
        throw(ArgumentError("Walker IDs must be distinct positive stream indices"))
    logds = _promoted_values(map(x -> _checked_logdensity(logdensity, x), positions))
    all(isfinite, logds) || throw(ArgumentError("Initial log densities must be finite"))
    owned_rng = copy(rng)
    cycle_part = RNGPartition(owned_rng, 0:(typemax(Int16) - 2))
    weights = move isa MoveMixture ? copy(move.weights) : nothing
    return EnsembleState(
        logdensity, moves, weights, _prepare_executor(executor, length(moves)), owned_rng, cycle_part,
        [rngpart_createrng(typeof(rng)) for _ in 1:n], ids, sortperm(ids),
        positions, deepcopy(positions), logds, copy(logds),
        _batch_workspace(logdensity, positions, logds), fill(false, n),
        zeros(Int, length(moves)), zeros(Int, length(moves)), 0, 0, true,
    )
end

function initialize(rng::Union{Philox4x{UInt64},Threefry4x{UInt64}}, logdensity,
    initial::AbstractMatrix; executor=SerialExecutor(), kwargs...)
    Base.require_one_based_indexing(initial)
    executor isa KernelExecutor && return _initialize_kernel(rng, logdensity, initial; kwargs...)
    return initialize(rng, logdensity, eachcol(initial); executor, kwargs...)
end

function _checked_logdensity(f, x)
    value = f(x)
    value isa Real || throw(ArgumentError("Log density must return a real number"))
    (isnan(value) || value == Inf) && throw(DomainError(value, "Invalid log density"))
    return float(value)
end

Base.@inline function _accept_candidate!(state, i, log_hastings, logd, acceptance_part)
    stored_logd = convert(eltype(state.logdensities), logd)
    (isnan(stored_logd) || stored_logd == Inf) &&
        throw(DomainError(stored_logd, "Invalid stored log density"))
    state.candidate_logdensities[i] = stored_logd
    T = eltype(state.positions[i])
    logratio = convert(T, log_hastings + stored_logd - state.logdensities[i])
    probability = isnan(logratio) ? zero(T) : clamp(exp(logratio), zero(T), one(T))
    rng = state.walker_rngs[i]
    set_rng!(rng, acceptance_part, state.walker_ids[i])
    state.accepted[i] = rand(rng, T) < probability
    return nothing
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

function _groups(move, part, proposal_idx, order)
    rng = AbstractRNG(part, _stream_index(4, proposal_idx))
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
        state.accepted[i] = false
        return nothing
    end
    return _evaluate_candidate!(state.batch_workspace, state, i, log_hastings, acceptance_part)
end

function _evaluate_group!(::SerialExecutor, state, move, part, idx, group, complement, acceptance_part)
    for i in group
        _evaluate!(state, move, part, idx, i, complement, acceptance_part)
    end
end
function _evaluate_chunk!(task, ntasks, state, move, part, idx, group, complement, acceptance_part)
    n = length(group)
    for k in (fld((task - 1) * n, ntasks) + 1):fld(task * n, ntasks)
        _evaluate!(state, move, part, idx, group[k], complement, acceptance_part)
    end
end

function _evaluate_chunks!(ntasks, state, move, part, idx, group, complement, acceptance_part)
    ntasks == 1 && return _evaluate_group!(SerialExecutor(), state, move, part,
        idx, group, complement, acceptance_part)
    @sync for task in Base.OneTo(ntasks)
        Threads.@spawn _evaluate_chunk!($task, $ntasks, state, move, part,
            idx, group, complement, acceptance_part)
    end
end

function _evaluate_group!(executor::_ThreadedExecutor, state, move, part, idx,
    group, complement, acceptance_part)
    schedule = executor.schedules[idx]
    sample = schedule.samples
    ntasks = sample < 4 ? 1 : min(schedule.ntasks, length(group))
    # Warm both execution paths before measuring. Use minima to exclude GC and
    # scheduler pauses from the estimate of useful work per walker.
    measuring = sample in (2, 3, 5, 6, 7)
    start = measuring ? time_ns() : UInt64(0)
    _evaluate_chunks!(ntasks, state, move, part, idx, group, complement, acceptance_part)
    sample == 8 && return nothing
    if measuring
        elapsed = (time_ns() - start) / length(group)
        if sample < 4
            schedule.serial_ns = min(schedule.serial_ns, elapsed)
        else
            schedule.threaded_ns = min(schedule.threaded_ns, elapsed)
        end
    end
    schedule.samples = sample + 1
    if sample == 3
        # Target at least 50 microseconds of serial work per task, then confirm
        # that the chosen task count actually beats serial execution.
        schedule.ntasks = clamp(floor(Int, schedule.serial_ns * length(group) / 50_000),
            1, min(Threads.nthreads(:default), length(group)))
        schedule.ntasks == 1 && (schedule.samples = 8)
    elseif sample == 7 && schedule.threaded_ns >= 0.8schedule.serial_ns
        schedule.ntasks = 1
    end
    return nothing
end

Base.@inline function _evaluate_candidate!(::Nothing, state, i, log_hastings, acceptance_part)
    logd = _checked_logdensity(state.logdensity, state.candidates[i])
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
    fill!(values, NaN)
    state.logdensity.batch!(values, positions)
    k = 0
    for i in group
        state.accepted[i] || continue
        k += 1
        _accept_candidate!(state, i, workspace.log_hastings[i], values[k], acceptance_part)
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
An exception during a sweep invalidates the state. Initialize a fresh state
after correcting the target instead of resuming a partially completed sweep.
"""
step!(state::EnsembleState) = _step!(state, state.batch_workspace)
_step!(state, workspace) = _step!(state)
function _step!(state)
    state.valid || throw(ArgumentError("Cannot resume a state after a failed sweep"))
    state.step < typemax(Int32) - 2 || throw(ArgumentError("RNG step range exhausted"))
    state.valid = false
    set_rng!(state.rng, state.cycle_partition, 0)
    steps = RNGPartition(state.rng, 0:(typemax(Int32) - 2))
    set_rng!(state.rng, steps, state.step)
    part = RNGPartition(state.rng, Base.OneTo(6 * _PROPOSALS_PER_PURPOSE))
    selection_rng = AbstractRNG(part, _stream_index(1, 1))
    idx = _select_move(state.weights, selection_rng, state.step + 1)
    move = state.moves[idx]
    groups = _groups(move, part, idx, state.walker_order)
    acceptance_part = _walker_rngpart(part, 3, idx)
    for active in eachindex(groups)
        group = groups[active]
        complement = _complement(groups, active)
        fitted_move = _prepare_group!(move, state, group, complement)
        _evaluate_group!(state.executor, state, fitted_move, part, idx, group,
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
    sample!(state, nsweeps)

Advance and collect complete sweeps. Positions have axes (coordinate, walker,
sweep). Rejections repeat the current state. Returned arrays own their storage.
Each state represents one coupled ensemble, not independent walker chains.

Return a named tuple with `positions`, `logdensities` and `accepted` (the latter
two indexed by walker and sweep), `move_indices` (one move index per sweep), and
`walker_ids` (one ID per walker). The initial positions are not included.
"""
function sample!(state::EnsembleState, nsweeps::Integer)
    nsweeps >= 0 || throw(ArgumentError("Sweep count must be nonnegative"))
    state.valid || throw(ArgumentError("Cannot resume a state after a failed sweep"))
    history = _allocate_history(state, nsweeps, state.batch_workspace)
    for sweep in 1:nsweeps
        step!(state)
        _store_history!(history, state, sweep, state.batch_workspace)
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
    for i in eachindex(state.positions)
        history.positions[:, i, sweep] .= state.positions[i]
    end
    history.logdensities[:, sweep] .= state.logdensities
    history.accepted[:, sweep] .= state.accepted
    history.move_indices[sweep] = state.active_index
end
