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

Task overhead can make cheap targets slower than [`SerialExecutor`](@ref).
Raise `min_chunk` to reduce the number of tasks. Compare full sampling runs
with the target and hardware you intend to use.
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
Loading Reactant and CUDA also enables single-device Reactant PJRT arrays on CUDA GPUs.
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
Zero-weight components keep their indices but do not increase the required
walker count.
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
            # Selection draws and cumulative sums use the normalized weights' precision.
            T = promote_type(Float64, eltype(values))
            scaled = values ./ convert(T, maximum(values))
            scaled ./ sum(scaled)
        else
            throw(ArgumentError("Mixture schedule must be :random or :cycle"))
        end
        new{typeof(ms),typeof(ws)}(ms, ws)
    end
end

"""
    minimum_walkers(move, dimension)

Return the minimum walker count for a built-in move or [`MoveMixture`](@ref)
in a positive dimension. Only positive-weight mixture components contribute.
This is a validity requirement, not a recommended population or a convergence
guarantee. [`initialize`](@ref) also checks finite positions and affine rank,
and rechecks the requirement after converting move parameters to the coordinate
precision.
"""
function minimum_walkers(move::Union{AbstractEnsembleMove,MoveMixture}, dimension::Integer)
    dimension > 0 || throw(ArgumentError("Dimension must be positive"))
    return _minimum_walkers(move, dimension)
end

_minimum_walkers(move::MoveMixture, dimension) =
    _minimum_walkers(move.moves, move.weights, dimension)
_minimum_walkers(moves::Tuple, weights, dimension) = maximum(
    _minimum_walkers(moves[i], dimension) for
    i in eachindex(moves) if isnothing(weights) || weights[i] > 0
)

const _PROPOSALS_PER_PURPOSE = typemax(Int16) - 2

mutable struct EnsembleState{F,M,W,E,R,V,L,B,A,Q,D}
    logdensity::F
    moves::M
    weights::W
    executor::E
    rng::R
    walker_ids::Vector{Int}
    walker_order::Vector{Int}
    draw_buffers::D
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
_allocate_move(move, positions, L) = move
_prepare_group!(move, state, group, complement, streams) = move
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
coordinate type `T`, after uniform power-of-two rescaling. Singular values at or
below `max(d, n) * eps(T)` times the largest one count as zero, as in
`LinearAlgebra.rank`. A full-rank ensemble whose
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
    # Bound centering and factorization without rounding normal significands.
    _, exponent = frexp(maximum(abs, centered))
    centered .= ldexp.(centered, -exponent)
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

"""
    initialize(rng, logdensity, initial; move=StretchMove(), executor=SerialExecutor(),
               walker_ids=1:nwalkers, logdensities=nothing)

Create one ensemble from a vector of finite coordinate vectors or a matrix
with axes (coordinate, walker). The coordinates must have full affine rank
and initial log densities other than NaN or +Inf. Initial values of -Inf are
allowed with a warning: stretch, DE and snooker walkers far outside the support
may never recover. The state copies the coordinates and retains a run key.

Pass a PureRNGs key such as `Philox4x64(42)`. The same key and initial points
reproduce a run within a fixed environment. Its bit position does not enter
the sweep address. Use `subrng(root, chain_id)` for independent ensembles.
Initialization leaves the supplied key untouched.

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
    rng::AbstractPureRNG, logdensity, initial::AbstractVector;
    executor::AbstractExecutor = SerialExecutor(), kwargs...,
)
    executor isa KernelExecutor && throw(ArgumentError("KernelExecutor requires an initial matrix"))
    prepared = _prepare_ensemble(logdensity, initial; kwargs...)
    (; positions, logdensities, weights, walker_ids, walker_order) = prepared
    moves = map(m -> _allocate_move(m, positions, eltype(logdensities)), prepared.moves)
    d, n = length(first(positions)), length(positions)
    T = eltype(first(positions))
    A = _acceptance_type(T, eltype(logdensities))
    return EnsembleState(logdensity, moves, weights, executor, _cpu_rng(rng), walker_ids, walker_order,
        _draw_buffers(T, A, walker_ids, d, moves),
        positions, deepcopy(positions), logdensities, copy(logdensities),
        _batch_workspace(logdensity, positions, logdensities, moves, weights),
        fill(false, n), zeros(A, n),
        zeros(Int, length(moves)), zeros(Int, length(moves)), 0, 0, true)
end

function _prepare_ensemble(logdensity, initial; move=StretchMove(),
    walker_ids = collect(eachindex(initial)),
    logdensities::Union{Nothing,AbstractVector} = nothing, kwargs...,
)
    isempty(kwargs) || throw(ArgumentError(
        "Unsupported keyword argument(s): $(join(keys(kwargs), ", "))"))
    move isa Union{AbstractEnsembleMove,MoveMixture} || throw(ArgumentError(
        "move must be an ensemble move or a MoveMixture, not $(typeof(move))"))
    validate_positions(initial)
    d = length(first(initial))
    T = float(_coordinate_type(initial))
    positions = [Vector{T}(x) for x in initial]
    moves = move isa MoveMixture ? map(m -> prepare_move(m, T, d), move.moves) :
        (prepare_move(move, T, d),)
    length(moves) <= _PROPOSALS_PER_PURPOSE || throw(ArgumentError("Too many mixture components"))
    n = length(positions)
    required = _minimum_walkers(moves, move isa MoveMixture ? move.weights : nothing, d)
    n >= required || throw(ArgumentError(
        "Too few walkers: $n given, the selected moves need $required in $d dimensions"))
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
    weights = move isa MoveMixture ? copy(move.weights) : nothing
    return (; positions, logdensities=logds, moves, weights, walker_ids=ids, walker_order=sortperm(ids))
end

function initialize(rng::AbstractPureRNG, logdensity,
    initial::AbstractMatrix; executor=SerialExecutor(), kwargs...)
    Base.require_one_based_indexing(initial)
    executor isa KernelExecutor && return _initialize_kernel(rng, logdensity, initial; kwargs...)
    return initialize(rng, logdensity, eachcol(initial); executor, kwargs...)
end

function initialize(rng::AbstractPureRNG, logdensity, initial; kwargs...)
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

# Float16 uniforms have only 2,048 possible values, including zero.
_acceptance_type(T, L) = promote_type(Float32, T, L)

@inline function _acceptance_probability(::Type{T}, proposed, current, log_hastings) where {T}
    L = promote_type(T, typeof(log_hastings))
    ratio = convert(T, (convert(L, proposed) - current) + log_hastings)
    return isnan(ratio) ? zero(T) : clamp(exp(ratio), zero(T), one(T))
end

Base.@inline function _accept_candidate!(state, i, log_hastings, logd, acceptance_part, batched=false)
    stored_logd = convert(eltype(state.logdensities), logd)
    if isnan(stored_logd)
        stored_logd = convert(eltype(state.logdensities), -Inf)
    elseif stored_logd == Inf
        _invalid_candidate(state, i, stored_logd, batched)
    end
    state.candidate_logdensities[i] = stored_logd
    T = eltype(state.acceptance_probabilities)
    probability = _acceptance_probability(T, stored_logd, state.logdensities[i], log_hastings)
    state.acceptance_probabilities[i] = probability
    state.accepted[i] = _uniform_at(acceptance_part, T, state.draw_buffers.slots[i]) < probability
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
    return _weighted_move(weights, _uniform(rng, T) * sum(weights))
end

function _weighted_move(weights::AbstractVector{T}, u) where {T}
    total = zero(T)
    last_positive = firstindex(weights)
    for i in eachindex(weights)
        total += weights[i]
        u < total && return i
        weights[i] > 0 && (last_positive = i)
    end
    return last_positive
end

function _draw_buffers(::Type{T}, ::Type{A}, ids, d, moves) where {T,A}
    n = length(ids)
    buffered = T <: _HardwareFloat && A <: _HardwareFloat
    sparse_ids = buffered && maximum(ids) != n ? ids : nothing
    # A sparse buffer uses one slot per stored walker. Its draw still uses the original ID.
    slots = isnothing(sparse_ids) ? ids : eachindex(ids)
    lanes = buffered ? max(1, maximum(_companion_lanes, moves)) : 1
    words = Vector{UInt64}(undef, Base.checked_mul(n, lanes))
    floats = if buffered
        width = maximum(move -> _scale_width(move, d), moves)
        block = isnothing(sparse_ids) ? nothing : Val(min(width, _NORMAL_BLOCK_SIZE))
        (scale=Vector{T}(undef, Base.checked_mul(n, width)),
            acceptance=Vector{A}(undef, n), ids=sparse_ids, block=block)
    end
    return (; permutation=similar(ids), words, floats, slots)
end

_draw_streams!(keys, words, ::Nothing, move) = (; keys..., words=nothing, ids=nothing)
function _draw_streams!(keys, words, floats, move)
    isnothing(floats.ids) || return _sparse_draw_streams!(keys, words, floats, move)
    n = length(floats.acceptance)
    lanes = _companion_lanes(move)
    if lanes > 0
        PureRNGs.rand_next!(keys.companion, view(words, 1:(lanes * n)))
    end
    if move isa StretchMove
        PureRNGs.rand_next!(keys.scale, view(floats.scale, 1:n))
    elseif move isa DEMove
        PureRNGs.randn_next!(keys.scale, view(floats.scale, 1:n))
    elseif move isa _AllocatedGaussianMove
        PureRNGs.randn_next!(keys.scale, floats.scale)
    end
    PureRNGs.rand_next!(keys.acceptance, floats.acceptance)
    return (; companion=keys.companion, words, scale=floats.scale, acceptance=floats.acceptance, ids=nothing)
end

function _sparse_draw_streams!(keys, words, floats, move)
    ids = floats.ids
    lanes = _companion_lanes(move)
    lanes > 0 && _sparse_draws!(PureRNGs.rand_at, words, keys.companion, ids, Val(lanes))
    if move isa StretchMove
        _sparse_draws!(PureRNGs.rand_at, floats.scale, keys.scale, ids, Val(1))
    elseif move isa DEMove
        _sparse_draws!(PureRNGs.randn_at, floats.scale, keys.scale, ids, Val(1))
    elseif move isa _AllocatedGaussianMove
        d = length(move.mean)
        if d <= _NORMAL_BLOCK_SIZE
            _sparse_draws!(PureRNGs.randn_at, floats.scale, keys.scale, ids, floats.block)
        else
            for (i, id) in pairs(ids)
                _normal_column!(keys.scale, id, view(floats.scale, ((i - 1) * d + 1):(i * d)))
            end
        end
    end
    _sparse_draws!(PureRNGs.rand_at, floats.acceptance, keys.acceptance, ids, Val(1))
    return (; companion=keys.companion, words, scale=floats.scale, acceptance=floats.acceptance, ids)
end

function _sparse_draws!(draw, values::Vector{T}, key, ids, ::Val{N}) where {T,N}
    for (i, id) in pairs(ids)
        address = UInt32(id) # initialize checks the walker ID range.
        if N == 1
            values[i] = draw(key, T, address)
        else
            # Static vector k contains scalar draws (k - 1) * N + 1 through k * N.
            draws = draw(key, SVector{N,T}, address)
            for j in 1:N
                values[(i - 1) * N + j] = draws[j]
            end
        end
    end
end

function _groups(rng, move, order, buffers)
    permutation = copyto!(buffers.permutation, order)
    n = length(permutation)
    words = buffers.words
    PureRNGs.rand_next!(rng, view(words, 1:n))
    for i in n:-1:2
        j = _bounded_index(words[i], rng, i, i)
        permutation[i], permutation[j] = permutation[j], permutation[i]
    end
    ngroups = group_count(move)
    if ngroups == 2
        split = fld(length(order), 2)
        left = view(permutation, 1:split)
        right = view(permutation, (split + 1):n)
        return isodd(words[1]) ? (left, right) : (right, left)
    end
    size, extra = divrem(length(order), ngroups)
    groups = Vector{Vector{Int}}(undef, ngroups)
    start = 1
    for i in eachindex(groups)
        stop = start + size - 1 + (i <= extra)
        groups[i] = permutation[start:stop]
        start = stop + 1
    end
    _shuffle!(subrng(rng, 0), groups)
    return groups
end

_complement(groups::Tuple, i) = i == 1 ? (groups[2],) : (groups[1],)
_complement(groups::NTuple{4}, i) = ntuple(j -> groups[j + (j >= i)], Val(3))
_complement(groups::AbstractVector, i) = groups[eachindex(groups) .!= i]

function _evaluate!(state, move, streams, i, complement)
    log_hastings = propose!(state.candidates[i], move, state.positions, i,
        complement, streams, state.draw_buffers.slots[i])
    if isnothing(log_hastings)
        copyto!(state.candidates[i], state.positions[i])
        state.candidate_logdensities[i] = state.logdensities[i]
        state.acceptance_probabilities[i] = 0
        state.accepted[i] = false
        return nothing
    end
    return _evaluate_candidate!(state.batch_workspace, state, i, log_hastings, streams.acceptance)
end

function _evaluate_group!(::SerialExecutor, state, move, streams, group, complement)
    for i in group
        _evaluate!(state, move, streams, i, complement)
    end
end
function _evaluate_chunk!(task, ntasks, state, move, streams, group, complement)
    n = length(group)
    for k in (fld((task - 1) * n, ntasks) + 1):fld(task * n, ntasks)
        _evaluate!(state, move, streams, group[k], complement)
    end
end

function _evaluate_group!(executor::ThreadedExecutor, state, move, streams, group, complement)
    ntasks = clamp(fld(length(group), executor.min_chunk), 1, Threads.nthreads(:default))
    ntasks == 1 && return _evaluate_chunk!(1, 1, state, move, streams, group, complement)
    tasks = map(Base.OneTo(ntasks)) do task
        Threads.@spawn _evaluate_chunk!(task, ntasks, state, move, streams, group, complement)
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

_evaluate_batch!(::Nothing, state, group, streams, move) = nothing
function _evaluate_batch!(workspace::BatchWorkspace, state, group, streams, move)
    count = 0
    for i in group
        if state.accepted[i]
            count += 1
            copyto!(view(workspace.positions, :, count), state.candidates[i])
        end
    end
    iszero(count) && return nothing
    values, positions = if isnothing(state.logdensity.batch_size)
        view(workspace.values, 1:count), view(workspace.positions, :, 1:count)
    else
        for j in (count + 1):size(workspace.positions, 2)
            copyto!(view(workspace.positions, :, j), view(workspace.positions, :, 1))
        end
        workspace.values, workspace.positions
    end
    # +Inf is invalid, so unfilled entries still raise an error while NaN rejects.
    fill!(values, Inf)
    state.logdensity.batch!(values, positions)
    k = 0
    for i in group
        state.accepted[i] || continue
        k += 1
        _accept_candidate!(state, i, workspace.log_hastings[i], values[k], streams.acceptance, true)
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
resuming a partially completed sweep. Argument errors raised before the sweep
starts leave the state valid.
"""
function step!(state::EnsembleState)
    return _step!(state, state.batch_workspace, nothing, nothing)
end
_step!(state, workspace, sweep, proposal_index) = _sweep!(state, sweep, proposal_index)

"""
    step!(state, sweep_key; proposal_index=1)

Advance one sweep using an external PureRNGs key. Its bit position does not
enter the sweep address. `proposal_index` in `1:$(_PROPOSALS_PER_PURPOSE)`
selects the proposal stream.
Inner mixtures use purpose 2 for selection, leaving purpose 1 for an outer mixture.
The supplied address replaces the standalone cycle/step address for this sweep only.
The sweep still counts: `sweep_count` advances, so a later standalone
`step!(state)` uses the next step address, and a cyclic mixture moves to its
next phase.
"""
function step!(state::EnsembleState, sweep_key::AbstractPureRNG; proposal_index::Integer=1)
    1 <= proposal_index <= _PROPOSALS_PER_PURPOSE ||
        throw(ArgumentError("Proposal index is outside the stream range"))
    return _step!(state, state.batch_workspace, sweep_key, Int(proposal_index))
end

_proposal_keys(sweep, i) = (companion=subrng(sweep, _COMPANION_PURPOSE, i),
    scale=subrng(sweep, _SCALE_PURPOSE, i), acceptance=subrng(sweep, _ACCEPTANCE_PURPOSE, i))

function _sweep_inputs(state, workspace, sweep_key, proposal_index)
    sweep = isnothing(sweep_key) ? subrng(state.rng, 0, state.step) : _cpu_rng(sweep_key)
    selection = isnothing(proposal_index) ? subrng(sweep, _MOVE_PURPOSE, 1) :
        subrng(sweep, _INNER_MOVE_PURPOSE, proposal_index)
    idx = _select_move(state.weights, selection, state.step + 1)
    stream_index = isnothing(proposal_index) ? idx : proposal_index
    move = state.moves[idx]
    groups = _groups(subrng(sweep, _GROUP_PURPOSE, stream_index), move, state.walker_order, state.draw_buffers)
    keys = _proposal_keys(sweep, stream_index)
    streams = _draw_streams!(keys, state.draw_buffers.words, state.draw_buffers.floats, move)
    return idx, groups, streams
end

function _sweep!(state, sweep, proposal_index)
    state.valid || throw(ArgumentError("Cannot resume a state after a failed sweep"))
    state.step < typemax(Int32) - 2 || throw(ArgumentError("RNG step range exhausted"))
    idx, groups, streams = _sweep_inputs(state, state.batch_workspace, sweep, proposal_index)
    move = state.moves[idx]
    state.valid = false

    for active in eachindex(groups)
        group = groups[active]
        complement = _complement(groups, active)
        fitted_move = _prepare_group!(move, state, group, complement, streams)
        _evaluate_group!(state.executor, state, fitted_move, streams, group, complement)
        _evaluate_batch!(state.batch_workspace, state, group, streams, fitted_move)
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
saved samples, progress output or custom storage, call [`step!`](@ref) in a loop
and keep [`snapshot`](@ref)s or read [`current_state`](@ref) after each sweep.
"""
function sample!(state::EnsembleState, nsamples::Integer; thin::Integer=1)
    _check_sample_count(nsamples, thin)
    state.valid || throw(ArgumentError("Cannot resume a state after a failed sweep"))
    history = _allocate_history(state, nsamples, state.batch_workspace)
    for sample in 1:nsamples
        step!(state, thin)
        _store_history!(history, state, sample, state.batch_workspace)
    end
    return (; history..., walker_ids=copy(state.walker_ids))
end

"""
    sample(rng, logdensity, initial, nsamples;
           warmup=0, thin=1, kwargs...)

Sample one coupled ensemble in one call. Discard `warmup` complete sweeps,
then retain `nsamples` sweeps separated by `thin` steps. Return the same owned
history as [`sample!`](@ref), with axes (coordinate, walker, sample) for positions.
The first retained sweep is `warmup + thin`. Warmup discards samples; it does
not adapt the sampler or check convergence.

The first argument must be a PureRNGs key. Use `subrng(root, run_id)` for
independent reproducible runs.

Pass `move`, `executor`, `walker_ids` and cached `logdensities` through `kwargs`
to `initialize`. Use `initialize`, [`step!`](@ref) and `sample!` directly when
continuation, cumulative diagnostics or custom storage is needed.
"""
function sample(
    rng::AbstractPureRNG,
    logdensity,
    initial,
    nsamples::Integer;
    warmup::Integer=0,
    thin::Integer=1,
    kwargs...,
)
    _check_sample_count(nsamples, thin)
    0 <= warmup <= typemax(Int) - nsamples * thin ||
        throw(ArgumentError("Warmup must be nonnegative and the total sweep count must fit in Int"))
    state = initialize(rng, logdensity, initial; kwargs...)
    step!(state, warmup)
    return sample!(state, nsamples; thin)
end

function _check_sample_count(nsamples, thin)
    nsamples >= 0 || throw(ArgumentError("Sample count must be nonnegative"))
    thin >= 1 || throw(ArgumentError("thin must be positive"))
    nsamples <= typemax(Int) ÷ thin || throw(ArgumentError("Sweep count overflows"))
    return nothing
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
    all(x -> length(x) == d && _kernel_all(isfinite, T, x), positions) ||
        throw(ArgumentError("Positions must have matching dimensions and finite coordinates"))
    _kernel_all(v -> !isnan(v) & (v != Inf), L, logdensities) ||
        throw(ArgumentError("Log densities must not contain NaN or +Inf"))
    for i in eachindex(positions)
        _kernel_copyto!(state.positions[i], positions[i])
        _kernel_copyto!(state.candidates[i], positions[i])
    end
    _kernel_copyto!(state.logdensities, logdensities)
    _kernel_copyto!(state.candidate_logdensities, logdensities)
    _kernel_fill!(state.accepted, false)
    _kernel_fill!(state.acceptance_probabilities, 0)
    state.active_index = 0
    return state
end
