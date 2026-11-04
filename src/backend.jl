struct KernelWorkspace{T,P,L,V,I,C,F,A,S,E}
    positions::P
    candidates::P
    batchpositions::P
    values::L
    logh::V
    indices::I
    controls::C
    factors::F
    valid::A
    invalid::A
    status::S
    host_controls::Matrix{Int}
    host_factors::Matrix{T}
    host_status::Vector{Int}
    execution::E
end

_with_kernel_device(f, initial) = f()
_kernel_synchronize(array, arrays...) = KA.synchronize(KA.get_backend(array))
_kernel_execution(initial) = nothing
Base.@inline _kernel_execute(::Nothing, f::F, args...) where {F} = f(args...)
_kernel_scalar(::Nothing, value) = value
_host_scalar(value) = value
_kernel_fill!(array, value) = fill!(array, value)
_kernel_copyto!(dest, src) = copyto!(dest, src)
_kernel_all(f, ::Type{T}, array) where {T} = all(x -> f(convert(T, x)), array)

Base.@inline function _launch_kernel(kernel, ndrange, args...)
    kernel(args...; ndrange)
    return nothing
end

Base.@inline function _launch_kernel(w::KernelWorkspace, kernel::F, workgroup, ndrange, args...) where {F}
    return _kernel_execute(w.execution, _launch_kernel,
        kernel(KA.get_backend(w.positions), workgroup), ndrange, args...)
end

function _fill_batch!(values)
    fill!(values, Inf)
    return nothing
end

_kernel_move(move, initial) = move
_kernel_move_supported(move) = false
_kernel_move_supported(::Union{StretchMove,DEMove,DESnookerMove}) = true
function _check_kernel_array(initial)
    KA.get_backend(initial) isa KA.CPU ||
        throw(ArgumentError("KernelExecutor supports CPU and CUDA arrays"))
end

function _initialize_kernel(rng, target, initial; logdensities=nothing, kwargs...)
    target isa BatchedLogDensity || throw(ArgumentError("KernelExecutor requires BatchedLogDensity"))
    _check_kernel_array(initial)
    return _with_kernel_device(initial) do
        # The host state checks each cached value, which needs host storage.
        cached = isnothing(logdensities) ? nothing : Array(logdensities)
        host = initialize(rng, target.scalar, Array(initial); logdensities=cached, kwargs...)
        moves = map(m -> _kernel_move(m, initial), host.moves)
        all(_kernel_move_supported, moves) ||
            throw(ArgumentError("KernelExecutor supports only the built-in moves"))
        d, n = length(first(host.positions)), length(host.positions)
        capacity = _batch_capacity(target, n, moves, host.weights)
        T, L = eltype(first(host.positions)), eltype(host.logdensities)
        positions = similar(initial, T, d, n)
        copyto!(positions, reduce(hcat, host.positions))
        candidates = copy(positions)
        logds = similar(initial, L, n)
        copyto!(logds, host.logdensities)
        accepted = similar(initial, Bool, n)
        _kernel_fill!(accepted, false)
        probabilities = similar(initial, T, n)
        _kernel_fill!(probabilities, 0)
        status = similar(initial, Int, 3)
        _kernel_fill!(status, 0)
        workspace = KernelWorkspace(positions, candidates, similar(positions, d, capacity),
            similar(logds, capacity), similar(initial, T, n), similar(initial, Int, n),
            similar(initial, Int, 4, n), similar(initial, T, 2, n),
            similar(accepted), copy(accepted), status, zeros(Int, 4, n), zeros(T, 2, n),
            zeros(Int, 3), _kernel_execution(positions))
        candidate_logds = copy(logds)
        _kernel_synchronize(positions, candidates, logds, candidate_logds, accepted,
            probabilities, status)
        return EnsembleState(target, moves, host.weights, KernelExecutor(),
            host.rng, host.cycle_partition, host.walker_rngs, host.walker_ids, host.walker_order,
            collect(eachcol(positions)), collect(eachcol(candidates)), logds, candidate_logds,
            workspace, accepted, probabilities, host.attempts, host.accepts, 0, 0, true)
    end
end

function _step!(state, workspace::KernelWorkspace, args...)
    return _with_kernel_device(workspace.positions) do
        try
            _step!(state, args...)
        catch
            # A callback can enqueue device work before throwing.
            _kernel_synchronize(workspace.positions, workspace.candidates,
                workspace.batchpositions, workspace.values, workspace.logh,
                workspace.indices, workspace.controls, workspace.factors,
                workspace.valid, workspace.invalid, workspace.status,
                state.logdensities, state.candidate_logdensities,
                state.accepted, state.acceptance_probabilities)
            foreach(_synchronize_move, state.moves)
            rethrow()
        end
    end
end

_synchronize_move(move) = nothing

function _stage_proposal!(controls, factors, j, move::StretchMove, rng, complement,
    companion_part, scale_part, id)
    set_rng!(rng, companion_part, id)
    controls[2, j] = rand(rng, only(complement))
    set_rng!(rng, scale_part, id)
    b = (move.scale - one(move.scale)) * rand(rng, eltype(factors)) + one(move.scale)
    factors[1, j] = b * (b / move.scale)
end
function _stage_proposal!(controls, factors, j, move::DEMove, rng, complement,
    companion_part, scale_part, id)
    set_rng!(rng, companion_part, id)
    controls[2, j], controls[3, j] = _de_companion_indices(rng, only(complement))
    set_rng!(rng, scale_part, id)
    factors[1, j] = move.gamma0 * (one(move.sigma) + move.sigma * randn(rng, eltype(factors)))
end
function _stage_proposal!(controls, factors, j, move::DESnookerMove, rng, complement,
    companion_part, scale_part, id)
    set_rng!(rng, companion_part, id)
    controls[2, j], controls[3, j], controls[4, j] = _de_snooker_companion_indices(rng, complement)
end

function _evaluate_group!(::KernelExecutor, state, move, part, move_index, proposal_index,
    group, complement, acceptance_part)
    w = state.batch_workspace
    n, d = length(group), size(w.positions, 1)
    companion_part = _walker_rngpart(part, _COMPANION_PURPOSE, proposal_index)
    scale_part = _walker_rngpart(part, _SCALE_PURPOSE, proposal_index)
    for (j, i) in enumerate(group)
        rng, id = state.walker_rngs[i], state.walker_ids[i]
        w.host_controls[1, j] = i
        _stage_proposal!(w.host_controls, w.host_factors, j, move, rng, complement,
            companion_part, scale_part, id)
        set_rng!(rng, acceptance_part, id)
        w.host_factors[2, j] = rand(rng, eltype(w.host_factors))
    end
    copyto!(w.controls, 1, w.host_controls, 1, 4n)
    copyto!(w.factors, 1, w.host_factors, 1, 2n)
    if move isa DESnookerMove
        _launch_kernel(w, _propose_snooker_kernel!, 64, n, w.candidates, w.positions, w.controls,
            w.logh, w.valid, state.accepted, state.candidate_logdensities,
            state.logdensities, state.acceptance_probabilities, move)
        _compact!(w, n)
    else
        _launch_kernel(w, _propose_linear_kernel!, (32, 4), (d, n), w.candidates,
            w.positions, w.controls, w.factors, w.logh, w.valid, w.indices, state.accepted, move)
        w.host_status[1] = n
    end
    return nothing
end

function _evaluate_batch!(w::KernelWorkspace, state, group, acceptance_part)
    n = w.host_status[1]
    iszero(n) && return nothing
    fixed = !isnothing(state.logdensity.batch_size)
    width = fixed ? length(w.values) : length(group)
    count = _kernel_scalar(w.execution, n)
    _launch_kernel(w, _gather_kernel!, (32, 4), (size(w.positions, 1), width),
        w.batchpositions, w.candidates, w.controls, w.indices, count)
    values = fixed ? w.values : view(w.values, 1:n)
    positions = fixed ? w.batchpositions : view(w.batchpositions, :, 1:n)
    # An unwritten value stays +Inf and fails the sweep.
    _kernel_execute(w.execution, _fill_batch!, w.values)
    state.logdensity.batch!(values, positions)
    return nothing
end

function _compact!(w::KernelWorkspace, n)
    # A fixed width compiles the kernel once for all ensemble sizes.
    _launch_kernel(w, _compact_kernel!, _SCAN_WIDTH, _SCAN_WIDTH, w.indices, w.status, w.valid, n)
    copyto!(w.host_status, 1, w.status, 1, 1)
    _kernel_synchronize(w.status)
    return nothing
end

function _commit_group!(state, group, w::KernelWorkspace)
    n = w.host_status[1]
    n > 0 && _launch_kernel(w, _accept_commit_kernel!, 64, length(group), w.positions, w.candidates,
        state.logdensities, state.candidate_logdensities, state.accepted,
        state.acceptance_probabilities, w.invalid, w.controls, w.factors, w.indices, w.logh,
        w.values, _kernel_scalar(w.execution, n))
    return nothing
end

# Linear moves use one explicit status synchronization per sweep.
function _accepted_count(state, w::KernelWorkspace)
    _launch_kernel(w, _sweep_status_kernel!, _SCAN_WIDTH, _SCAN_WIDTH,
        w.status, state.accepted, w.invalid)
    copyto!(w.host_status, w.status)
    _kernel_synchronize(w.status)
    i = w.host_status[2]
    iszero(i) || throw(DomainError(Inf, "Invalid batched log density for walker $i " *
        "(ID $(state.walker_ids[i])) in sweep $(state.step + 1): " *
        "+Inf is not allowed and batch! must fill every value"))
    return w.host_status[3]
end

function _snapshot(state, w::KernelWorkspace)
    current_state(state)
    return _with_kernel_device(w.positions) do
        result = (; positions=collect(eachcol(copy(w.positions))),
            logdensities=copy(state.logdensities),
            candidates=collect(eachcol(copy(w.candidates))),
            candidate_logdensities=copy(state.candidate_logdensities),
            accepted=copy(state.accepted), acceptance_probabilities=copy(state.acceptance_probabilities),
            walker_ids=copy(state.walker_ids), attempts=copy(state.attempts),
            acceptances=copy(state.accepts), move_index=state.active_index, sweep_count=state.step)
        _kernel_synchronize(first(result.positions), first(result.candidates),
            result.logdensities, result.candidate_logdensities, result.accepted,
            result.acceptance_probabilities)
        result
    end
end

function _allocate_history(state, nsweeps, w::KernelWorkspace)
    return _with_kernel_device(w.positions) do
        d, n = size(w.positions)
        (; positions=similar(w.positions, d, n, nsweeps),
            logdensities=similar(state.logdensities, n, nsweeps),
            accepted=similar(state.accepted, n, nsweeps), move_indices=Vector{Int}(undef, nsweeps))
    end
end
function _store_history!(history, state, sweep, w::KernelWorkspace)
    _with_kernel_device(w.positions) do
        index = _kernel_scalar(w.execution, sweep)
        _kernel_execute(w.execution, _copy_kernel_history!, history.positions,
            history.logdensities, history.accepted, w.positions, state.logdensities,
            state.accepted, index)
        _kernel_synchronize(history.positions, history.logdensities, history.accepted)
    end
    history.move_indices[sweep] = state.active_index
    return nothing
end

function _copy_kernel_history!(positions, logdensities, accepted, current_positions,
    current_logdensities, current_accepted, sweep)
    positions[:, :, sweep] = current_positions
    logdensities[:, sweep] = current_logdensities
    accepted[:, sweep] = current_accepted
    return nothing
end

function _synchronize!(state, positions, logdensities, workspace::KernelWorkspace)
    return _with_kernel_device(workspace.positions) do
        _synchronize!(state, positions, logdensities, nothing)
        _kernel_synchronize(workspace.positions, workspace.candidates, state.logdensities,
            state.candidate_logdensities, state.accepted, state.acceptance_probabilities)
        state
    end
end
