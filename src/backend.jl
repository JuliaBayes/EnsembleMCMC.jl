struct KernelWorkspace{T,P,L,V,I,C,F,A,S}
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
end

_with_kernel_device(f, initial) = f()
_kernel_move(move, initial) = move
_kernel_move_supported(move) = false
_kernel_move_supported(::Union{StretchMove,DEMove,DESnookerMove}) = true
function _check_kernel_array(initial)
    KA.get_backend(initial) isa KA.CPU ||
        throw(ArgumentError("KernelExecutor supports CPU and CUDA arrays"))
end

function _initialize_kernel(rng, target, initial; kwargs...)
    target isa BatchedLogDensity || throw(ArgumentError("KernelExecutor requires BatchedLogDensity"))
    _check_kernel_array(initial)
    return _with_kernel_device(initial) do
        host = initialize(rng, target.scalar, Array(initial); kwargs...)
        moves = map(m -> _kernel_move(m, initial), host.moves)
        all(_kernel_move_supported, moves) ||
            throw(ArgumentError("KernelExecutor supports only the built-in moves"))
        d, n = length(first(host.positions)), length(host.positions)
        T, L = eltype(first(host.positions)), eltype(host.logdensities)
        positions = similar(initial, T, d, n)
        copyto!(positions, reduce(hcat, host.positions))
        candidates = copy(positions)
        logds = similar(initial, L, n)
        copyto!(logds, host.logdensities)
        accepted = similar(initial, Bool, n)
        fill!(accepted, false)
        probabilities = similar(initial, T, n)
        fill!(probabilities, 0)
        status = similar(initial, Int, 3)
        fill!(status, 0)
        workspace = KernelWorkspace(positions, candidates, similar(positions),
            similar(logds), similar(initial, T, n), similar(initial, Int, n),
            similar(initial, Int, 4, n), similar(initial, T, 2, n),
            similar(accepted), copy(accepted), status, zeros(Int, 4, n), zeros(T, 2, n),
            zeros(Int, 3))
        candidate_logds = copy(logds)
        KA.synchronize(KA.get_backend(positions))
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
            KA.synchronize(KA.get_backend(workspace.positions))
            rethrow()
        end
    end
end

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
    backend = KA.get_backend(w.positions)
    if move isa DESnookerMove
        _propose_snooker_kernel!(backend, 64)(w.candidates, w.positions, w.controls,
            w.logh, w.valid, state.accepted, state.candidate_logdensities,
            state.logdensities, state.acceptance_probabilities, move; ndrange=n)
        _compact!(w, n)
    else
        _propose_linear_kernel!(backend, (32, 4))(w.candidates, w.positions, w.controls,
            w.factors, w.logh, w.valid, w.indices, state.accepted, move; ndrange=(d, n))
        w.host_status[1] = n
    end
    return nothing
end

function _evaluate_batch!(w::KernelWorkspace, state, group, acceptance_part)
    n = w.host_status[1]
    iszero(n) && return nothing
    backend = KA.get_backend(w.positions)
    _gather_kernel!(backend, (32, 4))(w.batchpositions, w.candidates, w.controls,
        w.indices; ndrange=(size(w.positions, 1), n))
    values = view(w.values, 1:n)
    # An unwritten value stays +Inf and fails the sweep.
    fill!(values, Inf)
    state.logdensity.batch!(values, view(w.batchpositions, :, 1:n))
    return nothing
end

function _compact!(w::KernelWorkspace, n)
    backend = KA.get_backend(w.positions)
    # A fixed width compiles the kernel once for all ensemble sizes.
    _compact_kernel!(backend, _SCAN_WIDTH)(w.indices, w.status, w.valid, n; ndrange=_SCAN_WIDTH)
    copyto!(w.host_status, 1, w.status, 1, 1)
    KA.synchronize(backend)
    return nothing
end

function _commit_group!(state, group, w::KernelWorkspace)
    n = w.host_status[1]
    n > 0 && _accept_commit_kernel!(KA.get_backend(w.positions), 64)(w.positions, w.candidates,
        state.logdensities, state.candidate_logdensities, state.accepted,
        state.acceptance_probabilities, w.invalid, w.controls, w.factors, w.indices, w.logh,
        w.values; ndrange=n)
    return nothing
end

# Runs once per sweep. Linear moves need no other host wait in a sweep.
function _accepted_count(state, w::KernelWorkspace)
    backend = KA.get_backend(w.positions)
    _sweep_status_kernel!(backend, _SCAN_WIDTH)(w.status, state.accepted, w.invalid;
        ndrange=_SCAN_WIDTH)
    copyto!(w.host_status, w.status)
    KA.synchronize(backend)
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
        KA.synchronize(KA.get_backend(w.positions))
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
        copyto!(view(history.positions, :, :, sweep), w.positions)
        copyto!(view(history.logdensities, :, sweep), state.logdensities)
        copyto!(view(history.accepted, :, sweep), state.accepted)
        KA.synchronize(KA.get_backend(w.positions))
    end
    history.move_indices[sweep] = state.active_index
    return nothing
end

function _synchronize!(state, positions, logdensities, workspace::KernelWorkspace)
    return _with_kernel_device(workspace.positions) do
        _synchronize!(state, positions, logdensities, nothing)
        KA.synchronize(KA.get_backend(workspace.positions))
        state
    end
end
