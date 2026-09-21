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
        status = similar(initial, Int, 3)
        fill!(status, 0)
        workspace = KernelWorkspace(positions, candidates, similar(positions),
            similar(logds), similar(initial, T, n), similar(initial, Int, n),
            similar(initial, Int, 4, n), similar(initial, T, 2, n),
            similar(accepted), status, zeros(Int, 4, n), zeros(T, 2, n), zeros(Int, 3))
        candidate_logds = copy(logds)
        KA.synchronize(KA.get_backend(positions))
        return EnsembleState(target, moves, host.weights, KernelExecutor(),
            host.rng, host.cycle_partition, host.walker_rngs, host.walker_ids, host.walker_order,
            collect(eachcol(positions)), collect(eachcol(candidates)), logds, candidate_logds,
            workspace, accepted, host.attempts, host.accepts, 0, 0, true)
    end
end

function _step!(state, workspace::KernelWorkspace)
    return _with_kernel_device(workspace.positions) do
        try
            _step!(state)
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

function _evaluate_group!(::KernelExecutor, state, move, part, idx, group, complement, acceptance_part)
    w = state.batch_workspace
    n, d = length(group), size(w.positions, 1)
    companion_part = _walker_rngpart(part, _COMPANION_PURPOSE, idx)
    scale_part = _walker_rngpart(part, _SCALE_PURPOSE, idx)
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
            w.logh, w.valid, state.accepted, move; ndrange=n)
        _compact_kernel!(backend, 1)(w.indices, w.status, w.valid, n; ndrange=1)
        copyto!(w.host_status, 1, w.status, 1, 1)
        KA.synchronize(backend)
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
    fill!(values, NaN)
    state.logdensity.batch!(values, view(w.batchpositions, :, 1:n))
    _validate_kernel!(backend, 1)(w.status, w.values, n; ndrange=1)
    return nothing
end

function _commit_group!(state, group, w::KernelWorkspace)
    backend, n = KA.get_backend(w.positions), w.host_status[1]
    if n > 0
        _accept_commit_kernel!(backend, 64)(w.positions, w.candidates, state.logdensities,
            state.candidate_logdensities, state.accepted, w.controls, w.factors,
            w.indices, w.logh, w.values, w.status; ndrange=n)
    end
    _count_accepted_kernel!(backend, 1)(w.status, state.accepted; ndrange=1)
    copyto!(w.host_status, w.status)
    KA.synchronize(backend)
    k = w.host_status[2]
    iszero(k) || throw(DomainError(only(Array(view(w.values, k:k))), "Invalid batched log density"))
    return nothing
end
_accepted_count(state, w::KernelWorkspace) = w.host_status[3]

function _snapshot(state, w::KernelWorkspace)
    current_state(state)
    return _with_kernel_device(w.positions) do
        result = (; positions=collect(eachcol(copy(w.positions))),
            logdensities=copy(state.logdensities), accepted=copy(state.accepted),
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
