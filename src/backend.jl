struct KernelWorkspace{P,L,V,I,C,F,A,S,E}
    positions::P
    candidates::P
    batchpositions::P
    values::L
    logh::V
    indices::I
    controls::C
    uniforms::F
    valid::A
    invalid::A
    status::S
    walker_ids::I
    permutation::I
    host_status::Vector{Int}
    execution::E
end

_with_kernel_device(f, initial) = f()
_kernel_synchronize(array, arrays...) = KA.synchronize(KA.get_backend(array))
_kernel_execution(initial) = nothing
Base.@inline _kernel_execute(::Nothing, f::F, args...) where {F} = f(args...)
_kernel_scalar(::Nothing, value) = value
_kernel_rng(w::KernelWorkspace, rng) = _kernel_rng(w.positions, rng)
_kernel_rng(positions, rng) = _cpu_rng(rng)
_device_rng(rng::AbstractPureRNG) = rng
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
        # Rank and initial scalar log densities are checked once on the host.
        cached = isnothing(logdensities) ? nothing : Array(logdensities)
        host = _prepare_ensemble(target.scalar, eachcol(Array(initial)); logdensities=cached, kwargs...)
        d, n = size(initial)
        T, L = eltype(first(host.positions)), eltype(host.logdensities)
        positions = similar(initial, T, d, n)
        moves = map(m -> _kernel_move(m, positions), host.moves)
        all(_kernel_move_supported, moves) ||
            throw(ArgumentError("KernelExecutor supports only the built-in moves"))
        capacity = _batch_capacity(target, n, moves, host.weights)
        copyto!(positions, reduce(hcat, host.positions))
        candidates = copy(positions)
        logds = similar(initial, L, n)
        copyto!(logds, host.logdensities)
        accepted = similar(initial, Bool, n)
        _kernel_fill!(accepted, false)
        probabilities = similar(initial, _acceptance_type(T, L), n)
        _kernel_fill!(probabilities, 0)
        status = similar(initial, Int, 3)
        _kernel_fill!(status, 0)
        workspace = KernelWorkspace(positions, candidates, similar(positions, d, capacity),
            similar(logds, capacity), similar(initial, T, n), similar(initial, Int, n),
            similar(initial, Int, 4, n), similar(probabilities),
            similar(accepted), copy(accepted), status,
            _kernel_vector(initial, host.walker_ids), similar(initial, Int, n),
            zeros(Int, 3), _kernel_execution(positions))
        candidate_logds = copy(logds)
        _kernel_synchronize(positions, candidates, logds, candidate_logds, accepted,
            probabilities, status)
        key = _kernel_rng(positions, rng)
        return EnsembleState(target, moves, host.weights, KernelExecutor(),
            key, host.walker_ids, host.walker_order, _kernel_draw_buffers(positions, host, key),
            collect(eachcol(positions)), collect(eachcol(candidates)), logds, candidate_logds,
            workspace, accepted, probabilities, zeros(Int, length(moves)), zeros(Int, length(moves)), 0, 0, true)
    end
end

function _step!(state, workspace::KernelWorkspace, sweep, proposal_index)
    return _with_kernel_device(workspace.positions) do
        try
            _sweep!(state, sweep, proposal_index)
        catch
            # A callback can enqueue device work before throwing.
            _kernel_synchronize(workspace.positions, workspace.candidates,
                workspace.batchpositions, workspace.values, workspace.logh,
                workspace.indices, workspace.controls, workspace.uniforms,
                workspace.valid, workspace.invalid, workspace.status,
                state.logdensities, state.candidate_logdensities,
                state.accepted, state.acceptance_probabilities)
            foreach(_synchronize_move, state.moves)
            rethrow()
        end
    end
end

_synchronize_move(move) = nothing

KA.@kernel function _snooker_controls_kernel!(controls, uniforms, permutation, walker_ids,
    groups, move, keys)
    j = @index(Global, Linear)
    walker = permutation[j]
    id = walker_ids[walker]
    active = 1
    while active < length(groups) && !(j in groups[active])
        active += 1
    end
    streams = _device_streams(keys)
    r, a, b = _proposal_parameters(move, groups, (; streams..., words=nothing, ids=nothing), id, active)
    controls[1, j] = walker
    controls[2, j], controls[3, j], controls[4, j] = permutation[r], permutation[a], permutation[b]
    uniforms[j] = _uniform_at(streams.acceptance, eltype(uniforms), id)
end

function _evaluate_group!(::KernelExecutor, state, move::DESnookerMove, streams, group, complement)
    w = state.batch_workspace
    n = length(group)
    _launch_kernel(w, _propose_snooker_kernel!, 64, n, w.candidates, w.positions, w.controls, group,
        w.logh, w.valid, state.accepted, state.candidate_logdensities,
        state.logdensities, state.acceptance_probabilities, move)
    _compact!(w, group)
    return nothing
end

function _evaluate_group!(::KernelExecutor, state, move::Union{StretchMove,DEMove}, keys, group, complement)
    w = state.batch_workspace
    n = length(group)
    width = isnothing(state.logdensity.batch_size) ? n : length(w.values)
    # Wide columns share a draw across 32 lanes. Shared memory requires an isbits element type.
    kernel, threads, count = size(w.positions, 1) >= 32 && isbitstype(eltype(w.positions)) ?
        (_draw_wide_linear_batch_kernel!, 32, 32 * width) : (_draw_linear_batch_kernel!, 64, width)
    _launch_kernel(w, kernel, threads, count, w.candidates, w.positions,
        w.batchpositions, w.values, w.controls, w.uniforms, w.logh, w.indices, state.accepted,
        w.permutation, w.walker_ids, group, complement, move, keys)
    w.host_status[1] = n
    return nothing
end

function _evaluate_batch!(w::KernelWorkspace, state, group, streams, move)
    n = w.host_status[1]
    iszero(n) && return nothing
    fixed = !isnothing(state.logdensity.batch_size)
    width = fixed ? length(w.values) : length(group)
    count = _kernel_scalar(w.execution, n)
    _launch_kernel(w, _gather_kernel!, (32, 4), (size(w.positions, 1), width),
        w.batchpositions, w.values, w.candidates, w.controls, w.indices, count)
    return _call_batch!(w, state, n, fixed)
end

function _evaluate_batch!(w::KernelWorkspace, state, group, streams, ::Union{StretchMove,DEMove})
    return _call_batch!(w, state, length(group), !isnothing(state.logdensity.batch_size))
end

function _call_batch!(w, state, n, fixed)
    values = fixed ? w.values : view(w.values, 1:n)
    positions = fixed ? w.batchpositions : view(w.batchpositions, :, 1:n)
    state.logdensity.batch!(values, positions)
    return nothing
end

function _compact!(w::KernelWorkspace, slots)
    # A fixed width compiles the kernel once for all ensemble sizes.
    _launch_kernel(w, _compact_kernel!, _SCAN_WIDTH, _SCAN_WIDTH, w.indices, w.status, w.valid, slots)
    copyto!(w.host_status, 1, w.status, 1, 1)
    _kernel_synchronize(w.status)
    return nothing
end

function _commit_group!(state, group, w::KernelWorkspace)
    n = w.host_status[1]
    n > 0 && _launch_kernel(w, _accept_commit_kernel!, 64, length(group), w.positions, w.candidates,
        state.logdensities, state.candidate_logdensities, state.accepted,
        state.acceptance_probabilities, w.invalid, w.controls, w.uniforms, w.indices, w.logh,
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
    kernel = _history_kernel!(KA.get_backend(current_positions), 128)
    kernel(positions, logdensities, accepted, current_positions, current_logdensities,
        current_accepted, sweep; ndrange=length(current_positions))
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
