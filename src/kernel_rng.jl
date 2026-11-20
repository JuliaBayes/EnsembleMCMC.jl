# Walker IDs and the initial permutation stay on the same device as the positions.
_kernel_vector(initial, values) = copyto!(similar(initial, eltype(values), length(values)), values)

_shuffle_workgroup_limit(execution) = 512

_purpose_buffer(initial, key::AbstractPureRNG) = similar(initial, typeof(key), 4)
_purpose_buffer(initial, key::T, buffer::AbstractVector{T}) where {T<:AbstractPureRNG} = buffer
_purpose_buffer(initial, key, buffer) = _purpose_buffer(initial, key)

@inline _store_purpose!(keys::AbstractVector, lane, key) = (keys[lane] = key)
@inline _purpose_key(keys::AbstractVector, lane) = keys[lane]

function _kernel_draw_buffers(initial, host, key)
    # Reactant kernel arguments need storage even when move selection skips weights.
    weights = host.weights isa AbstractVector{<:AbstractFloat} ?
        convert(Vector{eltype(host.weights)}, host.weights) : Float32[0]
    if !(eltype(weights) <: _HardwareFloat) && !(KA.get_backend(initial) isa KA.CPU)
        throw(ArgumentError("GPU random mixtures require Float16, Float32, or Float64 weights"))
    end
    counts = collect(Int, group_count.(host.moves))
    return (; order=_kernel_vector(initial, host.walker_order),
        weights=_kernel_vector(initial, weights), weight_sum=sum(weights),
        group_counts=_kernel_vector(initial, counts),
        schedule=similar(initial, Int, 5), host_schedule=zeros(Int, 5),
        purposes=_purpose_buffer(initial, key))
end

# Explicit sweep keys already name their sweep. A stored run key needs the sweep counter.
@inline function _device_sweep(key, step)
    root = _device_rng(key)
    counter = step[]
    return counter < 0 ? root : subrng(root, 0, counter)
end

@inline _device_streams(keys) = (companion=_purpose_key(keys, 2), scale=_purpose_key(keys, 3),
    acceptance=_purpose_key(keys, 4))

# The host dispatches the chosen move and calls the density once per active group.
# Only the move index and the order of its (at most four) groups leave the device.
KA.@kernel function _sweep_schedule_kernel!(schedule, group_counts, weights, weight_sum,
    key, step, known_move, proposal_index, purposes)
    lane = @index(Global, Linear)
    sweep = _device_sweep(key, step)
    idx = known_move[]
    outer = proposal_index[]
    if iszero(idx)
        selection = iszero(outer) ? subrng(sweep, _MOVE_PURPOSE, 1) :
            subrng(sweep, _INNER_MOVE_PURPOSE, outer)
        idx = _weighted_move(weights, _uniform(selection, eltype(weights)) * weight_sum)
    end
    purpose = (_GROUP_PURPOSE, _COMPANION_PURPOSE, _SCALE_PURPOSE, _ACCEPTANCE_PURPOSE)[lane]
    child = subrng(sweep, purpose, iszero(outer) ? idx : outer)
    _store_purpose!(purposes, lane, child)
    if lane == 1
        schedule[1] = idx
        _write_group_order!(schedule, child, group_counts[idx])
    end
end

@inline function _write_group_order!(schedule, group_key, count)
    if count == 2
        left_first = isodd(PureRNGs.rand_at(group_key, UInt64, 1))
        schedule[2] = left_first ? 1 : 2
        schedule[3] = left_first ? 2 : 1
        schedule[4] = 0
        schedule[5] = 0
    else
        for j in 1:4
            schedule[j + 1] = j
        end
        _shuffle!(subrng(group_key, 0), view(schedule, 2:5))
    end
end

@inline _group_rng(keys) = _purpose_key(keys, 1)

@inline _swap_index(rng, i) = _bounded_index(PureRNGs.rand_at(rng, UInt64, i), rng, i, i)

# Each nontrivial Fisher–Yates swap i points to its smaller partner. The first
# half of links gives each partner's first child, the second the next sibling.
@inline function _draw_swap!(parents, links, capacity, rng, i)
    parents[i] = i == 1 ? 1 : _swap_index(rng, i)
    links[i] = links[capacity + i] = typemax(eltype(links))
    return nothing
end

@inline function _link_swap!(links, capacity, parent, i)
    parent == i && return nothing
    child = eltype(links)(i)
    link = Int(parent)
    # Atomic minimum keeps the earlier child. Insert the later one into the
    # next sibling link until both appear in increasing order.
    while true
        previous = first(KA.@atomic :monotonic min(links[link], child))
        previous == typemax(eltype(links)) && break
        link = capacity + Int(min(previous, child))
        child = max(previous, child)
    end
    return nothing
end

@inline function _permuted_walker(order, parents, links, capacity, schedule, width, i)
    slot = i
    if width > 0
        group, offset = divrem(i - 1, width)
        slot = (schedule[2 + group] - 1) * width + offset + 1
    end
    origin = parents[slot]
    # Undo swaps in increasing order. After slot's swap, the next swap that
    # touches its partner is the next sibling. Each later hop takes a first child.
    next = links[origin == slot ? slot : capacity + slot]
    while next != typemax(eltype(links))
        origin = next
        next = links[origin]
    end
    return order[origin]
end

KA.@kernel function _permute_shared_kernel!(permutation, order, address, schedule,
    width, ::Val{capacity}) where {capacity}
    parents = @localmem Int32 (capacity,)
    links = @localmem Int32 (2 * capacity,)
    rng = _group_rng(address)
    for i in @index(Local, Linear):@groupsize()[1]:length(permutation)
        _draw_swap!(parents, links, capacity, rng, i)
    end
    @synchronize
    for i in @index(Local, Linear):@groupsize()[1]:length(permutation)
        _link_swap!(links, capacity, parents[i], i)
    end
    @synchronize
    for i in @index(Local, Linear):@groupsize()[1]:length(permutation)
        permutation[i] = _permuted_walker(order, parents, links, capacity, schedule, width, i)
    end
end

KA.@kernel function _draw_swaps_kernel!(parents, links, address)
    i = @index(Global, Linear)
    _draw_swap!(parents, links, length(parents), _group_rng(address), i)
end

KA.@kernel function _link_swaps_kernel!(parents, links)
    i = @index(Global, Linear)
    _link_swap!(links, length(parents), parents[i], i)
end

KA.@kernel function _permute_walkers_kernel!(permutation, order, parents, links, schedule, width)
    i = @index(Global, Linear)
    permutation[i] = _permuted_walker(order, parents, links, length(parents), schedule, width, i)
end

function _group_ranges(n, ::Val{2}, schedule)
    half = fld(n, 2)
    groups = (1:half, (half + 1):n)
    return (groups[schedule[2]], groups[schedule[3]])
end

function _group_ranges(n, ::Val{4}, schedule)
    width, extra = divrem(n, 4)
    return ntuple(Val(4)) do j
        i = schedule[j + 1]
        start = (i - 1) * width + min(i - 1, extra) + 1
        start:(start + width + (i <= extra) - 1)
    end
end

function _sweep_inputs(state, w::KernelWorkspace, sweep_key, proposal_index)
    b = state.draw_buffers
    key = isnothing(sweep_key) ? state.rng : _kernel_rng(w, sweep_key)
    step = _kernel_scalar(w.execution, isnothing(sweep_key) ? state.step : -1)
    known_move = state.weights isa AbstractVector{<:AbstractFloat} ? 0 :
        _select_move(state.weights, nothing, state.step + 1)
    n = length(w.permutation)
    purposes = isnothing(sweep_key) ? b.purposes : _purpose_buffer(w.positions, key, b.purposes)
    _launch_kernel(w, _sweep_schedule_kernel!, 4, 4, b.schedule, b.group_counts, b.weights,
        b.weight_sum, key, step, _kernel_scalar(w.execution, known_move),
        _kernel_scalar(w.execution, something(proposal_index, 0)), purposes)
    # Equal-sized groups can be laid out in sweep order on the device. The host
    # already knows the callback widths and the move for a fixed or cyclic schedule.
    width = 0
    if known_move != 0 && iszero(rem(n, group_count(state.moves[known_move])))
        idx = known_move
        width = div(n, group_count(state.moves[idx]))
        schedule = 0:4
    else
        schedule = copyto!(b.host_schedule, b.schedule)
        idx = schedule[1]
    end
    groups = group_count(state.moves[idx]) == 2 ? _group_ranges(n, Val(2), schedule) :
        _group_ranges(n, Val(4), schedule)
    if n <= 4096
        # The walker count bounds every link. Three Int32 entries per slot use at most 48 KiB.
        capacity = nextpow(2, n)
        threads = min(_shuffle_workgroup_limit(w.execution), capacity)
        _launch_kernel(w, _permute_shared_kernel!, threads, threads, w.permutation, b.order, purposes,
            b.schedule, width, Val(capacity))
    else
        # Reuse the proposal indices and controls until the first group starts.
        _launch_kernel(w, _draw_swaps_kernel!, 128, n, w.indices, w.controls, purposes)
        _launch_kernel(w, _link_swaps_kernel!, 128, n, w.indices, w.controls)
        _launch_kernel(w, _permute_walkers_kernel!, 128, n, w.permutation, b.order,
            w.indices, w.controls, b.schedule, width)
    end
    move = state.moves[idx]
    if move isa DESnookerMove
        # Partners depend only on the permutation. Geometry still uses each group's updated complement.
        _launch_kernel(w, _snooker_controls_kernel!, 64, n, w.controls, w.uniforms,
            w.permutation, w.walker_ids, groups, move, purposes)
    end
    return idx, groups, purposes
end
