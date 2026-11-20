using KernelAbstractions: @groupsize, @localmem, @synchronize

KA.@kernel function _history_kernel!(positions, logdensities, accepted,
    current_positions, current_logdensities, current_accepted, sweep)
    i = @index(Global, Linear)
    completed = sweep[] - 1
    positions[completed * length(current_positions) + i] = current_positions[i]
    n = length(current_accepted)
    if i <= n
        dest = completed * n + i
        logdensities[dest] = current_logdensities[i]
        accepted[dest] = current_accepted[i]
    end
end

@inline function _linear_batch_geometry(move::StretchMove, complement, streams, id, permutation, walker, d)
    slot, scale = _proposal_parameters(move, complement, streams, id)
    companion = permutation[slot]
    return companion, walker, companion, scale, (d - 1) * log(scale)
end
@inline function _linear_batch_geometry(move::DEMove, complement, streams, id, permutation, walker, d)
    first, second, scale = _proposal_parameters(move, complement, streams, id)
    return walker, permutation[first], permutation[second], scale, zero(scale)
end

@inline function _draw_linear_walker!(values, controls, uniforms, logh, indices, accepted,
    permutation, walker_ids, group, complement, move, keys, k, dimension)
    # Fixed-width callbacks repeat the first candidate in all padded columns.
    j = k <= length(group) ? k : 1
    walker = permutation[group[j]]
    id = walker_ids[walker]
    streams = (; _device_streams(keys)..., words=nothing, ids=nothing)
    origin, a, b, scale, correction = _linear_batch_geometry(move, complement, streams, id,
        permutation, walker, dimension)
    values[k] = Inf
    if k == j
        controls[1, j] = walker
        uniforms[j] = _uniform_at(streams.acceptance, eltype(uniforms), id)
        logh[j] = correction
        indices[j] = j
        accepted[walker] = false
    end
    return walker, origin, a, b, scale
end

@inline function _write_linear_coordinates!(candidates, positions, batchpositions, draw,
    k, n, coordinates)
    walker, origin, a, b, scale = draw
    for coordinate in coordinates
        value = _linear_candidate(positions[coordinate, origin], scale,
            positions[coordinate, a], positions[coordinate, b])
        batchpositions[coordinate, k] = value
        if k <= n
            candidates[coordinate, walker] = value
        end
    end
end

KA.@kernel function _draw_linear_batch_kernel!(candidates, positions, batchpositions, values,
    controls, uniforms, logh, indices, accepted, permutation, walker_ids, group, complement, move, keys)
    k = @index(Global, Linear)
    draw = _draw_linear_walker!(values, controls, uniforms, logh, indices, accepted,
        permutation, walker_ids, group, complement, move, keys, k, size(positions, 1))
    _write_linear_coordinates!(candidates, positions, batchpositions, draw,
        k, length(group), axes(positions, 1))
end

# A workgroup shares one draw while its lanes traverse contiguous coordinates.
KA.@kernel function _draw_wide_linear_batch_kernel!(candidates, positions, batchpositions, values,
    controls, uniforms, logh, indices, accepted, permutation, walker_ids, group, complement, move, keys)
    draw = @localmem Tuple{Int,Int,Int,Int,eltype(positions)} (1,)
    if isone(@index(Local, Linear))
        k = @index(Group, Linear)
        draw[1] = _draw_linear_walker!(values, controls, uniforms, logh, indices, accepted,
            permutation, walker_ids, group, complement, move, keys, k, size(positions, 1))
    end
    @synchronize
    k = @index(Group, Linear)
    lane = @index(Local, Linear)
    _write_linear_coordinates!(candidates, positions, batchpositions, draw[1],
        k, length(group), lane:@groupsize()[1]:size(positions, 1))
end

KA.@kernel function _propose_snooker_kernel!(
    candidates,
    positions,
    controls,
    group,
    logh,
    valid,
    accepted,
    candidate_logdensities,
    logdensities,
    acceptance_probabilities,
    move::DESnookerMove,
)
    j = group[@index(Global, Linear)]
    walker = @inbounds controls[1, j]
    reference = @inbounds controls[2, j]
    companion_a = @inbounds controls[3, j]
    companion_b = @inbounds controls[4, j]
    dimension = size(positions, 1)
    T = eltype(positions)

    @inbounds valid[j] = false
    @inbounds accepted[walker] = false
    candidate_logdensities[walker] = logdensities[walker]
    acceptance_probabilities[walker] = zero(T)
    old_scale = zero(T)
    for coordinate in 1:dimension
        difference = @inbounds positions[coordinate, walker] - positions[coordinate, reference]
        old_scale = max(old_scale, abs(difference))
    end
    if !iszero(old_scale) && isfinite(old_scale)
        old_sum = zero(T)
        for coordinate in 1:dimension
            difference = @inbounds positions[coordinate, walker] - positions[coordinate, reference]
            old_sum += abs2(difference / old_scale)
        end
        old_norm = old_scale * sqrt(old_sum)
        if !iszero(old_norm) && isfinite(old_norm)
            projection = zero(T)
            for coordinate in 1:dimension
                direction = @inbounds(
                    positions[coordinate, walker] - positions[coordinate, reference]
                ) / old_norm
                projection += _projected_difference(direction,
                    @inbounds(positions[coordinate, companion_a]),
                    @inbounds(positions[coordinate, companion_b]),
                )
            end
            displacement = move.scale * projection
            new_scale = zero(T)
            for coordinate in 1:dimension
                direction = @inbounds(
                    positions[coordinate, walker] - positions[coordinate, reference]
                ) / old_norm
                candidate = @inbounds positions[coordinate, walker] + direction * displacement
                @inbounds candidates[coordinate, walker] = candidate
                difference = candidate - @inbounds(positions[coordinate, reference])
                new_scale = max(new_scale, abs(difference))
            end
            if !iszero(new_scale) && isfinite(new_scale)
                new_sum = zero(T)
                for coordinate in 1:dimension
                    difference = @inbounds(
                        candidates[coordinate, walker] - positions[coordinate, reference]
                    )
                    new_sum += abs2(difference / new_scale)
                end
                new_norm = new_scale * sqrt(new_sum)
                if !iszero(new_norm) && isfinite(new_norm)
                    @inbounds logh[j] = (dimension - 1) * (log(new_norm) - log(old_norm))
                    @inbounds valid[j] = true
                end
            end
        end
    end
    if !valid[j]
        for coordinate in 1:dimension
            candidates[coordinate, walker] = positions[coordinate, walker]
        end
    end
end

# Upper bound for the workgroup size of the single-group scan and reduction kernels.
const _SCAN_WIDTH = 256

# One workgroup: each item scans a contiguous chunk, so the indices stay ascending.
KA.@kernel function _compact_kernel!(indices, status, valid, slots)
    counts = @localmem Int (_SCAN_WIDTH,)
    t = @index(Local, Linear)
    chunk = cld(length(slots), @groupsize()[1])
    count = 0
    for j in ((t - 1) * chunk + 1):min(t * chunk, length(slots))
        count += valid[slots[j]]
    end
    @inbounds counts[t] = count
    @synchronize
    t = @index(Local, Linear)
    if isone(t)
        total = 0
        for s in 1:@groupsize()[1]
            count = @inbounds counts[s]
            @inbounds counts[s] = total
            total += count
        end
        @inbounds status[1] = total
    end
    @synchronize
    t = @index(Local, Linear)
    chunk = cld(length(slots), @groupsize()[1])
    offset = @inbounds counts[t]
    for j in ((t - 1) * chunk + 1):min(t * chunk, length(slots))
        slot = slots[j]
        if valid[slot]
            offset += 1
            indices[offset] = slot
        end
    end
end

KA.@kernel function _gather_kernel!(batchpositions, values, candidates, controls, indices, n)
    coordinate, k = @index(Global, NTuple)
    j = @inbounds indices[k <= n ? k : 1]
    walker = @inbounds controls[1, j]
    @inbounds batchpositions[coordinate, k] = candidates[coordinate, walker]
    # An unwritten value stays +Inf and fails the sweep.
    isone(coordinate) && (values[k] = Inf)
end

KA.@kernel function _accept_commit_kernel!(
    positions,
    candidates,
    logdensities,
    candidate_logdensities,
    accepted,
    acceptance_probabilities,
    invalid,
    controls,
    uniforms,
    indices,
    logh,
    values,
    n,
)
    k = @index(Global, Linear)
    if k <= n
        j = @inbounds indices[k]
        walker = @inbounds controls[1, j]
        candidate_logdensity = @inbounds values[k]
        # A NaN candidate is a rejection. +Inf fails the sweep in `_accepted_count`.
        isnan(candidate_logdensity) && (candidate_logdensity = oftype(candidate_logdensity, -Inf))
        bad = candidate_logdensity == Inf
        @inbounds invalid[walker] = bad
        @inbounds candidate_logdensities[walker] = candidate_logdensity
        probability = _acceptance_probability(eltype(acceptance_probabilities), candidate_logdensity,
            @inbounds(logdensities[walker]), @inbounds(logh[j]))
        acceptance_probabilities[walker] = probability
        accept = !bad && @inbounds(uniforms[j]) < probability
        @inbounds accepted[walker] = accept
        if accept
            for coordinate = 1:size(positions, 1)
                @inbounds positions[coordinate, walker] = candidates[coordinate, walker]
            end
            @inbounds logdensities[walker] = candidate_logdensity
        end
    end
end

# One workgroup: status[2] gets the first invalid walker (0 if none), status[3] the accept count.
KA.@kernel function _sweep_status_kernel!(status, accepted, invalid)
    counts = @localmem Int (_SCAN_WIDTH,)
    firsts = @localmem Int (_SCAN_WIDTH,)
    t = @index(Local, Linear)
    count, first = 0, typemax(Int)
    for walker in t:@groupsize()[1]:length(accepted)
        count += @inbounds accepted[walker]
        @inbounds(invalid[walker]) && (first = min(first, walker))
    end
    @inbounds counts[t] = count
    @inbounds firsts[t] = first
    @synchronize
    t = @index(Local, Linear)
    if isone(t)
        count, first = 0, typemax(Int)
        for s in 1:@groupsize()[1]
            count += @inbounds counts[s]
            first = min(first, @inbounds firsts[s])
        end
        @inbounds status[2] = first == typemax(Int) ? 0 : first
        @inbounds status[3] = count
    end
end
