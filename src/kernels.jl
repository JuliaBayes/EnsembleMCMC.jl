using KernelAbstractions: @groupsize, @localmem, @synchronize

KA.@kernel function _propose_linear_kernel!(
    candidates,
    positions,
    controls,
    factors,
    logh,
    valid,
    indices,
    accepted,
    ::StretchMove,
)
    coordinate, j = @index(Global, NTuple)
    walker = @inbounds controls[1, j]
    companion = @inbounds controls[2, j]
    stretch = @inbounds factors[1, j]
    @inbounds candidates[coordinate, walker] = positions[coordinate, companion] +
        stretch * (positions[coordinate, walker] - positions[coordinate, companion])
    if isone(coordinate)
        @inbounds logh[j] = (size(positions, 1) - 1) * log(stretch)
        @inbounds valid[j] = true
        @inbounds indices[j] = j
        @inbounds accepted[walker] = false
    end
end

KA.@kernel function _propose_linear_kernel!(
    candidates,
    positions,
    controls,
    factors,
    logh,
    valid,
    indices,
    accepted,
    ::DEMove,
)
    coordinate, j = @index(Global, NTuple)
    walker = @inbounds controls[1, j]
    companion_a = @inbounds controls[2, j]
    companion_b = @inbounds controls[3, j]
    gamma = @inbounds factors[1, j]
    @inbounds candidates[coordinate, walker] = positions[coordinate, walker] +
        gamma * (positions[coordinate, companion_a] - positions[coordinate, companion_b])
    if isone(coordinate)
        @inbounds logh[j] = zero(gamma)
        @inbounds valid[j] = true
        @inbounds indices[j] = j
        @inbounds accepted[walker] = false
    end
end

KA.@kernel function _propose_snooker_kernel!(
    candidates,
    positions,
    controls,
    logh,
    valid,
    accepted,
    candidate_logdensities,
    logdensities,
    acceptance_probabilities,
    move::DESnookerMove,
)
    j = @index(Global, Linear)
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
            dot_a = zero(T)
            dot_b = zero(T)
            for coordinate in 1:dimension
                direction = @inbounds(
                    positions[coordinate, walker] - positions[coordinate, reference]
                ) / old_norm
                dot_a += direction * @inbounds(positions[coordinate, companion_a])
                dot_b += direction * @inbounds(positions[coordinate, companion_b])
            end
            displacement = move.scale * (dot_a - dot_b)
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
KA.@kernel function _compact_kernel!(indices, status, valid, n)
    counts = @localmem Int (_SCAN_WIDTH,)
    t = @index(Local, Linear)
    chunk = cld(n, @groupsize()[1])
    count = 0
    for j in ((t - 1) * chunk + 1):min(t * chunk, n)
        count += @inbounds valid[j]
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
    chunk = cld(n, @groupsize()[1])
    offset = @inbounds counts[t]
    for j in ((t - 1) * chunk + 1):min(t * chunk, n)
        if @inbounds valid[j]
            offset += 1
            @inbounds indices[offset] = j
        end
    end
end

KA.@kernel function _gather_kernel!(batchpositions, candidates, controls, indices)
    coordinate, k = @index(Global, NTuple)
    j = @inbounds indices[k]
    walker = @inbounds controls[1, j]
    @inbounds batchpositions[coordinate, k] = candidates[coordinate, walker]
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
    factors,
    indices,
    logh,
    values,
)
    k = @index(Global, Linear)
    j = @inbounds indices[k]
    walker = @inbounds controls[1, j]
    candidate_logdensity = @inbounds values[k]
    # A NaN candidate is a rejection. +Inf fails the sweep in `_accepted_count`.
    isnan(candidate_logdensity) && (candidate_logdensity = oftype(candidate_logdensity, -Inf))
    bad = candidate_logdensity == Inf
    @inbounds invalid[walker] = bad
    @inbounds candidate_logdensities[walker] = candidate_logdensity
    T = eltype(positions)
    logratio = convert(
        T,
        @inbounds(logh[j]) + candidate_logdensity - @inbounds(logdensities[walker]),
    )
    probability = isnan(logratio) ? zero(T) : clamp(exp(logratio), zero(T), one(T))
    acceptance_probabilities[walker] = probability
    accept = !bad && @inbounds(factors[2, j]) < probability
    @inbounds accepted[walker] = accept
    if accept
        for coordinate in 1:size(positions, 1)
            @inbounds positions[coordinate, walker] = candidates[coordinate, walker]
        end
        @inbounds logdensities[walker] = candidate_logdensity
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
