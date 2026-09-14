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

KA.@kernel function _compact_kernel!(indices, status, valid, n)
    index = @index(Global, Linear)
    count = zero(eltype(status))
    for j in 1:n
        if @inbounds valid[j]
            count += one(count)
            @inbounds indices[count] = j
        end
    end
    @inbounds status[1] = count
end

KA.@kernel function _gather_kernel!(batchpositions, candidates, controls, indices)
    coordinate, k = @index(Global, NTuple)
    j = @inbounds indices[k]
    walker = @inbounds controls[1, j]
    @inbounds batchpositions[coordinate, k] = candidates[coordinate, walker]
end

KA.@kernel function _validate_kernel!(status, values, count)
    index = @index(Global, Linear)
    @inbounds status[2] = 0
    for k in 1:count
        value = @inbounds values[k]
        if isnan(value) || isinf(value) && value > zero(value)
            @inbounds status[2] = k
            break
        end
    end
end

KA.@kernel function _accept_commit_kernel!(
    positions,
    candidates,
    logdensities,
    candidate_logdensities,
    accepted,
    acceptance_probabilities,
    controls,
    factors,
    indices,
    logh,
    values,
    status,
)
    k = @index(Global, Linear)
    if iszero(@inbounds(status[2]))
        j = @inbounds indices[k]
        walker = @inbounds controls[1, j]
        candidate_logdensity = @inbounds values[k]
        @inbounds candidate_logdensities[walker] = candidate_logdensity
        T = eltype(positions)
        logratio = convert(
            T,
            @inbounds(logh[j]) + candidate_logdensity - @inbounds(logdensities[walker]),
        )
        probability = isnan(logratio) ? zero(T) : clamp(exp(logratio), zero(T), one(T))
        acceptance_probabilities[walker] = probability
        accept = @inbounds factors[2, j] < probability
        @inbounds accepted[walker] = accept
        if accept
            for coordinate in 1:size(positions, 1)
                @inbounds positions[coordinate, walker] = candidates[coordinate, walker]
            end
            @inbounds logdensities[walker] = candidate_logdensity
        end
    end
end

KA.@kernel function _count_accepted_kernel!(status, accepted)
    index = @index(Global, Linear)
    count = zero(eltype(status))
    for walker in eachindex(accepted)
        count += @inbounds accepted[walker]
    end
    @inbounds status[3] = count
end
