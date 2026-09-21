function _kernel_move(move::_AllocatedGaussianMove, initial)
    T = eltype(move.mean)
    d, n = size(initial)
    return _AllocatedGaussianMove(move.shrinkage, similar(initial, T, d),
        similar(initial, T, d, d), similar(initial, T, d, n), Matrix{T}(undef, d, n))
end

_kernel_move_supported(::_AllocatedGaussianMove) = true

function _prepare_gaussian_group!(w::KernelWorkspace, move, state, indices)
    n = length(indices)
    for (j, i) in enumerate(indices)
        w.host_controls[1, j] = i
    end
    copyto!(w.controls, 1, w.host_controls, 1, 4n)
    backend = KA.get_backend(w.positions)
    _gather_gaussian_complement_kernel!(backend, (32, 4))(
        move.scratch, w.positions, w.controls; ndrange=(size(w.positions, 1), n))
    scale, valid = _fit_gaussian!(move.mean, move.factor, move.scratch, n, move.shrinkage)
    return _FittedGaussianMove(move, scale, valid)
end

function _evaluate_group!(::KernelExecutor, state, fitted::_FittedGaussianMove,
    part, move_index, proposal_index, group, complement, acceptance_part)
    w = state.batch_workspace
    move = fitted.move
    n = length(group)
    for (j, i) in enumerate(group)
        w.host_controls[1, j] = i
    end
    copyto!(w.controls, 1, w.host_controls, 1, 4n)
    backend = KA.get_backend(w.positions)
    if !fitted.valid
        w.host_status[1] = 0
        _reject_gaussian_group_kernel!(backend, 1)(
            w.candidates, w.positions, state.candidate_logdensities, state.logdensities,
            state.accepted, state.acceptance_probabilities, w.controls, w.valid, w.status, n; ndrange=1)
        return nothing
    end

    normal_part = _walker_rngpart(part, _SCALE_PURPOSE, proposal_index)
    for (j, i) in enumerate(group)
        rng, id = state.walker_rngs[i], state.walker_ids[i]
        set_rng!(rng, normal_part, id)
        randn!(rng, view(move.host_scratch, :, j))
        set_rng!(rng, acceptance_part, id)
        w.host_factors[2, j] = rand(rng, eltype(w.host_factors))
    end
    copyto!(move.scratch, 1, move.host_scratch, 1, size(w.positions, 1) * n)
    copyto!(w.factors, 1, w.host_factors, 1, 2n)
    _propose_gaussian_kernel!(backend, 64)(w.candidates, w.positions, w.controls,
        w.logh, w.valid, state.accepted, state.candidate_logdensities,
        state.logdensities, state.acceptance_probabilities, move.mean, move.factor, move.scratch,
        fitted.scale; ndrange=n)
    _compact_kernel!(backend, 1)(w.indices, w.status, w.valid, n; ndrange=1)
    copyto!(w.host_status, 1, w.status, 1, 1)
    KA.synchronize(backend)
    return nothing
end

KA.@kernel function _gather_gaussian_complement_kernel!(scratch, positions, controls)
    coordinate, j = @index(Global, NTuple)
    walker = @inbounds controls[1, j]
    @inbounds scratch[coordinate, j] = positions[coordinate, walker]
end

KA.@kernel function _reject_gaussian_group_kernel!(
    candidates, positions, candidate_logdensities, logdensities,
    accepted, acceptance_probabilities, controls, valid, status, n,
)
    index = @index(Global, Linear)
    for j in 1:n
        walker = @inbounds controls[1, j]
        @inbounds accepted[walker] = false
        @inbounds valid[j] = false
        @inbounds candidate_logdensities[walker] = logdensities[walker]
        @inbounds acceptance_probabilities[walker] = zero(eltype(acceptance_probabilities))
        for coordinate in axes(positions, 1)
            @inbounds candidates[coordinate, walker] = positions[coordinate, walker]
        end
    end
    @inbounds status[1] = zero(eltype(status))
    @inbounds status[2] = zero(eltype(status))
end

KA.@kernel function _propose_gaussian_kernel!(
    candidates,
    positions,
    controls,
    logh,
    valid,
    accepted,
    candidate_logdensities,
    logdensities,
    acceptance_probabilities,
    mean,
    factor,
    scratch,
    scale,
)
    j = @index(Global, Linear)
    walker = @inbounds controls[1, j]
    dimension = size(positions, 1)
    proposal_valid = true
    for row in 1:dimension
        offset = zero(eltype(positions))
        for column in 1:row
            offset += @inbounds factor[row, column] * scratch[column, j]
        end
        candidate = scale * (@inbounds(mean[row]) + offset)
        @inbounds candidates[row, walker] = candidate
        proposal_valid &= isfinite(candidate)
    end

    current_distance = zero(eltype(positions))
    candidate_distance = zero(eltype(positions))
    if proposal_valid
        for row in 1:dimension
            residual = @inbounds positions[row, walker] / scale - mean[row]
            for column in 1:(row - 1)
                residual -= @inbounds factor[row, column] * scratch[column, j]
            end
            residual /= @inbounds factor[row, row]
            @inbounds scratch[row, j] = residual
            current_distance += abs2(residual)
        end
        for row in 1:dimension
            residual = @inbounds candidates[row, walker] / scale - mean[row]
            for column in 1:(row - 1)
                residual -= @inbounds factor[row, column] * scratch[column, j]
            end
            residual /= @inbounds factor[row, row]
            @inbounds scratch[row, j] = residual
            candidate_distance += abs2(residual)
        end
        proposal_valid = isfinite(current_distance) && isfinite(candidate_distance)
    end
    @inbounds accepted[walker] = false
    @inbounds valid[j] = proposal_valid
    if proposal_valid
        @inbounds logh[j] = (candidate_distance - current_distance) / 2
    else
        @inbounds candidate_logdensities[walker] = logdensities[walker]
        @inbounds acceptance_probabilities[walker] = zero(eltype(acceptance_probabilities))
        for coordinate in 1:dimension
            @inbounds candidates[coordinate, walker] = positions[coordinate, walker]
        end
    end
end
