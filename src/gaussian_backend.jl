function _kernel_move(move::GaussianReplacementMove, initial, L)
    T = _gaussian_work_type(eltype(initial), L)
    d, n = size(initial)
    return _AllocatedGaussianMove(move.shrinkage, similar(initial, T, d), similar(initial, T, d),
        similar(initial, T, d, d), similar(initial, T, d, n), nothing)
end

_kernel_move_supported(::_AllocatedGaussianMove) = true

_synchronize_move(move::_AllocatedGaussianMove) =
    _kernel_synchronize(move.anchor, move.mean, move.factor, move.scratch)

function _prepare_gaussian_group!(w::KernelWorkspace, move, state, group, indices, streams)
    n = length(indices)
    _launch_kernel(w, _gather_gaussian_complement_kernel!, (32, 4),
        (size(w.positions, 1), n), move.scratch, w.positions, w.permutation, indices)
    scale, valid = _fit_gaussian!(move.anchor, move.mean, move.factor, move.scratch, n,
        move.shrinkage, w.execution)
    return _FittedGaussianMove(move, scale, valid)
end

function _evaluate_group!(::KernelExecutor, state, fitted::_FittedGaussianMove,
    streams, group, complement)
    w = state.batch_workspace
    move = fitted.move
    n = length(group)
    if !fitted.valid
        w.host_status[1] = 0
        _launch_kernel(w, _reject_gaussian_group_kernel!, 64, n,
            w.candidates, w.positions, state.candidate_logdensities, state.logdensities,
            state.accepted, state.acceptance_probabilities, w.permutation, group, w.valid)
        return nothing
    end

    _launch_kernel(w, _draw_gaussian_kernel!, 64, n, move.scratch, w.controls,
        w.uniforms, w.permutation, w.walker_ids, group, streams, Val(eltype(w.positions)))
    _launch_kernel(w, _propose_gaussian_kernel!, 64, n, w.candidates, w.positions, w.controls,
        w.logh, w.valid, state.accepted, state.candidate_logdensities,
        state.logdensities, state.acceptance_probabilities, move.anchor, move.mean, move.factor,
        move.scratch, fitted.scale)
    _compact!(w, 1:n)
    return nothing
end

KA.@kernel function _gather_gaussian_complement_kernel!(scratch, positions, permutation, group)
    coordinate, j = @index(Global, NTuple)
    walker = permutation[group[j]]
    @inbounds scratch[coordinate, j] = positions[coordinate, walker]
end

# PureRNGs bulk and static-vector normal draws follow scalar stream order.
# Very distant columns use the same child stream as the scalar executor.
function _device_normal_column!(scratch, j, key, id, ::Type{T}) where {T<:_HardwareFloat}
    d = UInt64(size(scratch, 1))
    if UInt64(id) > div(UInt64(1) << 56, d)
        key = subrng(key, id)
        id = 1
    end
    offset = (UInt64(id) - 1) * d
    for coordinate in axes(scratch, 1)
        scratch[coordinate, j] = PureRNGs.randn_at(key, T, offset + coordinate)
    end
    return nothing
end
_device_normal_column!(scratch, j, key, id, T) = _normal_column!(key, id, view(scratch, :, j))

KA.@kernel function _draw_gaussian_kernel!(scratch, controls, uniforms, permutation, walker_ids,
    group, keys, ::Val{T}) where {T}
    j = @index(Global, Linear)
    walker = permutation[group[j]]
    id = walker_ids[walker]
    streams = _device_streams(keys)
    controls[1, j] = walker
    uniforms[j] = _uniform_at(streams.acceptance, eltype(uniforms), id)
    _device_normal_column!(scratch, j, streams.scale, id, T)
end

KA.@kernel function _reject_gaussian_group_kernel!(
    candidates, positions, candidate_logdensities, logdensities,
    accepted, acceptance_probabilities, permutation, group, valid,
)
    j = @index(Global, Linear)
    walker = permutation[group[j]]
    @inbounds accepted[walker] = false
    @inbounds valid[j] = false
    @inbounds candidate_logdensities[walker] = logdensities[walker]
    @inbounds acceptance_probabilities[walker] = zero(eltype(acceptance_probabilities))
    for coordinate in axes(positions, 1)
        @inbounds candidates[coordinate, walker] = positions[coordinate, walker]
    end
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
    anchor,
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
        offset = zero(eltype(factor))
        for column in 1:row
            offset += @inbounds factor[row, column] * scratch[column, j]
        end
        proposal = @inbounds(anchor[row]) + scale * (@inbounds(mean[row]) + offset)
        candidate = convert(eltype(positions), proposal)
        @inbounds candidates[row, walker] = candidate
        proposal_valid &= isfinite(candidate)
    end

    current_distance = zero(eltype(logh))
    candidate_distance = zero(eltype(logh))
    if proposal_valid
        for row in 1:dimension
            residual = @inbounds (positions[row, walker] - anchor[row]) / scale - mean[row]
            for column in 1:(row - 1)
                residual -= @inbounds factor[row, column] * scratch[column, j]
            end
            residual /= @inbounds factor[row, row]
            @inbounds scratch[row, j] = residual
            wide = convert(eltype(logh), residual)
            current_distance += (wide / 2) * wide
        end
        for row in 1:dimension
            residual = @inbounds (candidates[row, walker] - anchor[row]) / scale - mean[row]
            for column in 1:(row - 1)
                residual -= @inbounds factor[row, column] * scratch[column, j]
            end
            residual /= @inbounds factor[row, row]
            @inbounds scratch[row, j] = residual
            wide = convert(eltype(logh), residual)
            candidate_distance += (wide / 2) * wide
        end
    end
    @inbounds accepted[walker] = false
    @inbounds valid[j] = proposal_valid
    if proposal_valid
        @inbounds logh[j] = candidate_distance - current_distance
    else
        @inbounds candidate_logdensities[walker] = logdensities[walker]
        @inbounds acceptance_probabilities[walker] = zero(eltype(acceptance_probabilities))
        for coordinate in 1:dimension
            @inbounds candidates[coordinate, walker] = positions[coordinate, walker]
        end
    end
end
