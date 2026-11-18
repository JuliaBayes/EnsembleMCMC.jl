# SPDX-License-Identifier: Apache-2.0

const _MOVE_PURPOSE = UInt64(1)
const _INNER_MOVE_PURPOSE = UInt64(2)
const _ACCEPTANCE_PURPOSE = UInt64(3)
const _GROUP_PURPOSE = UInt64(4)
const _COMPANION_PURPOSE = UInt64(5)
const _SCALE_PURPOSE = UInt64(6)

# StatefulRNG exposes the same key on the CPU, where all proposal controls are drawn.
_rng_root(rng::AbstractPureRNG, max_depth) = parent(PureRNGs.StatefulRNG(rng))
_rng_root(rng, max_depth) = throw(ArgumentError(
    "Unsupported RNG $(typeof(rng)); use a PureRNGs key such as Philox4x64(42)"))

_sample_root(rng::AbstractPureRNG) = rng
_sample_root(rng::AbstractRNG) = PureRNGs.Philox4x64((rand(rng, UInt64), rand(rng, UInt64)))

const _HardwareFloat = Union{Float16,Float32,Float64}
_uniform(rng, ::Type{T}) where {T<:_HardwareFloat} = PureRNGs.rand_at(rng, T, 1)
_uniform(rng, ::Type{T}) where {T} = rand(PureRNGs.StatefulRNG(rng), T)

# Dense IDs read a bulk fill. Sparse IDs address the same sequence directly.
_uniform_at(values::AbstractVector, T, id) = values[id]
_uniform_at(rng::AbstractPureRNG, ::Type{T}, id) where {T<:_HardwareFloat} = PureRNGs.rand_at(rng, T, id)
_uniform_at(rng::AbstractPureRNG, ::Type{T}, id) where {T} = _uniform(subrng(rng, id), T)
_normal_at(values::AbstractVector, ::Type{T}, id) where {T<:_HardwareFloat} = values[id]
_normal_at(rng::AbstractPureRNG, ::Type{T}, id) where {T<:_HardwareFloat} = PureRNGs.randn_at(rng, T, id)
_normal_at(rng::AbstractPureRNG, ::Type{T}, id) where {T} = randn(PureRNGs.StatefulRNG(subrng(rng, id)), T)

function _normal_column!(values::AbstractVector, id, destination)
    return copyto!(destination, 1, values, (id - 1) * length(destination) + 1, length(destination))
end
function _normal_column!(rng::AbstractPureRNG, id, destination::AbstractVector{T}) where {T<:_HardwareFloat}
    d = UInt64(length(destination))
    last = UInt64(id) * d
    # Every supported engine can address the first 2^56 64-bit draws.
    if last > UInt64(1) << 56
        return first(PureRNGs.randn_next!(subrng(rng, id), destination))
    end
    for j in eachindex(destination)
        destination[j] = PureRNGs.randn_at(rng, T, last - d + j)
    end
    return destination
end
_normal_column!(rng::AbstractPureRNG, id, destination) =
    randn!(PureRNGs.StatefulRNG(subrng(rng, id)), destination)

# Uniform on 1:n, including spans that do not divide a power of two.
# DE needs an ordered pair to have the same probability as its reverse.
@inline function _index_next(rng, n::Int)
    mask = typemax(UInt64) >> leading_zeros(UInt64(n - 1))
    while true
        word, rng = PureRNGs.rand_next(rng, UInt64)
        index = word & mask
        index < n && return Int(index) + 1, rng
    end
end

function _shuffle!(rng, items)
    for i in length(items):-1:2
        j, rng = _index_next(rng, i)
        items[i], items[j] = items[j], items[i]
    end
    return rng
end

# Multiply-high maps equally many words to each index after discarding the
# short residual interval. A rejection retries on its own address's child.
@inline function _bounded_index(word::UInt64, key, address, n::Int)
    span = UInt64(n)
    product = widemul(word, span)
    low = product % UInt64
    if low < span && low < rem(-span, span)
        return first(_index_next(subrng(key, address), n))
    end
    return Int(product >> 64) + 1
end

_companion_word(::Nothing, key, address) = PureRNGs.rand_at(key, UInt64, address)
_companion_word(words::AbstractVector, key, address) = words[address]

@inline function _companion_index(streams, id, lanes, slot, n)
    address = (UInt64(id) - UInt64(1)) * UInt64(lanes) + UInt64(slot)
    word = _companion_word(streams.words, streams.companion, address)
    return _bounded_index(word, streams.companion, address, n)
end
