# SPDX-License-Identifier: Apache-2.0

using Random123: random123_r

struct RNGPartition{R<:AbstractRNG,S,C,I}
    seed::S
    partctrsbase::C
    depth::Int
    idxs::I
end

function RNGPartition(parent_rng::AbstractRNG, partidxs::AbstractUnitRange{<:Integer})
    seed = rngpart_getseed(parent_rng)
    partcounters, depth = rngpart_getpartctrs(parent_rng)

    child_rngpart_partctrsbase = _rngpart_inc_partctrs(partcounters, depth, 1)

    parent_mod_partcounters = _rngpart_inc_partctrs(child_rngpart_partctrsbase, depth, length(partidxs))
    rngpart_setpartctrs!(parent_rng, parent_mod_partcounters, depth)

    R = typeof(parent_rng)
    S = typeof(seed)
    C = typeof(child_rngpart_partctrsbase)
    I = typeof(partidxs)
    RNGPartition{R,S,C,I}(seed, child_rngpart_partctrsbase, depth, partidxs)
end

Base.eachindex(rngpart::RNGPartition) = rngpart.idxs
Base.length(rngpart::RNGPartition) = length(eachindex(rngpart))
Base.size(rngpart::RNGPartition) = (length(rngpart),)

function set_rng!(rng::R, rngpart::RNGPartition{R}, i::Integer) where R <: AbstractRNG
    idxs = eachindex(rngpart)
    Base.checkindex(Bool, idxs, i) || throw(ArgumentError("Index $i not in partition indices $idxs of $rngpart"))

    j = i - minimum(idxs)
    mod_partcounters = _rngpart_inc_partctrs(rngpart.partctrsbase, rngpart.depth, j)
    mod_depth = rngpart.depth + 1

    Random.seed!(rng, rngpart.seed)
    rngpart_setpartctrs!(rng, mod_partcounters, mod_depth)

    rng
end

Random.AbstractRNG(rngpart::RNGPartition{R}, i::Integer) where R =
    set_rng!(rngpart_createrng(R), rngpart, i)

rngpart_createrng(::Type{T}) where {T <: Philox4x} = T(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

rngpart_getseed(rng::Philox4x) = (rng.key1, rng.key2)

rngpart_createrng(::Type{T}) where {T <: Threefry4x} = T(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

rngpart_getseed(rng::Threefry4x) = (rng.key1, rng.key2, rng.key3, rng.key4)

function rngpart_getpartctrs(rng::Union{Philox4x,Threefry4x})
    partctrinfo = _rngpart_split_uints(rng.ctr4, rng.ctr3, rng.ctr2)
    depth = _rngpart_getdepth(partctrinfo)
    partctrs = map(_rngpart_getpartctr, partctrinfo)
    (partctrs = partctrs, depth = depth)
end

function rngpart_setpartctrs!(rng::Union{Philox4x,Threefry4x}, partctrs::NTuple{6,UInt32}, depth::Integer)
    tagged_partctrs = _rngpart_settopbit(partctrs, depth)
    merged_partctrs = _rngpart_merge_uints(tagged_partctrs...)::NTuple{3,UInt64}

    rng.ctr4 = merged_partctrs[1]
    rng.ctr3 = merged_partctrs[2]
    rng.ctr2 = merged_partctrs[3]
    rng.ctr1 = 0

    random123_r(rng)

    rng
end

@inline _rngpart_split_uints() = ()
@inline _rngpart_split_uints(x::UInt64, xs::UInt64...) =
    (UInt32(x >> 32), UInt32(x << 32 >> 32), _rngpart_split_uints(xs...)...)

@inline _rngpart_merge_uints() = ()
@inline _rngpart_merge_uints(x1::UInt32, x2::UInt32, xs::UInt32...) =
    (UInt64(x1) << 32 | UInt64(x2), _rngpart_merge_uints(xs...)...)

_rngpart_topbit_mask(::Type{T}) where {T <: Unsigned} = ((typemax(T) >> 1) + one(T))

_rngpart_lowbits_mask(::Type{T}) where {T <: Unsigned} = typemax(T) >> 1

_rngpart_haspartctrtag(x::T) where {T<:Unsigned} = (x & _rngpart_topbit_mask(T)) > 0

_rngpart_getpartctr(x::T) where {T<:Unsigned} = (x & _rngpart_lowbits_mask(T))

function _rngpart_getdepth(partctrinfo::Tuple{T,Vararg{T}}) where {T<:Unsigned}
    cycle::Int = 1
    for i in eachindex(partctrinfo)
        x = partctrinfo[i]
        if _rngpart_haspartctrtag(x)
            cycle = i
        else
            x == 0 || throw(ArgumentError("Inconsistent partition counter information"))
        end
    end
    return cycle
end

function _rngpart_inc_partctrs(partctrs::Tuple{T,Vararg{T}}, depth::Integer, x::Integer) where {T<:Unsigned}
    1 <= depth <= length(partctrs) || throw(ArgumentError("Partition depth out of allowed range"))
    m = ntuple(i -> i == depth ? T(x) : zero(T), Val(length(partctrs)))
    partctrs .+ m
end

function _rngpart_settopbit(partctrs::Tuple{T,Vararg{T}}, depth::Integer) where {T<:Unsigned}
    1 <= depth <= length(partctrs) || throw(ArgumentError("Partition depth out of allowed range"))
    any(_rngpart_haspartctrtag, partctrs) && throw(ArgumentError("Partition counter(s) out of allowed range"))
    m = ntuple(i -> i <= depth ? _rngpart_topbit_mask(T) : zero(T), Val(length(partctrs)))
    partctrs .| m
end
