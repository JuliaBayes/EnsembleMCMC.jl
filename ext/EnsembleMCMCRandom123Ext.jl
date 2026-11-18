# SPDX-License-Identifier: Apache-2.0

module EnsembleMCMCRandom123Ext

import EnsembleMCMC
using PureRNGs: Philox4x64, Threefry4x64, subrng
using Random123: Philox4x, Threefry4x

_legacy_key(rng::Philox4x{UInt64,R}) where {R} =
    subrng(Philox4x64((rng.key1, rng.key2)), R)

_legacy_key(rng::Threefry4x{UInt64,R}) where {R} =
    subrng(Threefry4x64((rng.key1, rng.key2, rng.key3, rng.key4)), R)

function EnsembleMCMC._rng_root(rng::Union{Philox4x{UInt64},Threefry4x{UInt64}}, max_depth::Int)
    # BAT packs six tagged 31-bit partition counters into these three words.
    ctr4, ctr3, ctr2 = rng.ctr4, rng.ctr3, rng.ctr2
    slots = (ctr4 >> 32, ctr4 & 0xffffffff, ctr3 >> 32, ctr3 & 0xffffffff,
        ctr2 >> 32, ctr2 & 0xffffffff)
    depth = 1
    for (i, slot) in pairs(slots)
        if (slot & 0x80000000) != 0
            depth = i
        elseif slot != 0
            throw(ArgumentError("Inconsistent partition counter information"))
        end
    end
    depth <= max_depth || throw(ArgumentError(
        "RNG partition depth $depth is too deep; expected depth $max_depth or less"))

    # Legacy initialization reserved cycle streams and addressed stepping reserved purpose streams.
    reservation = max_depth == 2 ? typemax(Int16) : 1 + 6 * EnsembleMCMC._PROPOSALS_PER_PURPOSE
    (slots[depth] & 0x7fffffff) <= typemax(Int32) - reservation ||
        throw(ArgumentError("Partition counter(s) out of allowed range"))

    # Draws within an address only change ctr1 and the buffer lane.
    return subrng(_legacy_key(rng), ctr4, ctr3, ctr2)
end

end
