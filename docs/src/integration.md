# Writing an adapter

```@meta
CurrentModule = EnsembleMCMC
```

An adapter supplies model-specific densities and starting points, then converts
the sampler's output into its host package's result type. EnsembleMCMC owns the
moves, ensemble state, RNG addressing, and transition records.

Use [`sample`](@ref) for a complete run. Use [`initialize`](@ref), [`step!`](@ref),
and [`sample!`](@ref) when the host controls warmup, storage, or stopping. Inspect
the state through the public functions rather than its concrete fields.

## Reuse evaluated starting points

Pass `initialize(...; logdensities=cached_values)` when the host has already
evaluated the target. Initialization copies those values without calling the
target again. Their precision is independent of the coordinate precision.

The values must match the target and walker order. NaN and `+Inf` are invalid.
The adapter remains responsible for transforms and any Jacobian terms in its
target.

## Synchronize external updates

[`synchronize!`](@ref) copies replacement positions and matching log densities
into a valid state. It preserves walker IDs, RNG state, mixture phase, sweep
count, and acceptance counts. It clears the previous transition record.

The caller must preserve density consistency and affine rank. Call
[`validate_positions`](@ref) after retry initialization when that needs checking.
Ordinary synchronization does not repeat the rank check.

Arrays borrowed from this state may be supplied in their original walker order.
Other overlap, including reordered borrowed walkers, is unsupported.

## Read samples and transitions

| History field | Axes or meaning |
| --- | --- |
| `positions` | parameter × walker × retained sweep |
| `logdensities` | walker × retained sweep |
| `accepted` | walker × retained sweep |
| `move_indices` | selected move for each retained sweep |
| `walker_ids` | logical IDs in storage order |

Histories own their arrays. Rejected proposals repeat the current point. With
thinning, transition fields describe the retained sweep only.

[`current_state`](@ref) also exposes candidates, candidate log densities, actual
acceptance probabilities, and cumulative counts. Its arrays are borrowed and
must be treated as read-only. Read them before the next state mutation. Scalar
metadata is captured when queried. Use [`snapshot`](@ref) to keep an independent
copy. Do not inspect a state concurrently with stepping it.

For a kernel state, `current_state(state).positions` is a host vector of device
column views. Walker IDs, move indices, and counters are on the host.

## Supply an external RNG address

`step!(state, rng; proposal_index=1)` uses an external Random123 address without
changing the supplied RNG. Its type must match the initialization RNG. Reserve
two remaining partition levels for purposes and walkers.

Inner mixture selection uses purpose 2, leaving purpose 1 for an outer mixture.
This address replaces the normal step address for that sweep. The sweep count
and mixture phase still advance, so a later `step!(state)` uses the next normal
address. See [`step!`](@ref) for the supported address range.

For a standalone state, `initialize` needs four free partition levels. The
one-call `sample` interface accepts any `AbstractRNG` and draws two `UInt64`
values to seed its own Philox RNG.

## Interface guarantees

The public interface includes documented constructors, sampling methods, keywords,
defaults, and result fields, including qualified functions that are not exported.
Concrete state constructors and fields, scratch storage, RNG partition
helpers, and undocumented extension hooks are internal.

Seeded replay applies within a fixed software and hardware environment.
Numerical corrections can change trajectories.
