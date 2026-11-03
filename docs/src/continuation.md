# Continuing a run

Use `initialize`, `step!`, and `sample!` when you need to continue an ensemble,
report progress, or store output in blocks.

## Keep the sampler state

The lower-level interface uses a Random123 `Philox4x` or `Threefry4x` RNG with
`UInt64` counters. Add Random123 before using this interface:

```julia
using Pkg
Pkg.add("Random123")
```

```jldoctest continuation
using EnsembleMCMC, Random, Random123
using EnsembleMCMC: initialize, step!, sample!, current_state, snapshot

rng = Philox4x((42, 1));
initial = randn(rng, 2, 24);
state = initialize(rng, x -> -sum(abs2, x) / 2, initial);

step!(state, 100);              # discard warmup
first_block = sample!(state, 200);
next_block = sample!(state, 50);

(size(first_block.positions), size(next_block.positions), current_state(state).sweep_count)

# output

((2, 24, 200), (2, 24, 50), 350)
```

`step!` advances the state without storing a history. `sample!` advances the same
state and returns a new history. Neither restarts the ensemble.

`initialize` copies the starting coordinates and RNG. Unlike `sample`, it does
not advance the supplied RNG. Two calls with the same RNG and starting points
therefore make identical runs. Use distinct keys for separate ensembles, such
as `Philox4x((42, ensemble_id))`.

## Store samples and report progress

Collect blocks to limit memory use and keep completed output if a later block
fails:

```julia
for block in 1:10
    draws = sample!(state, 100)
    # Write draws to your chosen storage here.
    println("Completed ", current_state(state).sweep_count, " sweeps")
end
```

To retain a single sweep, use `snapshot(state)`. It copies the current sample
data and counters. `current_state(state)` gives a cheaper view into the sampler,
whose arrays must be read before the next change to the state. Do not edit them
or inspect a state while another task steps it.

A snapshot is saved sample data, not a restart checkpoint. Keep the sampler
state alive to continue a run. Restarting a sampler from a file is not supported.

## Measure acceptance after warmup

`acceptance_rate(state)` reports cumulative rates per move, including warmup.
Take differences of the counters to measure a later part of the run:

```julia
before = current_state(state)
attempts = copy(before.attempts)
acceptances = copy(before.acceptances)
draws = sample!(state, 1_000)
after = current_state(state)
production_acceptance = (after.acceptances .- acceptances) ./ (after.attempts .- attempts)
```

A move with no attempts in that interval has rate `NaN`. The `accepted` field in
a thinned history describes only the retained sweeps, so its average can differ
from the all-sweep rate.

## Handle target errors

A target exception during a sweep invalidates the sampler state. Fix the target
and initialize a new state. A failed state cannot continue or be inspected with
`current_state` or `snapshot`.

`sample!` returns its history only on success. If it fails, samples from that call
are lost. Histories returned by earlier calls remain valid.
