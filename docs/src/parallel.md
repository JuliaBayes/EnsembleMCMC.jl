# Threads and GPUs

```@meta
CurrentModule = EnsembleMCMC
```

## Julia threads

Start Julia with several threads, for example `julia --threads=4`, and pass
`executor=ThreadedExecutor()`:

```jldoctest threaded
using EnsembleMCMC, Random

rng = Xoshiro(42);
initial = randn(rng, 2, 24);
draws = EnsembleMCMC.sample(rng, x -> -sum(abs2, x) / 2, initial, 10;
    executor=ThreadedExecutor());
size(draws.positions)

# output

(2, 24, 10)
```

The target must allow concurrent calls and must not change its input. The sampler
updates groups in order. Walkers in one group use fixed positions from the other
groups while making their proposals.

Threads help when target evaluations outweigh task overhead. Cheap targets often
run faster with the default `SerialExecutor()`. Increase `min_chunk` to give each
task more walkers and reduce the number of tasks. Measure the full sampling run
for your target. Thread scheduling and chunk size do not change walker RNG streams.

## Evaluate a batch of points

Some targets can evaluate several points more efficiently together. Wrap the
scalar function and an in-place batch function in `BatchedLogDensity`:

```jldoctest batched
using EnsembleMCMC, Random

scalar(x) = -sum(abs2, x) / 2;
batch!(values, positions) = (values .= scalar.(eachcol(positions)));
target = BatchedLogDensity(scalar, batch!);
initial = [-1.0 0 1 0; 0 -1 0 1];
draws = EnsembleMCMC.sample(Xoshiro(42), target, initial, 10);
size(draws.positions)

# output

(2, 4, 10)
```

The broadcast shows the calling convention. Replace it with a faster batch
computation when the target supports one.

Each column of `positions` is one candidate. Fill every entry of `values` with
the same log density that `scalar` would return. Leave `positions` unchanged and
do not retain either array after the call. The sampler can pass views rather
than plain arrays. A batch contains only valid proposals, so its column count
can vary.

For a callback compiled for one shape, set `BatchedLogDensity(scalar, batch!;
batch_size=width)`. Choose a width large enough for every active proposal group.
The sampler passes full buffers of that width and pads unused columns with the
first valid proposal. Only real proposals affect acceptance. Empty groups skip
the callback.

Initialization calls `scalar` unless cached densities are supplied. Sampling
calls `batch!` once per nonempty proposal group. With `ThreadedExecutor`, threads
generate proposals, while the callback controls evaluation parallelism. Storage
stays on the CPU with either `SerialExecutor` or `ThreadedExecutor`.

## CUDA

Add CUDA to the environment. Pass a device matrix, a device batch function, and
`KernelExecutor()` to keep proposals and samples on the GPU:

```julia
using CUDA, EnsembleMCMC, Random

CUDA.allowscalar(false)
rng = Xoshiro(42)
initial = CuArray(randn(rng, Float32, 8, 32))
scalar(x) = -sum(abs2, x) / 2
batch!(values, positions) = (values .= vec(-sum(abs2, positions; dims=1) / 2))
target = BatchedLogDensity(scalar, batch!)

draws = EnsembleMCMC.sample(rng, target, initial, 200;
    warmup=100, move=DEMove(), executor=KernelExecutor())
host_positions = Array(draws.positions)
```

Initialization checks coordinates on the CPU and calls `scalar` with CPU
vectors. Keep host copies of any data that function needs. The batch function
must use device data during sampling.

Proposals, log densities, acceptance flags, snapshots, and history stay on the
input device. `Array(draws.positions)` explicitly copies the result to the CPU.
The RNG generates proposal controls on the CPU and transfers them to the device.
Small status records and counts return to the CPU during sampling. Successful
sweeps do not transfer coordinates or log densities back to the CPU.

The batch function must use the current task's CUDA stream or finish its work
before returning. Sampler calls complete on the input device before returning.
`KernelExecutor` supports every built-in move and mixture. It also accepts CPU
matrices for testing.

GPU sampling suits expensive parallel batch targets. Kernel launches and group
synchronization can make small targets slower. CPU and GPU arithmetic can also
round differently, so their trajectories need not match bit for bit.

Invalid batch outputs, such as `+Inf` or unwritten entries, fail at the end of the
sweep. Later group callbacks may run before that error is reported. A thrown
callback stops the sweep immediately. Either failure invalidates the state.

## Reactant

Load both Reactant and CUDA to use a single-device PJRT array on a CUDA GPU with
`KernelExecutor`. Compile the density once for a fixed batch width:

```julia
using Reactant, CUDA, EnsembleMCMC, Random

Reactant.set_default_backend("gpu")
rng = Xoshiro(42)
initial = Reactant.to_rarray(randn(rng, 2, 32))
scalar(x) = -sum(abs2, x) / 2
function batch!(values, positions)
    values .= vec(-sum(abs2, positions; dims=1) ./ 2)
    return nothing
end

# StretchMove splits the 32 walkers into two groups of 16.
values = similar(initial, 16)
positions = similar(initial, 2, 16)
compiled = Reactant.@compile batch!(values, positions)
target = BatchedLogDensity(scalar, compiled; batch_size=16)
draws = EnsembleMCMC.sample(rng, target, initial, 200; executor=KernelExecutor())
host_positions = Array(draws.positions)
```

The sampler also reuses compiled proposal, acceptance and history operations.
Setup includes compilation, so measure repeated sweeps separately when timing
this backend. Device transfers follow the CUDA path described above.
