using EnsembleMCMC, Reactant, CUDA, Random, Random123, Test
using EnsembleMCMC: initialize, step!, sample!, snapshot, synchronize!, current_state

Reactant.set_default_backend("gpu")
CUDA.allowscalar(false)

function batch_normal!(values, positions)
    values .= vec(-sum(abs2, positions; dims = 1) ./ 2)
    return nothing
end

@testset "Reactant transitions and retained history" begin
    scalar(x) = -sum(abs2, x) / 2
    initial = randn(Xoshiro(42), 2, 13)
    device_initial = Reactant.to_rarray(initial)
    compiled = Reactant.Compiler.compile(
        batch_normal!,
        (similar(device_initial, 7), Reactant.to_rarray(initial[:, 1:7])),
    )
    batch!(values, positions) = compiled(values, positions)
    target = BatchedLogDensity(scalar, batch!; batch_size = 7)
    # Duplicate points exercise varying compacted counts and failed Gaussian fits.
    repeated = [0.0 0 1 1 0 0 1 1 0 0 1 1 0; 0.0 0 0 0 1 1 1 1 0 0 0 0 1]
    for move in (StretchMove(), DEMove(), DESnookerMove(), GaussianReplacementMove())
        start = move isa Union{DESnookerMove,GaussianReplacementMove} ? repeated : initial
        device = initialize(
            Philox4x((12, 1)),
            target,
            Reactant.to_rarray(start);
            move,
            executor = KernelExecutor(),
        )
        host = initialize(
            Philox4x((12, 1)),
            BatchedLogDensity(scalar, batch_normal!),
            start;
            move,
            executor = KernelExecutor(),
        )
        for retained in (3, 5)
            actual, expected = sample!(device, retained), sample!(host, retained)
            @test Array(actual.accepted) == expected.accepted
            @test Array(actual.positions) ≈ expected.positions
            @test Array(actual.logdensities) ≈ expected.logdensities
        end
    end

    Reactant.set_default_backend("cpu")
    try
        device = initialize(
            Philox4x((17, 1)),
            target,
            device_initial;
            executor = KernelExecutor(),
        )
        host = initialize(
            Philox4x((17, 1)),
            BatchedLogDensity(scalar, batch_normal!),
            initial;
            executor = KernelExecutor(),
        )
        values = scalar.(eachcol(initial))
        for state in (device, host)
            step!(state)
            synchronize!(state, initial, values)
        end
        borrowed = current_state(device)
        synchronize!(device, borrowed.positions, borrowed.logdensities)
        saved = snapshot(device)
        actual, expected = sample!(device, 3), sample!(host, 3)
        @test Array(actual.positions) ≈ expected.positions
        @test Array(saved.logdensities) == values
        @test Reactant.XLA.platform_name(Reactant.XLA.default_backend()) == "cpu"
    finally
        Reactant.set_default_backend("gpu")
    end
end
