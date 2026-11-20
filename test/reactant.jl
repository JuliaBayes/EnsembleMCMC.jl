using EnsembleMCMC, Reactant, CUDA, Test
using PureRNGs: Philox4x64, randn_next
using EnsembleMCMC: initialize, step!, sample!, snapshot, synchronize!, current_state,
    validate_positions

Reactant.set_default_backend("gpu")
CUDA.allowscalar(false)

function batch_normal!(values, positions)
    values .= vec(-sum(abs2, positions; dims = 1) ./ 2)
    return nothing
end

@testset "Reactant shuffle launch boundaries" begin
    scalar(x) = -abs2(only(x)) / 2
    key = Philox4x64(23)
    for n in (256, 257, 4096, 4097)
        initial = first(randn_next(Philox4x64(42), 1, n))
        points = Reactant.to_rarray(initial)
        width = cld(n, 2)
        compiled = Reactant.Compiler.compile(
            batch_normal!, (similar(points, width), similar(points, 1, width)),
        )
        target = BatchedLogDensity(scalar, compiled; batch_size = width)
        device = initialize(key, target, points; move = DEMove(), executor = KernelExecutor())
        serial = initialize(key, scalar, initial; move = DEMove())
        actual, expected = sample!(device, 1), sample!(serial, 1)
        @test Array(actual.accepted) == expected.accepted
        @test Array(actual.positions) ≈ expected.positions
        @test Array(actual.logdensities) ≈ expected.logdensities
    end
end

@testset "Reactant transitions and retained history" begin
    scalar(x) = -sum(abs2, x) / 2
    initial = first(randn_next(Philox4x64(42), 2, 13))
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
            Philox4x64(UInt64.((12, 1))),
            target,
            Reactant.to_rarray(start);
            move,
            executor = KernelExecutor(),
        )
        host = initialize(
            Philox4x64(UInt64.((12, 1))),
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
            Philox4x64(UInt64.((17, 1))),
            target,
            device_initial;
            executor = KernelExecutor(),
        )
        host = initialize(
            Philox4x64(UInt64.((17, 1))),
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

@testset "Reactant preserves the sweep failure during cleanup" begin
    initial = Reactant.to_rarray([-1.0 0 1 0; 0 -1 0 1])
    scalar(x) = -sum(abs2, x) / 2
    failure = ErrorException("density callback failed")
    function batch!(values, positions)
        # A failed PJRT call can leave its input marked as donated.
        Reactant.mark_donated!(values)
        throw(failure)
    end
    target = BatchedLogDensity(scalar, batch!; batch_size = 2)
    state = initialize(Philox4x64(42), target, initial; executor = KernelExecutor())
    @test_throws failure step!(state)
    @test_throws ArgumentError current_state(state)
end

@testset "Reactant position validation" begin
    coordinates = [0.0 1 2 0 0; 0.0 1 2 1 NaN]
    points = Reactant.to_rarray(coordinates[:, 1:4])
    @test validate_positions(points) === nothing
    @test_throws ArgumentError("Walkers must have full affine rank") validate_positions(
        Reactant.to_rarray(coordinates[:, 1:3]),
    )
    nonfinite = Reactant.to_rarray(coordinates)
    @test_throws ArgumentError("Coordinates must be finite") validate_positions(nonfinite)
    @test validate_positions(view(nonfinite, :, [4, 1, 2])) === nothing

    scalar(x) = -sum(abs2, x) / 2
    target = BatchedLogDensity(scalar, batch_normal!; batch_size = 2)
    state = initialize(Philox4x64(42), target, points; executor = KernelExecutor())
    borrowed = current_state(state).positions
    @test validate_positions(borrowed) === nothing
    @test validate_positions(borrowed[[4, 1, 2]]) === nothing
    columns = collect(eachcol(nonfinite))
    @test validate_positions(columns[[4, 1, 2]]) === nothing
    @test_throws ArgumentError("Walkers must have full affine rank") validate_positions(
        columns[[3, 1, 2]],
    )
    @test_throws ArgumentError("Coordinates must be finite") validate_positions(columns[[5, 1, 4]])
end
