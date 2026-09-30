@testset "Batched log density" begin
    initial = Float32.(reduce(hcat, initial_walkers()))
    ids = collect(3:2:(2size(initial, 2) + 1))
    move = MoveMixture((StretchMove(), DEMove(), DESnookerMove()), [1, 1, 1]; schedule=:cycle)

    for executor in (SerialExecutor(), ThreadedExecutor())
        widths = Int[]
        batch!(values, positions) = begin
            push!(widths, size(positions, 2))
            values .= gaussian_logdensity.(eachcol(positions))
        end
        reference = initialize(test_rng(), gaussian_logdensity, initial;
            move, executor, walker_ids=ids)
        batched = initialize(test_rng(), BatchedLogDensity(gaussian_logdensity, batch!), initial;
            move, executor, walker_ids=ids)
        @test sample!(batched, 9) == sample!(reference, 9)
        @test length(widths) == 24
        @test sum(widths) == 9size(initial, 2)
    end

    @testset "Compacts degenerate proposals" begin
        widths = Int[]
        scalar_calls = Ref(0)
        scalar(x) = (scalar_calls[] += 1; gaussian_logdensity(x))
        batch!(values, positions) = begin
            push!(widths, size(positions, 2))
            values .= gaussian_logdensity.(eachcol(positions))
        end
        initial = Float32[0 0 0 1]
        state = initialize(test_rng(), BatchedLogDensity(scalar, batch!), initial;
            move=DESnookerMove())
        @test scalar_calls[] == size(initial, 2)
        step!(state)
        @test scalar_calls[] == size(initial, 2)
        @test !isempty(widths)
        @test sum(widths) < size(initial, 2)
        @test all(>(0), widths)
    end

    @testset "Callback failure invalidates the state" begin
        initial = reduce(hcat, initial_walkers())
        failing_batch!(values, positions) = error("batch failed")
        state = initialize(test_rng(),
            BatchedLogDensity(gaussian_logdensity, failing_batch!), initial)
        @test_throws ErrorException step!(state)
        @test_throws ArgumentError current_state(state)

        nan_batch!(values, positions) = fill!(values, NaN)
        rejected = initialize(test_rng(),
            BatchedLogDensity(gaussian_logdensity, nan_batch!), initial)
        before = snapshot(rejected)
        step!(rejected)
        @test !any(current_state(rejected).accepted)
        @test all(==(-Inf), current_state(rejected).candidate_logdensities)
        @test current_state(rejected).positions == before.positions

        for invalid_batch! in ((values, positions) -> fill!(values, Inf), (values, positions) -> nothing)
            invalid = initialize(test_rng(),
                BatchedLogDensity(gaussian_logdensity, invalid_batch!), initial)
            err = try step!(invalid); nothing; catch e; e; end
            @test err isa DomainError && occursin("sweep 1", err.msg) && occursin("walker", err.msg)
            @test_throws ArgumentError current_state(invalid)
        end
    end
end
