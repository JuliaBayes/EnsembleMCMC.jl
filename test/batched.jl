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

    @testset "Fixed batches preserve compacted transitions" begin
        initial = [0.0 0 0 0 1 1 1 1]
        move = MoveMixture((DESnookerMove(), StretchMove()), [1, 0])
        expected = sample!(initialize(test_rng(), gaussian_logdensity, initial; move), 5)
        for executor in (SerialExecutor(), KernelExecutor())
            widths = Int[]
            batch!(values, positions) = begin
                push!(widths, size(positions, 2))
                values .= gaussian_logdensity.(eachcol(positions))
            end
            target = BatchedLogDensity(gaussian_logdensity, batch!; batch_size=2)
            actual = sample!(initialize(test_rng(), target, initial; move, executor), 5)
            @test actual.positions ≈ expected.positions
            @test actual.accepted == expected.accepted
            @test !isempty(widths) && all(==(2), widths)
        end
    end

    @testset "Batch storage fits odd proposal groups" begin
        initial = reduce(hcat, initial_walkers()[1:13])
        snooker = MoveMixture((StretchMove(), DESnookerMove()), (0, 1); schedule=:cycle)
        for executor in (SerialExecutor(), KernelExecutor()),
            (move, capacity) in ((StretchMove(), 7), (snooker, 4))
            widths = Tuple{Int,Int}[]
            function batch!(values, positions)
                push!(widths, (size(positions, 2), size(parent(positions), 2)))
                return map!(gaussian_logdensity, values, eachcol(positions))
            end
            variable = BatchedLogDensity(gaussian_logdensity, batch!)
            padded = BatchedLogDensity(gaussian_logdensity, batch!; batch_size=capacity + 1)
            actual = sample!(initialize(test_rng(), variable, initial; move, executor), 2)
            @test maximum(first, widths) == capacity
            @test all(w -> last(w) == capacity, widths)
            empty!(widths)
            expected = sample!(initialize(test_rng(), padded, initial; move, executor), 2)
            @test all(==((capacity + 1, capacity + 1)), widths)
            @test actual == expected
        end
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

@testset "Inactive Gaussian components use no workspace" begin
    initial = first(randn_next(Philox4x64(42), 16, 40))
    batch!(values, positions) = map!(gaussian_logdensity, values, eachcol(positions))
    target = BatchedLogDensity(gaussian_logdensity, batch!)
    for schedule in (:cycle, :random), executor in (SerialExecutor(), KernelExecutor())
        move = MoveMixture((GaussianReplacementMove(), StretchMove()), (0, 1); schedule)
        mixed = initialize(test_rng(), target, initial; move, executor)
        reference = initialize(test_rng(), target, initial; executor)
        @test Base.summarysize(mixed) - Base.summarysize(reference) < sizeof(initial) ÷ 2
        for key in (nothing, Threefry4x64(93))
            if isnothing(key)
                step!(mixed)
                # Address the forced Stretch with the original mixture component index.
                step!(reference, subrng(test_rng(), 0, 0); proposal_index=2)
            else
                step!(mixed, key; proposal_index=17)
                step!(reference, key; proposal_index=17)
            end
            actual, expected = current_state(mixed), current_state(reference)
            @test actual.positions == expected.positions
            @test actual.candidates == expected.candidates
            @test actual.accepted == expected.accepted
            @test actual.move_index == 2
        end
        @test current_state(mixed).attempts == [0, 2size(initial, 2)]
    end
end
