@testset "Gaussian replacement move" begin
    @testset "Gamma stationarity" begin
        target(x) = x[1] > 0 ? 2log(x[1]) - x[1] : -Inf
        initial = reshape(collect(range(0.5, 6; length=32)), 1, :)
        state = initialize(test_rng(), target, initial; move=GaussianReplacementMove())
        step!(state, 300)
        draws = vec(sample!(state, 1_500).positions)
        @test mean(draws) ≈ 3 atol=0.12 rtol=0
        @test var(draws) ≈ 3 atol=0.25 rtol=0
    end

    @testset "Thread, walker ID, and resumed mixture replay" begin
        initial = initial_walkers()
        permutation = reverse(eachindex(initial))
        move = MoveMixture((GaussianReplacementMove(), DEMove()), [2, 1]; schedule=:cycle)
        reference = initialize(test_rng(), gaussian_logdensity, initial; move)
        threaded = initialize(test_rng(), gaussian_logdensity, initial;
            move, executor=ThreadedExecutor())
        reordered = initialize(test_rng(), gaussian_logdensity, initial[permutation];
            move, walker_ids=collect(permutation))
        expected = sample!(reference, 18)
        @test sample!(threaded, 18) == expected
        first_part = sample!(reordered, 7)
        second_part = sample!(reordered, 11)
        @test cat(first_part.positions, second_part.positions; dims=3)[:, permutation, :] ==
            expected.positions
        @test hcat(first_part.logdensities, second_part.logdensities)[permutation, :] ==
            expected.logdensities
        @test hcat(first_part.accepted, second_part.accepted)[permutation, :] == expected.accepted
        @test vcat(first_part.move_indices, second_part.move_indices) == expected.move_indices
    end

    @testset "Batched target" begin
        initial = initial_walkers()
        scalar_calls = Ref(0)
        widths = Int[]
        scalar(x) = (scalar_calls[] += 1; gaussian_logdensity(x))
        batch!(values, positions) = begin
            push!(widths, size(positions, 2))
            values .= gaussian_logdensity.(eachcol(positions))
        end
        move = GaussianReplacementMove(shrinkage=0)
        reference = initialize(test_rng(), gaussian_logdensity, initial; move)
        batched = initialize(test_rng(), BatchedLogDensity(scalar, batch!), initial; move)
        @test sample!(batched, 5) == sample!(reference, 5)
        @test scalar_calls[] == length(initial)
        @test widths == fill(length(initial) ÷ 2, 10)
    end

    @testset "Failed complement fits omit callbacks" begin
        # Full affine rank globally, but every three-point complement is singular.
        initial = [-1.0 0 0 1 0 0; 0 -1 0 0 1 0; 0 0 -1 0 0 1]
        calls = Ref(0)
        target(x) = (calls[] += 1; gaussian_logdensity(x))
        # Positive shrinkage makes the minimum walker count sufficient for this fit.
        state = initialize(test_rng(), target, initial; move=GaussianReplacementMove())
        step!(state)
        @test calls[] == 2size(initial, 2)

        # Every partition has at least one singular complement. Reject proposals
        # so the first group cannot change the second group's fitting geometry.
        initial = [0.0 0 0 0 0 0 1 0; 0 0 0 0 0 0 0 1]
        calls[] = 0
        reject_proposals(x) = (calls[] += 1; calls[] <= size(initial, 2) ? 0.0 : -Inf)
        state = initialize(Philox4x((1, 19)), reject_proposals, initial;
            move=GaussianReplacementMove(shrinkage=0))
        draws = sample!(state, 1)
        @test calls[] in (size(initial, 2), 3size(initial, 2) ÷ 2)
        @test !any(draws.accepted)
        @test draws.positions[:, :, 1] == initial

        # Translating constant points to zero must not change fit validity.
        # Seeded partitions can differ between Julia versions.
        initial = [1.0 1 1 0 0 2; 0.1 0.1 0.1 0 1 0]
        calls[] = 0
        state = initialize(Philox4x((3, 19)), reject_proposals, initial;
            move=GaussianReplacementMove())
        draws = sample!(state, 1)
        original_calls = calls[]
        @test !any(draws.accepted)
        @test draws.positions[:, :, 1] == initial
        initial = initial .- initial[:, 1]
        calls[] = 0
        step!(initialize(Philox4x((3, 19)), reject_proposals, initial;
            move=GaussianReplacementMove()))
        @test calls[] == original_calls
    end

    @testset "Extreme coordinate rescaling" begin
        initial = Float32[-1 0 1 0 -1 -1 1 1; 0 -1 0 1 -1 1 -1 1]
        reference = sample!(initialize(test_rng(), _ -> 0f0, initial;
            move=GaussianReplacementMove()), 4)
        for scale in (1f-25, 1f25)
            draws = sample!(initialize(test_rng(), _ -> 0f0, initial .* scale;
                move=GaussianReplacementMove()), 4)
            @test draws.accepted == reference.accepted
            @test draws.positions ./ scale ≈ reference.positions rtol=100eps(Float32)
        end
    end
end
