function test_kernel_executor(device=copy)
    @testset "Kernel executor" begin
        scalar(x::Vector) = -sum(abs2, x) / 2
        batch!(values, positions) = (values .= vec(-sum(abs2, positions; dims=1) / 2))
        target = BatchedLogDensity(scalar, batch!)
        initial = randn(MersenneTwister(18), 3, 12)
        for move in (StretchMove(), DEMove(), DESnookerMove(), GaussianReplacementMove())
            reference = initialize(test_rng(), scalar, initial; move)
            state = initialize(test_rng(), target, device(initial); move, executor=KernelExecutor())
            borrowed = current_state(state)
            @test borrowed.positions === current_state(state).positions
            saved = snapshot(state)
            expected = sample!(reference, 3)
            draws = sample!(state, 3)
            @test Array(draws.positions) ≈ expected.positions
            @test Array(draws.logdensities) ≈ expected.logdensities
            @test Array(draws.accepted) == expected.accepted
            @test reduce(hcat, Array.(saved.positions)) == initial
            fill!(saved.positions[1], NaN)
            step!(state)
            @test all(isfinite, reduce(hcat, Array.(current_state(state).positions)))
            @test Array(draws.positions) ≈ expected.positions
        end

        integer_initial = [-1 0 0 1 0 0; 0 -1 0 0 1 0; 0 0 -1 0 0 1]
        expected = sample!(initialize(test_rng(), scalar, integer_initial;
            move=GaussianReplacementMove()), 1)
        draws = sample!(initialize(test_rng(), target, device(integer_initial);
            move=GaussianReplacementMove(), executor=KernelExecutor()), 1)
        @test Array(draws.positions) ≈ expected.positions
        @test Array(draws.accepted) == expected.accepted
        move = MoveMixture((StretchMove(), DEMove(), DESnookerMove(), GaussianReplacementMove()),
            [1, 1, 1, 1]; schedule=:cycle)
        permutation = reverse(axes(initial, 2))
        reference = initialize(test_rng(), target, device(initial); move, executor=KernelExecutor())
        reordered = initialize(test_rng(), target, device(initial[:, permutation]);
            move, walker_ids=collect(permutation), executor=KernelExecutor())
        expected = sample!(reference, 8)
        first_part = sample!(reordered, 2)
        second_part = sample!(reordered, 6)
        @test cat(Array(first_part.positions), Array(second_part.positions); dims=3)[:, permutation, :] ==
            Array(expected.positions)
        @test vcat(first_part.move_indices, second_part.move_indices) == expected.move_indices
        @test sum(current_state(reference).acceptances) == count(Array(expected.accepted))

        for scale in (1f-25, 1f25)
            coordinates = Float32[-1 0 1 0 -1 -1 1 1; 0 -1 0 1 -1 1 -1 1] .* scale
            flat = BatchedLogDensity(_ -> 0.0, (v, x) -> fill!(v, 0))
            for move in (DESnookerMove(), GaussianReplacementMove())
                expected = sample!(initialize(test_rng(), flat, coordinates; move), 1)
                draws = sample!(initialize(test_rng(), flat, device(coordinates);
                    move, executor=KernelExecutor()), 1)
                @test Array(draws.accepted) == expected.accepted
                @test Array(draws.positions) ./ scale ≈
                    expected.positions ./ scale rtol=100eps(Float32)
            end
        end
        widths = Int[]
        flat = BatchedLogDensity(_ -> 0.0, (v, x) -> (push!(widths, size(x, 2)); fill!(v, 0)))
        duplicates = reshape([0.0, 0, 0, 0, 0, 0, 0, 1], 1, :)
        expected = sample!(initialize(test_rng(), flat, duplicates; move=DESnookerMove()), 1)
        expected_widths = copy(widths)
        empty!(widths)
        draws = sample!(initialize(test_rng(), flat, device(duplicates);
            move=DESnookerMove(), executor=KernelExecutor()), 1)
        @test widths == expected_widths
        @test Array(draws.positions) ≈ expected.positions

        widths = Int[]
        initial = [1.0 1 1 0 0 2; 0.1 0.1 0.1 0 1 0]
        flat = BatchedLogDensity(_ -> 0.0,
            (values, positions) -> (push!(widths, size(positions, 2)); fill!(values, -Inf)))
        draws = sample!(initialize(Philox4x((3, 19)), flat, device(initial);
            move=GaussianReplacementMove(), executor=KernelExecutor()), 1)
        original_widths = copy(widths)
        @test !any(Array(draws.accepted))
        @test Array(draws.positions[:, :, 1]) == initial
        empty!(widths)
        step!(initialize(Philox4x((3, 19)), flat, device(initial .- initial[:, 1]);
            move=GaussianReplacementMove(), executor=KernelExecutor()))
        @test widths == original_widths

        calls = Ref(0)
        incomplete!(values, positions) = (calls[] += 1; nothing)
        failed = initialize(test_rng(), BatchedLogDensity(scalar, incomplete!),
            device(initial); executor=KernelExecutor())
        @test_throws DomainError step!(failed)
        @test calls[] == 1
        @test_throws ArgumentError current_state(failed)
        throws!(values, positions) = (fill!(values, 0); error("callback failed"))
        failed = initialize(test_rng(), BatchedLogDensity(scalar, throws!),
            device(initial); executor=KernelExecutor())
        @test_throws ErrorException step!(failed)
        @test_throws ArgumentError snapshot(failed)
        invalid = BatchedLogDensity(scalar, (v, x) -> fill!(v, Inf))
        failed = initialize(test_rng(), invalid, device(initial); executor=KernelExecutor())
        @test_throws DomainError step!(failed)
        outside = BatchedLogDensity(scalar, (v, x) -> fill!(v, -Inf))
        state = initialize(test_rng(), outside, device(initial); executor=KernelExecutor())
        @test !any(Array(sample!(state, 1).accepted))
    end
end

test_kernel_executor()
