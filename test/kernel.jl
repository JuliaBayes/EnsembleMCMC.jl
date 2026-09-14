function test_kernel_executor(device=copy)
    @testset "Kernel executor" begin
        scalar(x::Vector) = -sum(abs2, x) / 2
        batch!(values, positions) = (values .= vec(-sum(abs2, positions; dims=1) / 2))
        target = BatchedLogDensity(scalar, batch!)
        initial = randn(MersenneTwister(18), 3, 12)
        for move in (StretchMove(), DEMove(), DESnookerMove())
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
        reference = initialize(test_rng(), scalar, initial; move=DEMove())
        state = initialize(test_rng(), target, device(initial); move=DEMove(), executor=KernelExecutor())
        previous_logdensities = copy(current_state(reference).logdensities)
        address = Philox4x((271, 12))
        step!(reference, address; proposal_index=2)
        step!(state, address; proposal_index=2)
        transition = snapshot(state)
        @test Array(transition.candidate_logdensities) ≈ scalar.(Array.(transition.candidates))
        @test Array(transition.acceptance_probabilities) ≈
            min.(1, exp.(Array(transition.candidate_logdensities) .- previous_logdensities))
        @test Array(transition.acceptance_probabilities) ≈ current_state(reference).acceptance_probabilities
        replacement = initial .+ 0.25
        replacement_logdensities = scalar.(Vector.(eachcol(replacement)))
        synchronize!(state, device(replacement), device(replacement_logdensities))
        synchronize!(reference, replacement, replacement_logdensities)
        step!(state, address; proposal_index=3)
        step!(reference, address; proposal_index=3)
        @test reduce(hcat, Array.(current_state(state).positions)) ≈ reduce(hcat, current_state(reference).positions)

        move = MoveMixture((StretchMove(), DEMove(), DESnookerMove()), [1, 1, 1]; schedule=:cycle)
        permutation = reverse(axes(initial, 2))
        reference = initialize(test_rng(), target, device(initial); move, executor=KernelExecutor())
        reordered = initialize(test_rng(), target, device(initial[:, permutation]);
            move, walker_ids=collect(permutation), executor=KernelExecutor())
        expected = sample!(reference, 6)
        first_part = sample!(reordered, 2)
        second_part = sample!(reordered, 4)
        @test cat(Array(first_part.positions), Array(second_part.positions); dims=3)[:, permutation, :] ≈
            Array(expected.positions)
        @test vcat(first_part.move_indices, second_part.move_indices) == expected.move_indices
        @test sum(current_state(reference).acceptances) == count(Array(expected.accepted))

        for scale in (1f-25, 1f25)
            coordinates = Float32[-1 0 1 0 -1 -1 1 1; 0 -1 0 1 -1 1 -1 1] .* scale
            flat = BatchedLogDensity(_ -> 0.0, (v, x) -> fill!(v, 0))
            expected = sample!(initialize(test_rng(), flat, coordinates; move=DESnookerMove()), 1)
            draws = sample!(initialize(test_rng(), flat, device(coordinates);
                move=DESnookerMove(), executor=KernelExecutor()), 1)
            @test Array(draws.accepted) == expected.accepted
            @test Array(draws.positions) ./ scale ≈ expected.positions ./ scale rtol=100eps(Float32)
        end
        widths = Int[]
        flat = BatchedLogDensity(_ -> 0.0, (v, x) -> (push!(widths, size(x, 2)); fill!(v, 0)))
        duplicates = reshape([0.0, 0, 0, 0, 0, 0, 0, 1], 1, :)
        expected = sample!(initialize(test_rng(), flat, duplicates; move=DESnookerMove()), 1)
        expected_widths = copy(widths)
        empty!(widths)
        degenerate = initialize(test_rng(), flat, device(duplicates);
            move=DESnookerMove(), executor=KernelExecutor())
        draws = sample!(degenerate, 1)
        transition = snapshot(degenerate)
        rejected = .!Array(transition.accepted)
        @test all(iszero, Array(transition.acceptance_probabilities)[rejected])
        @test Array.(transition.candidates[rejected]) == Array.(transition.positions[rejected])
        @test widths == expected_widths
        @test Array(draws.positions) ≈ expected.positions

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
