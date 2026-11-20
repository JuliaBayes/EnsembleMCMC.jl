using SparseArrays: sparse

function test_narrow_mixture_weights(device, executor)
    @testset "Small random mixture weights" begin
        target = BatchedLogDensity(x -> 0.0, (v, x) -> fill!(v, 0.0))
        for (weights, seed) in ((Float16[1, 2^-12], 2179), (Float16[2^-12, 1], 3056),
                (Float16[2^-12, 1], 1252), (Float32[1, 2^-25], 42_759_292))
            key = Philox4x64(seed)
            u = rand_at(subrng(key, 0, 0, 1, 1), Float64, 1)
            expected = u < first(weights) / sum(Float64, weights) ? 1 : 2
            move = MoveMixture((StretchMove(), DEMove()), weights)
            state = initialize(key, target, device([-1.0 0.0 1.0 2.0]); move, executor)
            @test current_state(step!(state)).move_index == expected
        end
    end
end

include("kernel_mixture.jl")

function test_kernel_executor(device=copy)
    @testset "Kernel executor" begin
        test_proposal_overflow(device; executor=KernelExecutor())
        test_keyed_kernel_mixture(device)
        test_narrow_mixture_weights(device, KernelExecutor())
        scalar(x::Vector) = -sum(abs2, x) / 2
        batch!(values, positions) = (values .= vec(-sum(abs2, positions; dims=1) / 2))
        target = BatchedLogDensity(scalar, batch!)
        @testset "Disabled mixture component" begin
            points = device([-1.0 0 1 0; 0 -1 0 1])
            move = MoveMixture((StretchMove(), GaussianReplacementMove(shrinkage=0)), (1, 0))
            state = initialize(test_rng(), target, points; move, executor=KernelExecutor())
            reference = initialize(test_rng(), target, points; executor=KernelExecutor())
            @test Array(sample!(state, 5).positions) == Array(sample!(reference, 5).positions)
            @test current_state(state).attempts == [20, 0]
        end
        @testset "Log-density offsets preserve acceptance" begin
            points = Float32[-1 0 1 0 -1 -1 1 1; 0 -1 0 1 -1 1 -1 1]
            states = map((0f0, -1f10)) do offset
                offset_density(x) = maximum(abs, x) < 100 ? offset : -Inf32
                offset_batch!(v, x) = (v .= ifelse.(vec(maximum(abs, x; dims=1)) .< 100, offset, -Inf32))
                initialize(test_rng(), BatchedLogDensity(offset_density, offset_batch!), device(points);
                    executor=KernelExecutor())
            end
            draws = map(state -> sample!(state, 10), states)
            @test Array(first(draws).positions) == Array(last(draws).positions)
            @test Array(first(draws).accepted) == Array(last(draws).accepted)
            @test Array(current_state(first(states)).acceptance_probabilities) ==
                Array(current_state(last(states)).acceptance_probabilities)
        end
        @testset "Snooker geometry far from the origin" begin
            points = 1.4e308 .+ 1e294 .* [-1.0 0 1 0 -1 -1 1 1; 0 -1 0 1 -1 1 -1 1]
            calls = Ref(0)
            density(x) = (calls[] += 1; -sum(abs2, (x .- 1.4e308) ./ 1e294) / 2)
            function density!(v, x)
                calls[] += length(v)
                v .= vec(-sum(abs2, (x .- 1.4e308) ./ 1e294; dims=1) ./ 2)
            end
            state = initialize(test_rng(), BatchedLogDensity(density, density!), device(points);
                move=DESnookerMove(), executor=KernelExecutor())
            step!(state)
            @test calls[] == 2size(points, 2)
        end
        initial = first(randn_next(Philox4x64(18), 3, 12))
        coordinates = Float32.(initial)
        values = Float64.(scalar.(Vector.(eachcol(coordinates))))
        cached = device(values)
        calls = Ref(0)
        counted(x) = (calls[] += 1; scalar(x))
        state = initialize(test_rng(), BatchedLogDensity(counted, batch!), device(coordinates);
            logdensities=cached, executor=KernelExecutor())
        @test calls[] == 0
        @test eltype(current_state(state).logdensities) === Float64
        fill!(cached, 99)
        @test Array(current_state(state).logdensities) == values
        reference = initialize(test_rng(), scalar, coordinates; logdensities=values)
        @test Array(sample!(state, 2).positions) ≈ sample!(reference, 2).positions

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
        ids = 17 .* reverse(axes(initial, 2))
        reference = initialize(test_rng(), scalar, initial; move=DEMove(), walker_ids=ids)
        state = initialize(test_rng(), target, device(initial);
            move=DEMove(), walker_ids=ids, executor=KernelExecutor())
        previous_logdensities = copy(current_state(reference).logdensities)
        # A sweep key can use a different engine from the run key.
        address = Threefry4x64(573)
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
        degenerate = initialize(test_rng(), flat, device(duplicates);
            move=DESnookerMove(), executor=KernelExecutor())
        draws = sample!(degenerate, 1)
        transition = snapshot(degenerate)
        rejected = .!Array(transition.accepted)
        @test all(iszero, Array(transition.acceptance_probabilities)[rejected])
        @test Array.(transition.candidates[rejected]) == Array.(transition.positions[rejected])
        @test widths == expected_widths
        @test Array(draws.positions) ≈ expected.positions

        widths = Int[]
        initial = [1.0 1 1 0 0 2; 0.1 0.1 0.1 0 1 0]
        flat = BatchedLogDensity(_ -> 0.0,
            (values, positions) -> (push!(widths, size(positions, 2)); fill!(values, -Inf)))
        draws = sample!(initialize(Philox4x64(13), flat, device(initial);
            move=GaussianReplacementMove(), executor=KernelExecutor()), 1)
        @test widths == [3]
        @test !any(Array(draws.accepted))
        @test Array(draws.positions[:, :, 1]) == initial
        empty!(widths)
        step!(initialize(Philox4x64(13), flat, device(initial .- initial[:, 1]);
            move=GaussianReplacementMove(), executor=KernelExecutor()))
        @test widths == [3]

        mixture = MoveMixture((DEMove(), GaussianReplacementMove()), [1, 1]; schedule=:cycle)
        duplicates = reshape([0.0, 0, 0, 0, 0, 0, 0, 1], 1, :)
        expected = initialize(test_rng(), flat, duplicates; move=mixture)
        actual = initialize(test_rng(), flat, device(duplicates);
            move=mixture, executor=KernelExecutor())
        step!(expected, 2)
        step!(actual, 2)
        @test reduce(hcat, Array.(current_state(actual).candidates)) ≈
            reduce(hcat, current_state(expected).candidates)
        @test Array(current_state(actual).candidate_logdensities) ==
            current_state(expected).candidate_logdensities
        @test Array(current_state(actual).acceptance_probabilities) ==
            current_state(expected).acceptance_probabilities

        points = duplicates .+ 1
        point_target = BatchedLogDensity(x -> x[1] in (1, 2) ? 0.0 : -Inf,
            (v, x) -> (v .= ifelse.((vec(x) .== 1) .| (vec(x) .== 2), 0.0, -Inf)))
        mixture = MoveMixture((DEMove(gamma0=eps()/4, sigma=0), GaussianReplacementMove()),
            [1, 1]; schedule=:cycle)
        state = initialize(test_rng(), point_target, device(points); move=mixture, executor=KernelExecutor())
        step!(state, 2)
        @test all(iszero, Array(current_state(state).acceptance_probabilities))

        extremes = reshape(Float32[1.2f38, 1.2f38, 1.2f38, 3f38, 3f38, 3f38], 1, :)
        state = initialize(test_rng(), flat, device(extremes);
            move=GaussianReplacementMove(), executor=KernelExecutor())
        @test all(1:8) do _
            step!(state)
            all(x -> all(isfinite, Array(x)), current_state(state).candidates)
        end

        calls = Ref(0)
        incomplete!(values, positions) = (calls[] += 1; nothing)
        failed = initialize(test_rng(), BatchedLogDensity(scalar, incomplete!),
            device(initial); executor=KernelExecutor())
        @test_throws DomainError step!(failed)
        @test calls[] == 2 # The kernel path checks values once per sweep.
        @test_throws ArgumentError current_state(failed)
        throws!(values, positions) = (fill!(values, 0); error("callback failed"))
        failed = initialize(test_rng(), BatchedLogDensity(scalar, throws!),
            device(initial); executor=KernelExecutor())
        @test_throws ErrorException step!(failed)
        @test_throws ArgumentError snapshot(failed)
        invalid = BatchedLogDensity(scalar, (v, x) -> fill!(v, Inf))
        failed = initialize(test_rng(), invalid, device(initial); executor=KernelExecutor())
        err = try step!(failed); nothing catch e; e end
        @test err isa DomainError && occursin("walker 1 (ID 1) in sweep 1", err.msg)
        outside = BatchedLogDensity(scalar, (v, x) -> fill!(v, -Inf))
        state = initialize(test_rng(), outside, device(initial); executor=KernelExecutor())
        @test !any(Array(sample!(state, 1).accepted))

        # NaN candidates are rejections, so they must match a -Inf reference exactly.
        region!(v, x) = (v .= ifelse.(view(x, 1, :) .> 1, NaN, vec(-sum(abs2, x; dims=1)) ./ 2))
        nan_target = BatchedLogDensity(scalar, region!)
        reference_target(x) = x[1] > 1 ? -Inf : scalar(x)
        inside = clamp.(initial, -1, 0.9)
        for move in (StretchMove(), DEMove())
            expected = sample!(initialize(test_rng(), reference_target, inside; move), 50)
            draws = sample!(initialize(test_rng(), nan_target, device(inside);
                move, executor=KernelExecutor()), 50)
            @test Array(draws.accepted) == expected.accepted
            @test Array(draws.positions) ≈ expected.positions
            @test all(<=(1), Array(draws.positions)[1, :, :])
        end

        # Compare wide linear proposals and both shuffle sizes with the serial sampler.
        linear = MoveMixture((StretchMove(), DEMove()), [1, 1]; schedule=:cycle)
        for (move, d, n) in ((linear, 65, 601), (DEMove(), 2, 4097))
            wide = first(randn_next(Philox4x64(5), d, n))
            reference = initialize(test_rng(), scalar, wide; move)
            state = initialize(test_rng(), target, device(wide); move, executor=KernelExecutor())
            expected, draws = sample!(reference, 5), sample!(state, 5)
            @test Array(draws.accepted) == expected.accepted
            @test Array(draws.positions) ≈ expected.positions
            @test current_state(state).acceptances == current_state(reference).acceptances
        end
        backend = EnsembleMCMC.KA.get_backend(device(initial))
        @test all(Iterators.product((1, 7, 256, 257, 1000, 4099), (1, 64, 256))) do (n, width)
            slots = 4:(n + 3)
            valid = first(rand_next(Philox4x64(n), Bool, last(slots)))
            indices, status = device(zeros(Int, n)), device(zeros(Int, 3))
            EnsembleMCMC._compact_kernel!(backend, width)(indices, status, device(valid), slots;
                ndrange=width)
            count = Array(status)[1]
            Array(indices)[1:count] == filter(i -> valid[i], slots)
        end
    end
end

test_narrow_mixture_weights(copy, SerialExecutor())
test_kernel_executor()

@testset "Sparse CPU kernel history" begin
    initial = [1.0 -1.0 -1.0 1.0 0.0 0.0; 0.0 0.0 1.0 -1.0 2.0 -2.0]
    density(x) = -sum(abs2, x) / 2
    batch!(values, positions) = map!(density, values, eachcol(positions))
    target = BatchedLogDensity(density, batch!)
    key = Philox4x64(731)
    dense = initialize(key, target, initial; executor=KernelExecutor())
    compressed = initialize(key, target, sparse(initial); executor=KernelExecutor())
    @test sample!(compressed, 5) == sample!(dense, 5)

    address = Threefry4x64(92)
    step!(compressed, address)
    step!(dense, address)
    @test snapshot(compressed) == snapshot(dense)
end
