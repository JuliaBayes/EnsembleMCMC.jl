function test_proposal_overflow(device=identity; executor=SerialExecutor())
    @testset "Gaussian half-quadratic overflow" begin
        initial = Float32[-1 1 -1 3.3f19]
        # The full quadratic overflows although half of it fits.
        @test isinf(abs2(initial[4] / sqrt(3f0)))
        for width in (2f0, 4f0)
            scalar(x) = -abs2(x[1] / width)
            batch!(v, x) = (v .= -abs2.(vec(x) ./ width))
            target = executor isa KernelExecutor ? BatchedLogDensity(scalar, batch!) : scalar
            state = initialize(Philox4x((8, 1)), target, device(initial);
                move=GaussianReplacementMove(), executor)
            step!(state)
            saved = snapshot(state)
            x, y = Float64(initial[4]), Float64(only(Array(saved.candidates[4])))
            # The frozen complement [-1, 1] has inflated proposal variance 3.
            logratio = Float64(scalar([Float32(y)])) - Float64(scalar(initial[:, 4])) + (y^2 - x^2) / 6
            @test Array(saved.acceptance_probabilities)[4] == min(1, exp(logratio))
            @test Array(saved.accepted)[4] == (width == 2)
        end
    end

    @testset "Linear proposals across the coordinate range" begin
        # These seeds put walker i in the first group with difference a - b.
        for (initial, move, seed, i, a, b) in (
            ([-1.0 1.0], StretchMove(), 5, 1, 1, 2),
            ([0.1 -1.0], StretchMove(), 24, 1, 1, 2),
            ([-1.0 1.0 0.0 0.0], DEMove(gamma0=0.5, sigma=0), 18, 4, 2, 1),
            ([1.0 -1.0 0.0 -1.0], DEMove(gamma0=1, sigma=0), 8, 4, 1, 2),
        )
            function run(scale)
                scalar(x) = -abs2(x[1] / scale) / 2
                batch!(v, x) = (v .= -abs2.(vec(x) ./ scale) ./ 2)
                target = executor isa KernelExecutor ? BatchedLogDensity(scalar, batch!) : scalar
                state = initialize(Philox4x((seed, 1)), target, device(initial .* scale); move, executor)
                step!(state)
                snapshot(state)
            end
            expected, actual = run(1.0), run(1e308)
            candidate = only(Array(expected.candidates[i]))
            origin = move isa StretchMove ? initial[b] : initial[i]
            factor = move isa StretchMove ? (candidate - origin) / (initial[a] - initial[b]) : move.gamma0
            difference = 1e308 * initial[a] - 1e308 * initial[b]
            @test isinf(factor * difference)
            @test abs(candidate) < floatmax(Float64) / 1e308
            @test only(Array(actual.candidates[i])) / 1e308 ≈ candidate
            @test Array(actual.acceptance_probabilities)[i] ≈ Array(expected.acceptance_probabilities)[i]
            @test Array(actual.accepted)[i] == Array(expected.accepted)[i]
        end
    end

    @testset "Snooker projections beyond the coordinate range" begin
        points = [0 .1 .6 -.6; 0 .6 0 0]
        scale = floatmax(Float64)
        initial = scale .* points
        scalar(x) = -sum(abs2, x ./ scale) / 2
        batch!(v, x) = (v .= vec(-sum(abs2, x ./ scale; dims=1) / 2))
        target = executor isa KernelExecutor ? BatchedLogDensity(scalar, batch!) : scalar
        move = DESnookerMove()
        # Walker 2 moves first, with walker 1 as reference and companions 4 and 3.
        state = initialize(Philox4x((57, 19)), target, device(initial); move, executor)
        step!(state)
        saved = snapshot(state)
        x = points[:, 2]
        direction = x / hypot(x...)
        displacement = move.scale * sum(direction .* (points[:, 4] - points[:, 3]))
        candidate = x + direction * displacement
        logratio = (sum(abs2, x) - sum(abs2, candidate)) / 2 + log(hypot(candidate...) / hypot(x...))
        @test isinf(initial[1, 4] - initial[1, 3])
        @test Array(saved.candidates[2]) ./ scale ≈ candidate
        @test Array(saved.acceptance_probabilities)[2] ≈ min(1, exp(logratio))
    end

    @testset "Zero and unrepresentable linear displacements" begin
        scalar(x) = -abs2(x[1] / 1e308) / 2
        batch!(v, x) = (v .= -abs2.(vec(x) ./ 1e308) ./ 2)
        target = executor isa KernelExecutor ? BatchedLogDensity(scalar, batch!) : scalar
        tiny = nextfloat(0.0)
        # Walker 4 picks the opposite-sign companions and its jitter rounds the factor to zero.
        state = initialize(Philox4x((18, 1)), target, device([-1e308 1e308 tiny tiny]);
            move=DEMove(gamma0=tiny, sigma=1), executor)
        step!(state)
        saved = snapshot(state)
        @test only(Array(saved.candidates[4])) == tiny
        @test Array(saved.accepted)[4]

        for initial in ([-1e308 1e308 0.0 0.0], [1e308 0.0 0.0 0.0])
            state = initialize(Philox4x((2, 1)), target, device(initial);
                move=DEMove(gamma0=2, sigma=0), executor)
            step!(state)
            saved = snapshot(state)
            unrepresentable = .!isfinite.(map(x -> only(Array(x)), saved.candidates))
            @test any(unrepresentable)
            @test all(iszero, Array(saved.acceptance_probabilities)[unrepresentable])
        end

        # This seed overflows the jittered factor for all four walkers.
        state = initialize(Philox4x((1, 1)), target, device([-1.0 1.0 0.0 0.0]);
            move=DEMove(gamma0=floatmax(Float64), sigma=10), executor)
        step!(state)
        saved = snapshot(state)
        @test all(x -> isinf(only(Array(x))), saved.candidates)
        @test !any(Array(saved.accepted))
    end
end
