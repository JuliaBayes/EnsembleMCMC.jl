function test_proposal_overflow(device=identity; executor=SerialExecutor())
    @testset "Gaussian half-quadratic overflow" begin
        initial = Float32[-1 1 -1 3.3f19]
        for width in (2f0, 4f0)
            scalar(x) = -abs2(x[1] / width)
            batch!(v, x) = (v .= -abs2.(vec(x) ./ width))
            target = executor isa KernelExecutor ? BatchedLogDensity(scalar, batch!) : scalar
            state = initialize(Philox4x((2, 1)), target, device(initial);
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
        for (initial, move, seed) in (
            ([-1.0 1.0], StretchMove(), 2),
            ([0.1 -1.0], StretchMove(), 8),
            ([-1.0 1.0 0.0 0.0], DEMove(gamma0=0.5, sigma=0), 2),
            ([1.0 -1.0 0.0 -1.0], DEMove(gamma0=1, sigma=0), 2),
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
            # Compare the first affected walker before later complements can diverge.
            i = move isa DEMove ? 4 : 1
            @test only(Array(actual.candidates[i])) / 1e308 ≈ only(Array(expected.candidates[i]))
            @test Array(actual.acceptance_probabilities)[i] ≈ Array(expected.acceptance_probabilities)[i]
            @test Array(actual.accepted)[i] == Array(expected.accepted)[i]
        end
    end

    @testset "Zero and unrepresentable linear displacements" begin
        scalar(x) = -abs2(x[1] / 1e308) / 2
        batch!(v, x) = (v .= -abs2.(vec(x) ./ 1e308) ./ 2)
        target = executor isa KernelExecutor ? BatchedLogDensity(scalar, batch!) : scalar
        tiny = nextfloat(0.0)
        state = initialize(Philox4x((4, 1)), target, device([-1e308 1e308 tiny tiny]);
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

        state = initialize(Philox4x((2, 1)), target, device([-1.0 1.0 0.0 0.0]);
            move=DEMove(gamma0=floatmax(Float64), sigma=10), executor)
        step!(state)
        @test !any(Array(snapshot(state).accepted))
    end
end
