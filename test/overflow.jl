using PureRNGs: rand_at, randn_at


function test_proposal_overflow(device=identity; executor=SerialExecutor())
    @testset "Rare transitions with Float16 coordinates" begin
        key = Philox4x64(45)
        acceptance = subrng(key, 3, 1)
        # At the first address the Float16 uniform is zero. At the second the Float32
        # uniform is zero. In both cases the Float64 draw exceeds the shelf probability.
        for (ids, logp) in (([1, 2], -12.0), ([7267203, 2], -20.0))
            shelf(x) = abs(x) > 5 ? -Inf : abs(x) == 1 ? 0.0 : logp
            target = BatchedLogDensity(x -> shelf(only(x)), (v, x) -> (v .= shelf.(vec(x))))
            state = initialize(key, target, device(Float16[-1 1]); executor, walker_ids=ids)
            step!(state, key)
            saved = current_state(state)
            @test Array(saved.candidate_logdensities)[1] == logp
            @test Array(saved.acceptance_probabilities)[1] ≈ exp(logp) rtol=2eps(Float64)
            @test Array(saved.accepted)[1] == (rand_at(acceptance, Float64, ids[1]) < exp(logp))
        end
    end

    @testset "Density difference with a wider Hastings correction" begin
        x = 1e20
        shelf(z) = abs(z) > 2x ? -Inf32 : abs(z) > x/10 ? -floatmax(Float32) : floatmax(Float32)
        target = BatchedLogDensity(z -> shelf(only(z)), (v, z) -> (v .= shelf.(vec(z))))
        state = initialize(Philox4x64(3), target, device([-1.0 1.0 -1.0 x]);
            move=GaussianReplacementMove(), executor)
        step!(state)
        saved = current_state(state)
        y = only(Array(saved.candidates[4]))
        # Walker 4's complement has mean zero and inflated proposal variance 3.
        ratio = Float64(shelf(y)) - Float64(shelf(x)) + (y^2 - x^2)/6
        @test ratio < 0 && isinf(shelf(y) - shelf(x))
        @test Array(saved.acceptance_probabilities)[4] == exp(ratio)
        @test !Array(saved.accepted)[4]
    end

    @testset "Float16 DE jitter survives representable cancellation" begin
        key = Philox4x64(1)
        target = BatchedLogDensity(x -> 0.0, (v, x) -> fill!(v, 0.0))
        # Walker 1 starts at -1 and its first-group companion difference is 1.
        noise = randn_at(subrng(key, 0, 0, 6, 1), Float16, 1)
        for (unit, sigma) in ((1, 0.0), (1, 1e-5), (1024, 1e-9), (1024, 1e-15))
            initial = unit .* Float16[-1 0 1 2]
            state = initialize(key, target, device(initial); move=DEMove(; gamma0=1, sigma), executor)
            # The unperturbed terms cancel exactly; round just the remaining displacement.
            expected = Float16(unit * sigma * Float64(noise))
            @test only(Array(current_state(step!(state)).candidates[1])) == expected
            @test iszero(expected) == (sigma == 0 || sigma == 1e-15)
        end
        # Wide device columns share one scale across all coordinates.
        wide = Float16.(round.(first(randn_next(Philox4x64(83), Float32, 32, 64))))
        state = initialize(key, target, device(wide); move=DEMove(gamma0=1), executor)
        @test any(x -> any(!isinteger, Array(x)), current_state(step!(state)).candidates)
    end

    @testset "Gaussian half-quadratic overflow" begin
        initial = Float32[-1 1 -1 3.3f19]
        # The full quadratic overflows although half of it fits.
        @test isinf(abs2(initial[4] / sqrt(3f0)))
        for width in (2f0, 4f0)
            scalar(x) = -abs2(x[1] / width)
            batch!(v, x) = (v .= -abs2.(vec(x) ./ width))
            target = executor isa KernelExecutor ? BatchedLogDensity(scalar, batch!) : scalar
            state = initialize(Philox4x64(3), target, device(initial);
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

    @testset "Gaussian Hastings ratio with a wider target" begin
        for (T, far) in ((Float16, 1000), (Float32, 1e20)), width in (2, 4)
            initial = reshape(T[-1, 1, -1, far], 1, :)
            scalar(x) = -abs2(Float64(x[1]) / width)
            batch!(v, x) = (v .= -abs2.(Float64.(vec(x)) ./ width))
            target = BatchedLogDensity(scalar, batch!)
            state = initialize(Philox4x64(3), target, device(initial);
                move=GaussianReplacementMove(), executor)
            step!(state)
            saved = snapshot(state)
            x, y = Float64(initial[4]), Float64(only(Array(saved.candidates[4])))
            # Changing only the active walker preserves its addressed draw from q=N(0,3).
            control = copy(initial)
            control[4] = 4
            reference = initialize(Philox4x64(3), target, device(control);
                move=GaussianReplacementMove(), executor)
            step!(reference)
            @test y == only(Array(snapshot(reference).candidates[4]))
            @test Array(saved.candidate_logdensities)[4] == -abs2(y / width)
            logratio = (x^2 - y^2) * (inv(width^2) - 1/6)
            @test Array(saved.acceptance_probabilities)[4] == min(1, exp(logratio))
            @test Array(saved.accepted)[4] == (logratio > 0)
        end
    end

    @testset "Opposing infinite terms in the Gaussian log ratio" begin
        initial = [-1.0 1.0 -1.0 3.7e154]
        for height in (1e308, 1.6e308)
            shelf(x) = abs(x) < 10 ? height : -height
            target = BatchedLogDensity(x -> shelf(only(x)), (v, x) -> (v .= shelf.(vec(x))))
            state = initialize(Philox4x64(3), target, device(initial);
                move=GaussianReplacementMove(), executor)
            step!(state)
            saved = snapshot(state)
            x, y = BigFloat(initial[4]), BigFloat(only(Array(saved.candidates[4])))
            logratio = 2BigFloat(height) + (y^2 - x^2) / 6
            @test Array(saved.acceptance_probabilities)[4] == (logratio > 0)
            @test Array(saved.accepted)[4] == (logratio > 0)
        end
    end

    @testset "Gaussian updates with a large coordinate shift" begin
        a, b = 0.6, 0.9
        # The frozen pair gives variance 3(b-a)^2/4 in normalized coordinates.
        function run(scale)
            initial = [a b a -0.8] .* scale
            μ, variance = (a + b) / 2, 0.75 * (b - a)^2
            scalar(x) = -abs2(only(x) / scale - μ) / (2variance)
            batch!(v, x) = (v .= -abs2.(vec(x) ./ scale .- μ) ./ (2variance))
            state = initialize(Philox4x64(3), BatchedLogDensity(scalar, batch!), device(initial);
                move=GaussianReplacementMove(), executor)
            step!(state)
            saved = snapshot(state)
            return only(Array(saved.candidates[4])) / scale, Array(saved.acceptance_probabilities)[4]
        end
        y, p = run(floatmax(Float64))
        @test y ≈ first(run(1.0))
        @test p ≈ 1
    end

    @testset "Linear proposals across the coordinate range" begin
        # These seeds put walker i in the first group with difference a - b.
        for (initial, move, seed, i, a, b) in (
            ([-1.0 1.0], StretchMove(), 1, 1, 1, 2),
            ([0.1 -1.0], StretchMove(), 28, 1, 1, 2),
            ([-1.0 1.0 0.0 0.0], DEMove(gamma0=0.5, sigma=0), 25, 4, 2, 1),
            ([1.0 -1.0 0.0 -1.0], DEMove(gamma0=1, sigma=0), 8, 4, 1, 2),
        )
            function run(scale)
                scalar(x) = -abs2(x[1] / scale) / 2
                batch!(v, x) = (v .= -abs2.(vec(x) ./ scale) ./ 2)
                target = executor isa KernelExecutor ? BatchedLogDensity(scalar, batch!) : scalar
                state = initialize(Philox4x64(seed), target, device(initial .* scale); move, executor)
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

    @testset "Four-walker snooker starting geometry" begin
        key = Philox4x64(725)
        trapped = [0.0 3 1 1; 0 0 1 2]
        target = BatchedLogDensity(x -> 0.0, (v, x) -> fill!(v, 0.0))
        moves = (DESnookerMove(), StretchMove())
        for move in (first(moves), MoveMixture(moves, (1, 0)))
            @test_throws ArgumentError initialize(key, target, device(trapped); move, executor)
        end
        near = copy(trapped)
        near[1, 4] = nextfloat(near[1, 4])
        @test initialize(key, target, device(near); move=first(moves), executor) isa EnsembleMCMC.EnsembleState
        mobile = copy(trapped)
        mobile[1, 4] = 2
        for (initial, move) in ((mobile, first(moves)), (trapped, MoveMixture(moves, (1, 1); schedule=:cycle)))
            state = initialize(key, target, device(initial); move, executor)
            @test reduce(hcat, Array.(current_state(step!(state, 8)).positions)) != initial
        end
        state = initialize(key, target, device(Float32.(mobile)); move=first(moves), executor)
        step!(state)
        saved = snapshot(state)
        # The one-ULP Float64 perturbation rounds back onto the trap in Float32.
        @test_throws ArgumentError synchronize!(state, device(near), zeros(4))
        @test Array.(current_state(state).positions) == Array.(saved.positions)
        @test Array(current_state(state).accepted) == Array(saved.accepted)
        @test Array(current_state(state).logdensities) == Array(saved.logdensities)
        synchronize!(state, device(mobile), zeros(4))
        @test reduce(hcat, Array.(current_state(state).positions)) == mobile
        @test reduce(hcat, Array.(current_state(step!(state, 8)).positions)) != mobile
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
        state = initialize(Philox4x64(21), target, device(initial); move, executor)
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

    @testset "Unrepresentable linear candidates skip the target" begin
        initial = Float16[-30000 -10000 10000 30000]
        key = Philox4x64(6)
        finite_batches = Vector{Vector{Float16}}()
        for width in (nothing, 3)
            calls = Vector{Vector{Float16}}()
            function batch!(v, x)
                push!(calls, vec(Array(x)))
                all(isfinite, last(calls)) || error("nonfinite batch input")
                fill!(v, 0.0)
            end
            target = BatchedLogDensity(x -> 0.0, batch!; batch_size=width)
            start = isnothing(width) ? initial : initial ./ Float16(256)
            state = initialize(key, target, device(start); executor)
            isnothing(width) || synchronize!(state, device(initial), zeros(4))
            saved = current_state(step!(state))
            bad = .!isfinite.(map(x -> only(Array(x)), saved.candidates))
            @test count(bad) == 1
            @test map(x -> only(Array(x)), saved.positions)[bad] == vec(initial)[bad]
            @test all(==(-Inf), Array(saved.candidate_logdensities)[bad])
            @test all(iszero, Array(saved.acceptance_probabilities)[bad])
            @test !any(Array(saved.accepted)[bad])
            if isnothing(width)
                @test sort(length.(calls)) == [1, 2]
                append!(finite_batches, calls)
            else
                @test calls == [vcat(batch, fill(first(batch), width-length(batch))) for batch in finite_batches]
            end
        end
    end

    @testset "Accepted linear proposals can outgrow their initial range" begin
        batch!(v, x) = (all(isfinite, x) || error("nonfinite batch input"); fill!(v, 0.0))
        state = initialize(Philox4x64(77), BatchedLogDensity(x -> 0.0, batch!),
            device(Float16[-1 0 1 2]); move=DEMove(gamma0=4, sigma=0), executor)
        overflow = false
        for _ in 1:16
            saved = current_state(step!(state))
            overflow |= any(x -> !isfinite(only(Array(x))), saved.candidates)
        end
        @test overflow
        @test all(x -> isfinite(only(Array(x))), current_state(state).positions)
    end

    @testset "Zero and unrepresentable linear displacements" begin
        function scalar(x)
            isfinite(only(x)) || error("nonfinite scalar input")
            return -abs2(x[1] / 1e308) / 2
        end
        function batch!(v, x)
            all(isfinite, x) || error("nonfinite batch input")
            v .= -abs2.(vec(x) ./ 1e308) ./ 2
        end
        target = executor isa KernelExecutor ? BatchedLogDensity(scalar, batch!) : scalar
        tiny = nextfloat(0.0)
        # Walker 4 picks the opposite-sign companions and its jitter rounds the factor to zero.
        state = initialize(Philox4x64(105), target, device([-1e308 1e308 tiny tiny]);
            move=DEMove(gamma0=tiny, sigma=1), executor)
        step!(state)
        saved = snapshot(state)
        @test only(Array(saved.candidates[4])) == tiny
        @test Array(saved.accepted)[4]

        for initial in ([-1e308 1e308 0.0 0.0], [1e308 0.0 0.0 0.0])
            state = initialize(Philox4x64(1), target, device(initial);
                move=DEMove(gamma0=2, sigma=0), executor)
            step!(state)
            saved = snapshot(state)
            unrepresentable = .!isfinite.(map(x -> only(Array(x)), saved.candidates))
            @test any(unrepresentable)
            @test all(iszero, Array(saved.acceptance_probabilities)[unrepresentable])
        end

        # This seed overflows the jittered factor for all four walkers.
        state = initialize(Philox4x64(2), target, device([-1.0 1.0 0.0 0.0]);
            move=DEMove(gamma0=floatmax(Float64), sigma=10), executor)
        step!(state)
        saved = snapshot(state)
        @test all(x -> isinf(only(Array(x))), saved.candidates)
        @test !any(Array(saved.accepted))
    end
end
