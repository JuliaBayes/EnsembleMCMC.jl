@testset "Position validation" begin
    positions = Float32[-1 0 1 0; 0 1 0 -1]
    saved = copy(positions)
    validate_positions(positions)
    validate_positions(collect(eachcol(positions)))
    @test positions == saved
    collapsed = copy(positions)
    collapsed[2, :] .= 0
    @test_throws ArgumentError validate_positions(collapsed)
    nonfinite = copy(positions)
    nonfinite[1, 1] = Inf32
    @test_throws ArgumentError validate_positions(nonfinite)
end

@testset "Float16 affine rank across population sizes" begin
    for n in (1023, 1024, 1025)
        positions = reshape(Float16.(isodd.(1:n)), 1, :)
        @test validate_positions(positions) === nothing
        @test_throws ArgumentError validate_positions(vcat(positions, positions))
    end
end

@testset "Position validation across numeric scales" begin
    for positions in (
        reshape([-1e308, 1e308], 1, :),
        reshape(repeat(Float32[0, 1f38], 16), 1, :),
        Float32[-1 0 1 0; 0 1 0 -1] .* nextfloat(0f0),
    )
        state = initialize(test_rng(), _ -> 0.0, positions)
        @test reduce(hcat, current_state(state).positions) == positions
    end
end

@testset "Log-density offsets preserve acceptance" begin
    initial = Float32[-1 0 1 0 -1 -1 1 1; 0 -1 0 1 -1 1 -1 1]
    for batched in (false, true)
        states = map((0f0, -1f10)) do offset
            scalar(x) = maximum(abs, x) < 100 ? offset : -Inf32
            batch!(v, x) = map!(scalar, v, eachcol(x))
            target = batched ? BatchedLogDensity(scalar, batch!) : scalar
            initialize(test_rng(), target, initial)
        end
        draws = map(state -> sample!(state, 10), states)
        @test first(draws).positions == last(draws).positions
        @test first(draws).accepted == last(draws).accepted
        @test current_state(first(states)).acceptance_probabilities ==
            current_state(last(states)).acceptance_probabilities
    end
end

@testset "Cached initial log densities" begin
    positions = [Float32.(x) for x in initial_walkers()]
    calls = Ref(0)
    target(x) = (calls[] += 1; -sum(abs2, Float64.(x)) / 2)
    reference = initialize(test_rng(), target, positions)
    values = copy(current_state(reference).logdensities)
    calls[] = 0
    cached = initialize(test_rng(), target, reduce(hcat, positions); logdensities=values)
    @test calls[] == 0
    @test eltype(current_state(cached).logdensities) === Float64
    @test snapshot(cached) == snapshot(reference)
    values .= 100
    @test snapshot(cached) == snapshot(reference)
    step!(cached, 8)
    step!(reference, 8)
    @test snapshot(cached) == snapshot(reference)
    @test_throws DimensionMismatch initialize(test_rng(), target, positions; logdensities=[0.0])
    for invalid in (NaN, Inf)
        @test_throws DomainError initialize(test_rng(), target, positions; logdensities=fill(invalid, length(positions)))
    end
    @test_throws ArgumentError initialize(test_rng(), target, positions; logdensities=fill(1im, length(positions)))
    negative_infinity = @test_logs (:warn, r"All 24 initial log densities are -Inf") initialize(
        test_rng(), target, positions; logdensities=fill(-Inf, length(positions)))
    @test all(==(-Inf), current_state(negative_infinity).logdensities)
end

@testset "Transition details and synchronization" begin
    target(x) = -sum(abs2, x) / 2 - sum(abs2, x)^2 / 10
    state = initialize(test_rng(), target, initial_walkers(); move=DEMove())
    before = snapshot(state)
    step!(state)
    after = snapshot(state)
    @test after.candidate_logdensities ≈ target.(after.candidates)
    @test after.acceptance_probabilities ≈ min.(1, exp.(after.candidate_logdensities .- before.logdensities))
    @test after.positions == [after.accepted[i] ? after.candidates[i] : before.positions[i] for i in eachindex(after.accepted)]
    saved = deepcopy(after)
    step!(state)
    @test after == saved

    positions = [x .+ 0.25 for x in after.positions]
    expected = deepcopy(positions)
    counters = snapshot(state)
    synchronize!(state, positions, target.(positions))
    positions[1] .= 1e6
    current = current_state(state)
    @test current.positions == expected
    @test current.logdensities == target.(expected)
    @test (current.sweep_count, current.attempts, current.acceptances) ==
        (counters.sweep_count, counters.attempts, counters.acceptances)
    @test !any(current.accepted) && all(iszero, current.acceptance_probabilities)
    saved = snapshot(state)
    invalid = copy(saved.logdensities)
    invalid[end] = NaN
    @test_throws ArgumentError synchronize!(state, positions, invalid)
    @test snapshot(state) == saved
end

@testset "Externally addressed sweeps advance cyclic mixture phase" begin
    root = Philox4x64(UInt64.((991, 19)))
    lp, initial = gaussian_logdensity, initial_walkers()
    chain = subrng(root, 2)
    move = MoveMixture((StretchMove(), DEMove()), [1, 1]; schedule=:cycle)
    serial = initialize(chain, lp, initial; move)
    threaded = initialize(chain, lp, initial; move, executor=ThreadedExecutor())
    indices = Int[]
    for outer_step in (1, 3, 5, 7)
        key = subrng(root, 2, 3, outer_step)
        advanced = last(rand_next(key, UInt64))
        step!(serial, key; proposal_index=2)
        step!(threaded, advanced; proposal_index=2)
        @test snapshot(serial) == snapshot(threaded)
        push!(indices, current_state(serial).move_index)
    end
    @test indices == [1, 2, 1, 2]
    @test current_state(serial).sweep_count == length(indices)

    function candidates(key, proposal_index)
        state = initialize(chain, lp, initial; move=DEMove())
        return current_state(step!(state, key; proposal_index)).candidates
    end
    address = subrng(root, 2, 3, 0)
    expected = candidates(address, 2)
    for key in (subrng(root, 3, 3, 0), subrng(root, 2, 4, 0), subrng(root, 2, 3, 1),
        subrng(Philox4x64R7(UInt64.((991, 19))), 2, 3, 0), subrng(Threefry4x64(UInt64.((991, 19, 0, 0))), 2, 3, 0))
        @test candidates(key, 2) != expected
    end
    @test candidates(address, 3) != expected
end

@testset "Nonfinite starts and degenerate transitions" begin
    target(x) = x[1] > 0 ? -x[1]^2 / 2 : -Inf
    state = @test_logs (:warn, r"1 of 4 initial log densities are -Inf") initialize(
        test_rng(), target, [[-1.0], [1.0], [2.0], [3.0]]; move=DEMove())
    step!(state, 16)
    @test all(isfinite, current_state(state).logdensities)
    coincident = [[0.0], [0.0], [0.0], [1.0]]
    snooker = initialize(test_rng(), gaussian_logdensity, coincident; move=DESnookerMove())
    step!(snooker)
    transition = current_state(snooker)
    @test transition.candidate_logdensities == gaussian_logdensity.(transition.candidates)
    @test all(iszero, transition.acceptance_probabilities[.!transition.accepted])
end
