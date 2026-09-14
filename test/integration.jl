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
    negative_infinity = initialize(test_rng(), target, positions; logdensities=fill(-Inf, length(positions)))
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
end

@testset "Addressed sweeps and mixture phase" begin
    move = MoveMixture((StretchMove(), DEMove()), [1, 1]; schedule=:cycle)
    serial = initialize(test_rng(), gaussian_logdensity, initial_walkers(); move)
    threaded = initialize(test_rng(), gaussian_logdensity, initial_walkers(); move, executor=ThreadedExecutor())
    indices = Int[]
    for outer_step in (1, 3, 5, 7)
        rng = Philox4x((991, outer_step))
        expected_rng = rand(copy(rng), UInt64, 4)
        step!(serial, rng; proposal_index=2)
        step!(threaded, rng; proposal_index=2)
        @test rand(copy(rng), UInt64, 4) == expected_rng
        @test snapshot(serial) == snapshot(threaded)
        push!(indices, current_state(serial).move_index)
    end
    @test indices == [1, 2, 1, 2]
    saved = snapshot(serial)
    @test_throws ArgumentError step!(serial, Threefry4x((1, 2, 3, 4)))
    @test snapshot(serial) == saved
end

@testset "Nonfinite starts and degenerate transitions" begin
    target(x) = x[1] > 0 ? -x[1]^2 / 2 : -Inf
    state = initialize(test_rng(), target, [[-1.0], [1.0], [2.0], [3.0]]; move=DEMove())
    step!(state, 16)
    @test all(isfinite, current_state(state).logdensities)
    coincident = [[0.0], [0.0], [0.0], [1.0]]
    snooker = initialize(test_rng(), gaussian_logdensity, coincident; move=DESnookerMove())
    step!(snooker)
    transition = current_state(snooker)
    @test transition.candidate_logdensities == gaussian_logdensity.(transition.candidates)
    @test all(iszero, transition.acceptance_probabilities[.!transition.accepted])
end
