using EnsembleMCMC
using LinearAlgebra
using Random
using Random123
using Statistics
using Test

gaussian_logdensity(x) = -sum(abs2, x) / 2
initial_walkers() = [randn(Philox4x((731, i)), 2) for i in 1:24]
test_rng() = Philox4x((573, 19))

@testset "EnsembleMCMC" begin
    @testset "Gaussian target: $(typeof(move))" for move in
        (StretchMove(), DEMove(), DESnookerMove())
        state = initialize(test_rng(), gaussian_logdensity, initial_walkers(); move)
        step!(state, 500)
        draws = sample!(state, 2_000)
        coordinates = reshape(draws.positions, 2, :)
        @test maximum(abs, vec(mean(coordinates; dims=2))) < 0.12
        @test cov(coordinates; dims=2) ≈ Matrix{Float64}(I, 2, 2) atol=0.15 rtol=0
    end

    @testset "Affine trajectory: $(typeof(move))" for move in
        (StretchMove(), DEMove())
        matrix = [1.3 0.4; -0.2 0.8]
        shift = [-0.6, 1.1]
        initial = initial_walkers()
        target(y) = gaussian_logdensity(matrix \ (y - shift))
        reference = initialize(test_rng(), gaussian_logdensity, initial; move)
        transformed = initialize(test_rng(), target, [matrix * x + shift for x in initial]; move)
        expected = sample!(reference, 32)
        actual = sample!(transformed, 32)
        @test actual.accepted == expected.accepted
        @test all(
            actual.positions[:, w, s] ≈ matrix * expected.positions[:, w, s] + shift
            for w in axes(expected.positions, 2), s in axes(expected.positions, 3)
        )
    end

    @testset "Thread and walker order replay" begin
        initial = initial_walkers()
        permutation = reverse(eachindex(initial))
        move = MoveMixture((StretchMove(), DEMove(), DESnookerMove()), [1, 1, 1]; schedule=:cycle)
        reference = initialize(test_rng(), gaussian_logdensity, initial; move)
        threaded = initialize(test_rng(), gaussian_logdensity, initial;
            move, executor=ThreadedExecutor())
        reordered = initialize(test_rng(), gaussian_logdensity, initial[permutation];
            move, walker_ids=collect(permutation))
        expected = sample!(reference, 30)
        actual = sample!(threaded, 30)
        reordered_draws = sample!(reordered, 30)
        @test actual == expected
        @test reordered_draws.positions[:, permutation, :] == expected.positions
        @test reordered_draws.logdensities[permutation, :] == expected.logdensities
        @test reordered_draws.accepted[permutation, :] == expected.accepted
        @test reordered_draws.walker_ids[permutation] == expected.walker_ids
    end

    @testset "Resume and snapshot ownership" begin
        initial = initial_walkers()
        saved_initial = deepcopy(initial)
        move = MoveMixture((StretchMove(), DEMove()), [2, 1]; schedule=:cycle)
        state = initialize(test_rng(), gaussian_logdensity, initial; move)
        reference = initialize(test_rng(), gaussian_logdensity, saved_initial; move)
        initial[1][1] = 1e6
        first_part = sample!(state, 7)
        second_part = sample!(state, 11)
        expected = sample!(reference, 18)
        @test cat(first_part.positions, second_part.positions; dims=3) == expected.positions
        @test vcat(first_part.move_indices, second_part.move_indices) == expected.move_indices
        @test hcat(first_part.logdensities, second_part.logdensities) == expected.logdensities
        @test hcat(first_part.accepted, second_part.accepted) == expected.accepted
        snapshot = deepcopy(second_part)
        @test step!(state) === state
        @test second_part == snapshot
        second_part.positions[1, 1, end] = 1e6
        @test current_state(state).positions == current_state(step!(reference)).positions
        @test current_state(state).sweep_count == 19
    end

    @testset "Fixed mixture selection" begin
        moves = (StretchMove(), DEMove())
        @test_throws ArgumentError MoveMixture(moves, [2, 1]; schedule=:invalid)
        @test_throws ArgumentError MoveMixture(moves, [0.7, 0.3]; schedule=:cycle)
        deterministic = initialize(test_rng(), gaussian_logdensity, initial_walkers();
            move=MoveMixture(moves, [2, 1]; schedule=:cycle))
        draws = sample!(deterministic, 9)
        @test draws.move_indices == repeat([1, 1, 2], 3)
        @test current_state(deterministic).attempts == [6, 3] .* length(initial_walkers())
        @test sum(current_state(deterministic).acceptances) == count(draws.accepted)
        random = initialize(test_rng(), gaussian_logdensity, initial_walkers();
            move=MoveMixture(moves, [0.7, 0.3]))
        random_draws = sample!(random, 100)
        @test 50 < count(==(1), random_draws.move_indices) < 90
        @test current_state(random).attempts ==
            [count(==(i), random_draws.move_indices) for i in 1:2] .* length(initial_walkers())
        integer_weights = initialize(test_rng(), gaussian_logdensity, initial_walkers();
            move=MoveMixture(moves, [7, 3]))
        float_weights = initialize(test_rng(), gaussian_logdensity, initial_walkers();
            move=MoveMixture(moves, [7.0, 3.0]; schedule=:random))
        @test sample!(integer_weights, 12) == sample!(float_weights, 12)
        float_cycle = initialize(test_rng(), gaussian_logdensity, initial_walkers();
            move=MoveMixture(moves, [2.0, 1.0]; schedule=:cycle))
        @test sample!(float_cycle, 9) == draws
    end

    @testset "Cached densities and degenerate snooker geometry" begin
        calls = Ref(0)
        target(x) = (calls[] += 1; gaussian_logdensity(x))
        initial = initial_walkers()
        state = initialize(test_rng(), target, initial)
        @test calls[] == length(initial)
        step!(state)
        @test calls[] == 2length(initial)
        calls[] = 0
        coincident = [[0.0], [0.0], [0.0], [1.0]]
        degenerate = initialize(test_rng(), target, coincident; move=DESnookerMove())
        step!(degenerate)
        @test calls[] < 2length(coincident)
        @test count(!, current_state(degenerate).accepted) > 0
    end

    @testset "Failed sweeps cannot resume" begin
        calls = Ref(0)
        enabled = Ref(false)
        initial = initial_walkers()
        target(x) = begin
            calls[] += 1
            enabled[] && calls[] > length(initial) + length(initial) ÷ 2 && error("target failed")
            gaussian_logdensity(x)
        end
        state = initialize(test_rng(), target, initial)
        enabled[] = true
        @test_throws ErrorException step!(state)
        before_retry = calls[]
        enabled[] = false
        @test_throws ArgumentError step!(state)
        @test_throws ArgumentError step!(state, 0)
        @test_throws ArgumentError sample!(state, 0)
        @test_throws ArgumentError current_state(state)
        @test_throws ArgumentError snapshot(state)
        @test calls[] == before_retry
    end

    @testset "Single-sweep integration and ownership" begin
        initial = initial_walkers()
        move = MoveMixture((StretchMove(), DEMove()), [2, 1]; schedule=:cycle)
        state = initialize(test_rng(), gaussian_logdensity, initial; move)
        reference = initialize(test_rng(), gaussian_logdensity, initial; move)
        first_state = current_state(state)
        @test first_state.sweep_count == first_state.move_index == 0
        @test !any(first_state.accepted) && all(iszero, first_state.attempts)
        @test first_state.positions === current_state(state).positions
        @test all(name -> getproperty(first_state, name) === getproperty(current_state(state), name),
            (:logdensities, :accepted, :walker_ids, :attempts, :acceptances))
        expected = sample!(reference, 3)
        for sweep in 1:3
            @test step!(state) === state
            current = current_state(state)
            @test reduce(hcat, current.positions) == expected.positions[:, :, sweep]
            @test current.logdensities == expected.logdensities[:, sweep]
            @test current.accepted == expected.accepted[:, sweep]
            @test current.move_index == expected.move_indices[sweep]
            @test current.walker_ids == expected.walker_ids
            @test current.sweep_count == sweep
        end
        saved = snapshot(state)
        expected_snapshot = deepcopy(saved)
        step!(state)
        @test saved == expected_snapshot
        saved.positions[1][1] = 1e6
        saved.logdensities[1] = 1e6
        saved.accepted[1] = !saved.accepted[1]
        saved.walker_ids[1] = 0
        saved.attempts[1] = 0
        saved.acceptances[1] = 0
        @test snapshot(state) == snapshot(step!(reference))
    end

    @testset "Matrix initialization and bulk stepping" begin
        initial = Float32.([-1 -1 1 1; -1 1 -1 1])
        ids = [4, 2, 3, 1]
        state = initialize(test_rng(), gaussian_logdensity, initial; walker_ids=ids)
        reference = initialize(test_rng(), gaussian_logdensity, collect(eachcol(initial)); walker_ids=ids)
        initial[1, 1] = 1e6
        @test step!(state, 5) === state
        for _ in 1:5
            step!(reference)
        end
        @test snapshot(state) == snapshot(reference)
        @test current_state(state).walker_ids == ids
        before = snapshot(state)
        @test step!(state, 0) === state
        @test_throws ArgumentError step!(state, -1)
        @test snapshot(state) == before
    end
end

include("norm.jl")
include("batched.jl")
include("kernel.jl")

include("integration.jl")
