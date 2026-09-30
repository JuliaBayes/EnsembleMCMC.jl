error_of(f) = try f(); nothing; catch err; err; end

struct TargetFailure <: Exception
    walker::Vector{Float64}
end

@testset "Initialization errors" begin
    lp = gaussian_logdensity
    initial = initial_walkers()
    @test_throws ArgumentError initialize(test_rng(), lp, [randn(Philox4x((1, i)), 3) for i in 1:5])
    @test_throws ArgumentError initialize(test_rng(), lp, Vector{Float64}[])
    @test_throws ArgumentError initialize(test_rng(), lp, 3.0)
    for ids in (1:3, fill(1, 24), [0; 2:24], [1.5; 2:24], [typemax(Int32); 2:24], string.(1:24))
        @test_throws ArgumentError initialize(test_rng(), lp, initial; walker_ids=ids)
    end
    for rng in (Xoshiro(1), MersenneTwister(1), Philox4x(UInt32, (1, 2), 10))
        err = error_of(() -> initialize(rng, lp, initial))
        @test err isa ArgumentError && occursin("Philox4x", err.msg)
    end
    for move in (:foo, StretchMove, [StretchMove()])
        @test_throws ArgumentError initialize(test_rng(), lp, initial; move)
    end
    err = error_of(() -> initialize(test_rng(), lp, initial; nonsense=1))
    @test err isa ArgumentError && occursin("nonsense", err.msg) && length(err.msg) < 100
    @test_throws ArgumentError initialize(test_rng(), lp, reduce(hcat, initial); nonsense=1)
    @test_throws ArgumentError initialize(test_rng(), lp, initial; executor=KernelExecutor())
    nan_at_3(x) = x == initial[3] ? NaN : lp(x)
    err = error_of(() -> initialize(test_rng(), nan_at_3, initial))
    @test err isa DomainError && occursin("walker 3", err.msg)
    @test_throws ArgumentError initialize(test_rng(), x -> "bad", initial)

    moves = (StretchMove(), DEMove())
    @test sample!(initialize(test_rng(), lp, initial; move=MoveMixture(moves, (2, 1); schedule=:cycle)), 6) ==
        sample!(initialize(test_rng(), lp, initial; move=MoveMixture(moves, [2, 1]; schedule=:cycle)), 6)
    @test sample!(initialize(test_rng(), lp, initial; move=MoveMixture(moves, (0.5, 1))), 6) ==
        sample!(initialize(test_rng(), lp, initial; move=MoveMixture(moves, [0.5, 1.0])), 6)
end

@testset "RNG partition depth" begin
    derive(rng) = AbstractRNG(EnsembleMCMC.RNGPartition(copy(rng), 1:3), 2)
    lp, initial = gaussian_logdensity, initial_walkers()
    depth2 = derive(test_rng())
    state = initialize(depth2, lp, initial)
    @test step!(state, 2) === state
    depth3 = derive(depth2)
    err = error_of(() -> initialize(depth3, lp, initial))
    @test err isa ArgumentError && occursin("depth 3", err.msg)
    depth4 = derive(depth3)
    @test step!(state, depth4; proposal_index=2) === state
    saved = snapshot(state)
    err = error_of(() -> step!(state, derive(depth4)))
    @test err isa ArgumentError && occursin("depth 5", err.msg)
    @test snapshot(state) == saved
    @test step!(state) === state
end

@testset "Addressed proposal index bounds" begin
    state = initialize(test_rng(), gaussian_logdensity, initial_walkers())
    maximum_index = EnsembleMCMC._PROPOSALS_PER_PURPOSE
    for index in (0, -1, maximum_index + 1)
        @test_throws ArgumentError step!(state, Philox4x((1, 2)); proposal_index=index)
    end
    @test current_state(state).sweep_count == 0
    step!(state, Philox4x((1, 2)); proposal_index=maximum_index)
    @test current_state(state).sweep_count == 1
end

@testset "Threefry4x sampling" begin
    rng = Threefry4x((5, 6, 7, 8))
    first_run = sample!(initialize(rng, gaussian_logdensity, initial_walkers()), 20)
    @test first_run == sample!(initialize(rng, gaussian_logdensity, initial_walkers()), 20)
    @test first_run.positions != sample!(initialize(Threefry4x((5, 6, 7, 9)),
        gaussian_logdensity, initial_walkers()), 20).positions
    threaded = initialize(rng, gaussian_logdensity, initial_walkers(); executor=ThreadedExecutor())
    @test sample!(threaded, 20) == first_run
end

@testset "Synchronization shapes and continuation" begin
    target(x) = -sum(abs2, x) / 2
    state = initialize(test_rng(), target, initial_walkers(); move=DEMove())
    step!(state, 3)
    saved = snapshot(state)
    positions = [x .+ 0.5 for x in saved.positions]
    values = target.(positions)
    @test_throws DimensionMismatch synchronize!(state, positions[1:end-1], values[1:end-1])
    @test_throws DimensionMismatch synchronize!(state, positions, values[1:end-1])
    @test_throws ArgumentError synchronize!(state, [[x; 0.0] for x in positions], values)
    @test_throws ArgumentError synchronize!(state, vcat([[NaN, 0.0]], positions[2:end]), values)
    @test_throws ArgumentError synchronize!(state, positions, [Inf; values[2:end]])
    @test snapshot(state) == saved
    current = current_state(state)
    synchronize!(state, current.positions, current.logdensities)
    @test snapshot(state).positions == saved.positions

    other = initialize(test_rng(), target, [x ./ 2 for x in initial_walkers()]; move=DEMove())
    step!(other, 3)
    synchronize!(state, positions, values)
    synchronize!(other, reduce(hcat, positions), values)
    step!(state, 4)
    step!(other, 4)
    # Cumulative acceptance counts differ, because they include the sweeps before synchronization.
    @test all(name -> getproperty(snapshot(state), name) == getproperty(snapshot(other), name),
        (:positions, :logdensities, :candidates, :accepted, :acceptance_probabilities, :attempts))
    current = current_state(state)
    @test current.logdensities == target.(current.positions)
    @test current.sweep_count == 7
end

@testset "ThreadedExecutor errors and chunking" begin
    @test_throws ArgumentError ThreadedExecutor(min_chunk=0)
    initial = initial_walkers()
    calls = Threads.Atomic{Int}(0)
    failing(x) = (Threads.atomic_add!(calls, 1) > 30 && throw(TargetFailure(copy(x))); gaussian_logdensity(x))
    for executor in (SerialExecutor(), ThreadedExecutor())
        calls[] = 0
        state = initialize(test_rng(), failing, initial; executor)
        @test_throws TargetFailure step!(state)
        @test_throws ArgumentError current_state(state)
    end
    plus_infinity(x) = x[1] > 1 ? Inf : gaussian_logdensity(x)
    state = initialize(test_rng(), plus_infinity, [x ./ 4 for x in initial]; executor=ThreadedExecutor())
    err = error_of(() -> step!(state, 50))
    @test err isa DomainError && occursin("walker", err.msg) && occursin("sweep", err.msg)

    caller = current_task()
    off_caller = Threads.Atomic{Int}(0)
    counted(x) = (current_task() !== caller && Threads.atomic_add!(off_caller, 1); gaussian_logdensity(x))
    reference = sample!(initialize(test_rng(), gaussian_logdensity, initial), 5)
    for (min_chunk, spawns) in ((1, Threads.nthreads(:default) > 1), (12, false), (1000, false))
        off_caller[] = 0
        state = initialize(test_rng(), counted, initial; executor=ThreadedExecutor(; min_chunk))
        @test sample!(state, 5) == reference
        @test (off_caller[] > 0) == spawns
    end
end

@testset "NaN candidate densities reject" begin
    nan_calls = Threads.Atomic{Int}(0)
    target(x) = x[1] > 1 ? (Threads.atomic_add!(nan_calls, 1); NaN) : gaussian_logdensity(x)
    batch!(values, positions) = map!(target, values, eachcol(positions))
    initial = [x ./ 4 for x in initial_walkers()]
    runs = map((SerialExecutor(), ThreadedExecutor())) do executor
        scalar = initialize(test_rng(), target, initial; executor)
        batched = initialize(test_rng(), BatchedLogDensity(target, batch!), reduce(hcat, initial); executor)
        draws = sample!(scalar, 200)
        @test draws == sample!(batched, 200)
        @test all(<=(1), draws.positions[1, :, :])
        @test all(isfinite, draws.logdensities)
        draws
    end
    @test runs[1] == runs[2]
    @test nan_calls[] > 0
    @test_throws DomainError initialize(test_rng(), target, [x .+ [2.0, 0.0] for x in initial])
end

@testset "Thinned samples" begin
    move = MoveMixture((StretchMove(), DEMove(), DESnookerMove()), [1, 2, 1]; schedule=:cycle)
    thinned = initialize(test_rng(), gaussian_logdensity, initial_walkers(); move)
    full = initialize(test_rng(), gaussian_logdensity, initial_walkers(); move)
    draws = sample!(thinned, 5; thin=3)
    reference = sample!(full, 15)
    kept = 3:3:15
    @test size(draws.positions, 3) == length(draws.move_indices) == 5
    @test draws.positions == reference.positions[:, :, kept]
    @test draws.logdensities == reference.logdensities[:, kept]
    @test draws.accepted == reference.accepted[:, kept]
    @test draws.move_indices == reference.move_indices[kept]
    @test snapshot(thinned) == snapshot(full)
    @test current_state(thinned).sweep_count == 15
    @test size(sample!(thinned, 0; thin=4).positions, 3) == 0
    @test current_state(thinned).sweep_count == 15
    @test_throws ArgumentError sample!(thinned, 3; thin=0)
    @test_throws ArgumentError sample!(thinned, typemax(Int); thin=2)
end

@testset "Acceptance rate accessor" begin
    move = MoveMixture((StretchMove(), DEMove()), [1, 0]; schedule=:cycle)
    state = initialize(test_rng(), gaussian_logdensity, initial_walkers(); move)
    @test all(isnan, acceptance_rate(state))
    step!(state, 10)
    rates = acceptance_rate(state)
    current = current_state(state)
    @test rates[1] == current.acceptances[1] / current.attempts[1]
    @test 0 < rates[1] < 1 && isnan(rates[2])
end

@testset "Generic and nearly degenerate positions" begin
    big = [BigFloat.(x) for x in initial_walkers()]
    validate_positions(big)
    state = initialize(test_rng(), gaussian_logdensity, big; move=DEMove())
    step!(state, 3)
    @test eltype(first(current_state(state).positions)) === BigFloat
    collapsed = reduce(hcat, big)
    collapsed[2, :] .= 1
    @test_throws ArgumentError validate_positions(collapsed)

    flat = reduce(hcat, initial_walkers())
    flat[2, :] .*= 1e-10
    @test_logs (:warn, r"nearly degenerate") validate_positions(flat)
    @test_logs (:warn, r"nearly degenerate") initialize(test_rng(), gaussian_logdensity, flat)
    flat[2, :] .= 0
    @test_throws ArgumentError validate_positions(flat)
    @test_logs validate_positions(reduce(hcat, initial_walkers()))
end
