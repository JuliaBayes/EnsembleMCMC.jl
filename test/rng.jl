@testset "Standalone and externally addressed mixture sweeps" begin
    moves = (StretchMove(), DEMove())
    mixture = MoveMixture(moves, (1, 1))
    initial = initial_walkers()
    ids = 3 .* eachindex(initial)
    permutation = reverse(eachindex(initial))

    for root in (Philox4x64(42), Threefry4x64(42))
        state = initialize(root, gaussian_logdensity, initial; move=mixture, walker_ids=ids)
        for (number, outer) in enumerate((nothing, 7, nothing))
            sweep = isnothing(outer) ? subrng(root, 0, number - 1) : subrng(root, 19)
            selection = isnothing(outer) ? subrng(sweep, 1, 1) : subrng(sweep, 2, outer)
            selected = first(rand_next(selection, Float64)) < 0.5 ? 1 : 2
            index = something(outer, selected)

            reference = initialize(root, gaussian_logdensity, current_state(state).positions[permutation];
                move=moves[selected], walker_ids=ids[permutation])
            step!(reference, last(rand_next(sweep, UInt64)); proposal_index=index)
            isnothing(outer) ? step!(state) : step!(state, sweep; proposal_index=outer)
            actual = current_state(state)
            @test actual.move_index == selected
            @test actual.candidates[permutation] == current_state(reference).candidates
            acceptance = subrng(sweep, 3, index)
            uniforms = [rand_at(acceptance, Float64, id) for id in ids]
            @test actual.accepted == (uniforms .< actual.acceptance_probabilities)
        end
    end
end

@testset "Equal-size bins for companion choices" begin
    key = Philox4x64(941)
    address = 19
    # At this address the retry stream's first three low-bit candidates are 6, 7, 1.
    # With six choices only the third succeeds and yields position 2.
    sparse = (companion=key, words=zeros(UInt64, 2), ids=[10])
    @test EnsembleMCMC._companion_index(sparse, 1, 2, 1, 6) == 2 # Logical address 19.
    span = big(2)^64
    for bin in 0:5
        first_word = cld(bin * span, 6)
        first_accepted = cld(bin * span + mod(span, 6), 6)
        last_accepted = cld((bin + 1) * span, 6) - 1
        for word in first_word:(first_accepted - 1)
            @test EnsembleMCMC._bounded_index(UInt64(word), key, address, 6) == 2
        end
        for word in (first_accepted, last_accepted)
            @test EnsembleMCMC._bounded_index(UInt64(word), key, address, 6) == bin + 1
        end
    end
end

@testset "Sparse Gaussian blocks match scalar walker addresses" begin
    key = Threefry2x32(83)
    for d in (3, 8, 13)
        initial = first(randn_next(Philox4x64(31), Float32, d, 2d + 2))
        ids = 31 .* collect(axes(initial, 2))
        ids[2] = typemax(Int32) - 2
        state = initialize(key, gaussian_logdensity, initial; move=GaussianReplacementMove(), walker_ids=ids)
        keys = (companion=subrng(key, 5), scale=subrng(key, 6), acceptance=subrng(key, 3))
        buffers = state.draw_buffers
        draws = EnsembleMCMC._draw_streams!(keys, buffers.words, buffers.floats, only(state.moves))
        expected = [randn_at(keys.scale, Float32, UInt64(id - 1) * d + j) for j in 1:d, id in ids]
        @test reshape(draws.scale, size(initial)) == expected
    end
end

@testset "Addressed DE sweep with sparse walker IDs" begin
    initial = reshape(Float64[0, 2, 8, 32, 128, 512, 2048], 1, :)
    ids = [14, 3, 50, 19, typemax(Int32) - 2, 2, 38]
    key = Philox4x64(83)
    move = DEMove(gamma0=0.5, sigma=0)
    state = initialize(key, x -> 0.0, initial; move, walker_ids=ids)
    step!(state, key; proposal_index=23)
    # The addressed shuffle updates walkers [5, 2, 7, 1], then [3, 6, 4].
    # Every proposal adds one half of its ordered companion difference.
    @test only.(current_state(state).positions) == [-252, 254, -1262, 1049, 116, 1598, 2288]
end

@testset "BigFloat Gaussian proposals keep the walker's precision" begin
    setprecision(BigFloat, 192) do
        key = Philox2x32(83)
        initial = reshape(BigFloat[-3, -1, 1, 3], 1, :)
        state = initialize(key, gaussian_logdensity, initial; move=GaussianReplacementMove(shrinkage=0))
        step!(state, key)
        # Walkers 1 and 4 update first. Their complement {-1, 1} has mean 0 and variance 2.
        for id in (1, 4)
            rng = StatefulRNG(subrng(key, 6, 1, id))
            expected = sqrt(BigFloat(2)) * randn(rng, BigFloat)
            @test only(current_state(state).candidates[id]) ≈ expected rtol=8eps(BigFloat)
        end
    end
end
