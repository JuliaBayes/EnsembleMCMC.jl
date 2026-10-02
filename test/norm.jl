@testset "Snooker geometry across coordinate scales" begin
    for T in (Float32, Float64)
        initial = T[-1 0 1 0 -1 -1 1 1; 0 -1 0 1 -1 1 -1 1]
        reference = sample!(initialize(test_rng(), _ -> zero(T), initial;
            move=DESnookerMove()), 1)
        exponent = T === Float32 ? 25 : 200
        for scale in (T(10)^(-exponent), T(10)^exponent)
            draws = sample!(initialize(test_rng(), _ -> zero(T), initial .* scale;
                move=DESnookerMove()), 1)
            @test draws.accepted == reference.accepted
            @test draws.positions ./ scale ≈ reference.positions rtol=100eps(T)
        end
    end
end

@testset "Snooker geometry far from the origin" begin
    initial = 1.4e308 .+ 1e294 .* [-1.0 0 1 0 -1 -1 1 1; 0 -1 0 1 -1 1 -1 1]
    calls = Ref(0)
    target(x) = (calls[] += 1; -sum(abs2, (x .- 1.4e308) ./ 1e294) / 2)
    state = initialize(test_rng(), target, initial; move=DESnookerMove())
    step!(state)
    @test calls[] == 2size(initial, 2)
end

@testset "Snooker projections beyond the coordinate range" begin
    initial = floatmax(Float64) .* [0 .1 .6 -.6; 0 .6 0 0]
    calls = Ref(0)
    target(x) = (calls[] += 1; -sum(abs2, x ./ floatmax(Float64)) / 2)
    state = initialize(Philox4x((1, 19)), target, initial; move=DESnookerMove())
    step!(state)
    @test calls[] > size(initial, 2)
end
