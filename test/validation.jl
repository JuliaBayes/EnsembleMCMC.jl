# Moment checks with bounds from the Monte Carlo standard error (see test_moments).

function run_chain(rng, target, initial; move, burn=1_000, nsamples=8_000, kwargs...)
    state = initialize(rng, target, initial; move, kwargs...)
    step!(state, burn)
    return state, sample!(state, nsamples)
end

@testset "Correlated Gaussian moments: $name $T" for T in (Float64, Float32), (name, move) in (
        ("Stretch", StretchMove()), ("DE", DEMove()), ("Snooker", DESnookerMove()),
        ("Gauss", GaussianReplacementMove(shrinkage=0)),
        ("Mix", MoveMixture((StretchMove(), DEMove(), GaussianReplacementMove()), [0.5, 0.3, 0.2])))
    covariance = [4.0 1.2; 1.2 0.5]
    center = [1.0, -2.0]
    precision = T.(inv(covariance))
    target(x) = (y = x .- T.(center); -dot(y, precision * y) / 2)
    initial = [T.(center .+ randn(Philox4x((17, i)), 2)) for i in 1:32]
    state, draws = run_chain(Philox4x((29, 3)), target, initial; move)
    @test eltype(draws.positions) === T
    test_moments(Float64.(draws.positions), center, covariance)
    rates = acceptance_rate(state)
    @test all(r -> 0.1 < r < 0.95, rates)
end

@testset "Stretch acceptance on a 2D Gaussian" begin
    state, _ = run_chain(test_rng(), gaussian_logdensity, initial_walkers(); move=StretchMove(),
        nsamples=2_000)
    # emcee reports about 0.7 for a = 2 in two dimensions.
    @test 0.6 < only(acceptance_rate(state)) < 0.8
end

@testset "Threefry4x Gaussian moments" begin
    _, draws = run_chain(Threefry4x((3, 1, 4, 1)), gaussian_logdensity, initial_walkers();
        move=StretchMove())
    test_moments(draws.positions, zeros(2), Matrix{Float64}(I, 2, 2))
end

@testset "Banana moments: $name" for (name, move) in (
        ("Stretch", StretchMove()), ("DE", DEMove()), ("Snooker", DESnookerMove()),
        ("Mix", MoveMixture((StretchMove(), DEMove(), GaussianReplacementMove()), [0.4, 0.4, 0.2])))
    # x1 ~ N(0, 1), x2 | x1 ~ N(b(x1^2 - 1), 1): E = (0, 0), Var(x2) = 1 + 2b^2.
    b = 1.0
    banana(x) = -x[1]^2 / 2 - (x[2] - b * (x[1]^2 - 1))^2 / 2
    initial = [randn(Philox4x((41, i)), 2) for i in 1:32]
    _, draws = run_chain(Philox4x((43, 5)), banana, initial; move, nsamples=16_000)
    test_moments(draws.positions, zeros(2), [1.0 0.0; 0.0 1 + 2b^2])
end
