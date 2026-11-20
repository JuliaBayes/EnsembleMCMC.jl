@testset "Gaussian replacement move" begin
    @testset "Gamma stationarity" begin
        target(x) = x[1] > 0 ? 2log(x[1]) - x[1] : -Inf
        initial = reshape(collect(range(0.5, 6; length=32)), 1, :)
        state = initialize(test_rng(), target, initial; move=GaussianReplacementMove())
        step!(state, 300)
        draws = vec(sample!(state, 1_500).positions)
        @test mean(draws) ≈ 3 atol=0.12 rtol=0
        @test var(draws) ≈ 3 atol=0.25 rtol=0
    end

    @testset "Thread, walker ID, and resumed mixture replay" begin
        initial = initial_walkers()
        permutation = reverse(eachindex(initial))
        move = MoveMixture((GaussianReplacementMove(), DEMove()), [2, 1]; schedule=:cycle)
        reference = initialize(test_rng(), gaussian_logdensity, initial; move)
        threaded = initialize(test_rng(), gaussian_logdensity, initial;
            move, executor=ThreadedExecutor())
        reordered = initialize(test_rng(), gaussian_logdensity, initial[permutation];
            move, walker_ids=collect(permutation))
        expected = sample!(reference, 18)
        @test sample!(threaded, 18) == expected
        first_part = sample!(reordered, 7)
        second_part = sample!(reordered, 11)
        @test cat(first_part.positions, second_part.positions; dims=3)[:, permutation, :] ==
            expected.positions
        @test hcat(first_part.logdensities, second_part.logdensities)[permutation, :] ==
            expected.logdensities
        @test hcat(first_part.accepted, second_part.accepted)[permutation, :] == expected.accepted
        @test vcat(first_part.move_indices, second_part.move_indices) == expected.move_indices
    end

    @testset "Batched target" begin
        initial = initial_walkers()
        scalar_calls = Ref(0)
        widths = Int[]
        scalar(x) = (scalar_calls[] += 1; gaussian_logdensity(x))
        batch!(values, positions) = begin
            push!(widths, size(positions, 2))
            values .= gaussian_logdensity.(eachcol(positions))
        end
        move = GaussianReplacementMove(shrinkage=0)
        reference = initialize(test_rng(), gaussian_logdensity, initial; move)
        batched = initialize(test_rng(), BatchedLogDensity(scalar, batch!), initial; move)
        @test sample!(batched, 5) == sample!(reference, 5)
        @test scalar_calls[] == length(initial)
        @test widths == fill(length(initial) ÷ 2, 10)
    end

    @testset "Failed complement fits omit callbacks" begin
        # Full affine rank globally, but every three-point complement is singular.
        initial = [-1.0 0 0 1 0 0; 0 -1 0 0 1 0; 0 0 -1 0 0 1]
        calls = Ref(0)
        target(x) = (calls[] += 1; gaussian_logdensity(x))
        # Positive shrinkage makes the minimum walker count sufficient for this fit.
        state = initialize(test_rng(), target, initial; move=GaussianReplacementMove())
        step!(state)
        @test calls[] == 2size(initial, 2)

        # Every partition has at least one singular complement. Reject proposals
        # so the first group cannot change the second group's fitting geometry.
        initial = [0.0 0 0 0 0 0 1 0; 0 0 0 0 0 0 0 1]
        calls[] = 0
        reject_proposals(x) = (calls[] += 1; calls[] <= size(initial, 2) ? 0.0 : -Inf)
        state = initialize(Philox4x64(UInt64.((1, 19))), reject_proposals, initial;
            move=GaussianReplacementMove(shrinkage=0))
        draws = sample!(state, 1)
        @test calls[] in (size(initial, 2), 3size(initial, 2) ÷ 2)
        @test !any(draws.accepted)
        @test draws.positions[:, :, 1] == initial

        # The seed puts the three constant points in one group.
        initial = [1.0 1 1 0 0 2; 0.1 0.1 0.1 0 1 0]
        calls[] = 0
        state = initialize(Philox4x64(13), reject_proposals, initial;
            move=GaussianReplacementMove())
        draws = sample!(state, 1)
        @test calls[] == 3size(initial, 2) ÷ 2
        @test !any(draws.accepted)
        @test draws.positions[:, :, 1] == initial
        initial = initial .- initial[:, 1]
        calls[] = 0
        step!(initialize(Philox4x64(13), reject_proposals, initial;
            move=GaussianReplacementMove()))
        @test calls[] == 3size(initial, 2) ÷ 2
    end

    @testset "Shrinkage options" begin
        @test GaussianReplacementMove().shrinkage === :auto
        @test GaussianReplacementMove(shrinkage=0.25).shrinkage == 0.25
        @test_throws ArgumentError GaussianReplacementMove(shrinkage=:none)
        @test_throws ArgumentError GaussianReplacementMove(shrinkage=1.5)
        @test EnsembleMCMC.minimum_walkers(GaussianReplacementMove(), 3) == 6
        @test EnsembleMCMC.minimum_walkers(GaussianReplacementMove(shrinkage=0), 3) == 8
        @test EnsembleMCMC.minimum_walkers(GaussianReplacementMove(shrinkage=0.5), 1) == 4
    end

    function gaussian_distances(fit, scale, points)
        group = axes(points, 2)
        distances = similar(points, length(group))
        work = similar(points, length(group), size(points, 1))
        EnsembleMCMC._gaussian_distances!(work, eachcol(points), group, fit, scale) do i, distance
            distances[i] = 2distance
        end
        return distances
    end

    @testset "Group distances against LinearAlgebra, $T, d=$d" for T in (Float32, Float64),
        d in (1, 5, 128, 129)
        rng = Philox4x64(UInt64.((11, d)))
        A, rng = randn_next(rng, T, d, d)
        factor = Matrix(cholesky(Symmetric(A * A' + d * I)).L)
        points, _ = randn_next(rng, T, d, 3)
        fit = (; anchor=zeros(T, d), mean=zeros(T, d), factor)
        reference = [sum(abs2, LowerTriangular(factor) \ v) / 2 for v in eachcol(points)]
        @test gaussian_distances(fit, one(T), points) ≈ reference rtol=1000eps(T)
    end

    @testset "Fit against LinearAlgebra, $T, d=$d, shrinkage=$s" for T in (Float32, Float64),
        d in (1, 3, 10), s in (0, 0.3, :auto)
        rng = Philox4x64(UInt64.((13, d)))
        n = 2d + 3
        A, rng = randn_next(rng, T, d, d)
        points, rng = randn_next(rng, T, d, n)
        shift, _ = randn_next(rng, T, d)
        X = A * points .+ 10shift
        anchor, μ, factor = zeros(T, d), zeros(T, d), zeros(T, d, d)
        shrinkage = s isa Symbol ? s : T(s)
        scale, valid = EnsembleMCMC._fit_gaussian!(anchor, μ, factor, copy(X), n, shrinkage)
        @test valid
        Y = Float64.(X)
        C = cov(Y; dims=2)
        expected = s === :auto ? (1 + d / n) * C : (1 - s) * C + s * tr(C) / d * I
        L = LowerTriangular(Float64.(factor))
        @test anchor .+ scale .* μ ≈ vec(mean(Y; dims=2)) rtol=sqrt(eps(T))
        @test Float64(scale)^2 * L * L' ≈ expected rtol=sqrt(eps(T))
    end

    @testset "Auto shrinkage falls back when the complement is too small" begin
        X = first(randn_next(Philox4x64(UInt64.((17, 1))), 4, 4))
        fit(s) = begin
            anchor, μ, factor = zeros(4), zeros(4), zeros(4, 4)
            scale, valid = EnsembleMCMC._fit_gaussian!(anchor, μ, factor, copy(X), 4, s)
            (; anchor, μ, factor, scale, valid)
        end
        @test fit(:auto) == fit(0.5)
        @test fit(:auto).valid
    end

    @testset "Rank of the frozen covariance in coordinate units" begin
        # The first complementary group is columns [2, 4, 1].
        group = [5, 6, 3]
        X = [4.0 0 4 1 0 1; 4 0 5 1 1 2]
        # The rank-one complement requires half shrinkage, giving mean (5/3, 5/3)
        # and covariance (13/3) * [1 1/2; 1/2 1]. Set the target equal to this fit.
        function target(x)
            u, v = x .- 5/3
            return -2/13 * (u^2 - u*v + v^2)
        end
        function run(X, shrinkage)
            state = initialize(Philox4x64(3), target, X; move=GaussianReplacementMove(; shrinkage))
            step!(state)
            return snapshot(state)
        end
        @test run(X, :auto).acceptance_probabilities[group] ≈ ones(3)
        X[:, [2, 4, 1]] .= [-3 1 2; -3 1 2]
        rejected = run(X, 0)
        @test reduce(hcat, rejected.candidates[group]) == X[:, group]
        @test !any(rejected.accepted[group])

        X = [0.0 0 2 1 3 1; 1 0 -1 0 1 2]
        for s in (:auto, 0)
            expected = reduce(hcat, run(X, s).candidates[group])
            scaled = copy(X)
            scaled[2, [2, 4, 1]] .*= 1e-10
            actual = reduce(hcat, run(scaled, s).candidates[group])
            actual[2, :] ./= 1e-10
            @test actual ≈ expected
        end
    end

    @testset "Float32 fit far from the origin" begin
        X = first(randn_next(Philox4x64(UInt64.((19, 1))), Float32, 2, 40)) .+ 1f6
        anchor, μ, factor = zeros(Float32, 2), zeros(Float32, 2), zeros(Float32, 2, 2)
        scale, valid = EnsembleMCMC._fit_gaussian!(anchor, μ, factor, copy(X), 40, 0f0)
        @test valid
        L = LowerTriangular(Float64.(factor))
        # Scaling before the anchor subtraction gave a relative error near 5e-3.
        @test Float64(scale)^2 * L * L' ≈ cov(Float64.(X); dims=2) rtol=1e-5
    end

    @testset "Auto shrinkage has an affine-invariant Hastings ratio" begin
        # The Cholesky factor is not equivariant, so compare densities, not draws.
        matrix = [1e3 2.0 0; -0.5 1e-2 0; 0 1 1]
        shift = [1e4, -3.0, 0.5]
        X = first(randn_next(Philox4x64(UInt64.((29, 1))), 3, 12))
        points = first(randn_next(Philox4x64(UInt64.((29, 2))), 3, 5))
        distances(X, points) = begin
            fit = (anchor=zeros(3), mean=zeros(3), factor=zeros(3, 3))
            scale, valid = EnsembleMCMC._fit_gaussian!(fit.anchor, fit.mean, fit.factor,
                copy(X), size(X, 2), :auto)
            @test valid
            gaussian_distances(fit, scale, points)
        end
        @test distances(matrix * X .+ shift, matrix * points .+ shift) ≈ distances(X, points) rtol=1e-8
    end

    @testset "Ill-conditioned rotated target keeps acceptance" begin
        # Shrinkage 0.5 toward tr(C)/d * I gives acceptance near 0.003 here.
        d = 10
        Q = Matrix(qr(first(randn_next(Philox4x64(UInt64.((23, 1))), d, d))).Q)
        variances = exp.(range(0, log(1e3), length=d))
        precision = Symmetric(Q * Diagonal(inv.(variances)) * Q')
        L = cholesky(Symmetric(Q * Diagonal(variances) * Q')).L
        initial = [L * first(randn_next(Philox4x64(UInt64.((23, i + 1))), d)) for i in 1:100]
        state = initialize(test_rng(), x -> -dot(x, precision * x) / 2, initial;
            move=GaussianReplacementMove())
        step!(state, 300)
        current = current_state(state)
        @test sum(current.acceptances) / sum(current.attempts) > 0.3
    end

    @testset "Extreme coordinate rescaling" begin
        initial = Float32[-1 0 1 0 -1 -1 1 1; 0 -1 0 1 -1 1 -1 1]
        reference = sample!(initialize(test_rng(), _ -> 0f0, initial;
            move=GaussianReplacementMove()), 4)
        for scale in (1f-25, 1f25)
            draws = sample!(initialize(test_rng(), _ -> 0f0, initial .* scale;
                move=GaussianReplacementMove()), 4)
            @test draws.accepted == reference.accepted
            @test draws.positions ./ scale ≈ reference.positions rtol=100eps(Float32)
        end
    end
end
