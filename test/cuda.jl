# GPU tests. `Pkg.test()` does not run them. Run from the repository root on a CUDA machine:
#   julia -e 'using Pkg; Pkg.activate(temp=true); Pkg.develop(path="."); Pkg.add(["CUDA", "Random123", "Test"]); include("test/cuda.jl")'
# The tests use the current device. Call `CUDA.device!(i)` before the include to choose another one.
using CUDA, EnsembleMCMC, Random, Random123, Test
using EnsembleMCMC: initialize, step!, sample!, current_state, snapshot, synchronize!
test_rng() = Philox4x((573, 19))
include("kernel.jl")

# A device with no room for a context (for example one full from another job) cannot host the test.
function other_usable_device(current; required=2^28)
    for device in CUDA.devices()
        device == current && continue
        usable = try
            CUDA.device!(() -> CUDA.free_memory() >= required, device)
        catch err
            @info "Cannot use CUDA device $device" exception = err
            false
        end
        usable && return device
    end
    return nothing
end

function test_cuda()
    CUDA.allowscalar(false)
    test_kernel_executor(CuArray)

    @testset "CUDA storage and launch boundaries" begin
        initial = randn(MersenneTwister(31), 33, 132)
        scalar(x) = -sum(abs2, x) / 2
        batch!(v, x) = (v .= vec(-sum(abs2, x; dims=1) / 2))
        target = BatchedLogDensity(scalar, batch!)
        state = initialize(test_rng(), target, CuArray(initial); move=DEMove(), executor=KernelExecutor())
        expected = sample!(initialize(test_rng(), scalar, initial; move=DEMove()), 2)
        draws = sample!(state, 2)
        @test Array(draws.positions) ≈ expected.positions
        @test Array(draws.accepted) == expected.accepted
        saved = snapshot(state)
        @test all(x -> x isa CUDA.AnyCuArray,
            (draws.positions, draws.logdensities, draws.accepted, saved.positions[1], saved.logdensities, saved.accepted))

        replacement = CuArray(initial)
        other = other_usable_device(CUDA.device(replacement))
        if isnothing(other)
            @info "No second usable CUDA device, skipping the multi-device test"
        else
            logdensities = vec(-sum(abs2, replacement; dims=1)/2)
            CUDA.device!(other) do
                synchronize!(state, replacement, logdensities)
                @test CUDA.device() == other
                saved = snapshot(state)
                @test CUDA.device(saved.logdensities) == CUDA.device(replacement)
                @test Array(saved.logdensities) == Array(logdensities)
            end
        end

        event = CUDA.CuEvent()
        throws!(v, x) = (fill!(v, 0); CUDA.record(event); error("callback failed"))
        failed = initialize(test_rng(), BatchedLogDensity(scalar, throws!), CuArray(initial);
            executor=KernelExecutor())
        @test_throws ErrorException step!(failed)
        @test CUDA.isdone(event)
    end

    @testset "Wrapped CUDA arrays: $name" for (name, wrap) in (
        ("non-contiguous view", X -> view(X, 1:3, :)),
        ("transpose", X -> transpose(copy(transpose(X)))),
        ("adjoint", X -> copy(X')'),
        ("PermutedDimsArray", X -> PermutedDimsArray(copy(transpose(X)), (2, 1))),
    )
        scalar(x) = -sum(abs2, x) / 2
        batch!(v, x) = (v .= vec(-sum(abs2, x; dims=1) / 2))
        target = BatchedLogDensity(scalar, batch!)
        host = randn(MersenneTwister(8), 4, 16)
        initial = wrap(CuArray(host))
        @test initial isa CUDA.AnyCuArray && !(initial isa CuArray)
        reference = Array(initial)
        state = initialize(test_rng(), target, initial; move=DEMove(), executor=KernelExecutor())
        expected = sample!(initialize(test_rng(), scalar, reference; move=DEMove()), 3)
        draws = sample!(state, 3)
        @test Array(draws.positions) ≈ expected.positions
        @test Array(draws.accepted) == expected.accepted
    end
end

if CUDA.functional()
    test_cuda()
else
    @info "CUDA is not functional, skipping the CUDA tests"
end
