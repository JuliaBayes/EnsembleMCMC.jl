using PureRNGs: Philox4x64, Threefry4x64, rand_at, subrng

function test_keyed_kernel_mixture(device, prepare_batch=(f, points, width) -> f)
    @testset "Keyed random kernel mixture" begin
        initial = reshape(collect(1.0:13.0), 1, :)
        ids = reverse(17 .* axes(initial, 2))
        points = device(initial)
        width = cld(size(initial, 2), 2)
        scalar(_) = 0.0
        batch!(values, positions) = (values .= vec(sum(positions .* 0; dims=1)); nothing)
        target = BatchedLogDensity(scalar, prepare_batch(batch!, points, width); batch_size=width)
        key = Philox4x64(42)
        move = MoveMixture((StretchMove(), DEMove()), (1, 1))
        reference = initialize(key, scalar, initial; move, walker_ids=ids)
        kernel = initialize(key, target, points; move, walker_ids=ids, executor=KernelExecutor())
        selected = (Int[], Int[])
        for sweep in 0:5
            external = isodd(sweep)
            if external
                sweep_key = Threefry4x64(90 + sweep)
                index = 3sweep
                selection = subrng(sweep_key, 2, index)
                step!(reference, sweep_key; proposal_index=index)
                step!(kernel, sweep_key; proposal_index=index)
            else
                selection = subrng(subrng(key, 0, sweep), 1, 1)
                step!(reference)
                step!(kernel)
            end
            # The two weights are equal. Draw at the selection address independently.
            expected = rand_at(selection, Float64, 1) < 0.5 ? 1 : 2
            push!(selected[1 + external], expected)
            actual, serial = current_state(kernel), current_state(reference)
            @test actual.move_index == serial.move_index == expected
            @test Array(actual.accepted) == serial.accepted
            @test reduce(hcat, Array.(actual.positions)) ≈ reduce(hcat, serial.positions)
        end
        @test all(indices -> sort(unique(indices)) == [1, 2], selected)
    end
end
