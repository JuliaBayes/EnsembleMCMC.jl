@setup_workload begin
    rng = PureRNGs.Philox4x64(42)
    initial_rng = subrng(rng, 1)
    ensembles = (first(PureRNGs.randn_next(initial_rng, Float64, 3, 12)),
        first(PureRNGs.randn_next(subrng(initial_rng, 1), Float32, 3, 12)))
    logdensity = x -> -sum(abs2, x) / 2
    batched = BatchedLogDensity(logdensity,
        (values, positions) -> map!(logdensity, values, eachcol(positions)))

    @compile_workload begin
        for initial in ensembles,
            move in (StretchMove(), DEMove(), DESnookerMove(), GaussianReplacementMove()),
            (target, executor) in ((logdensity, SerialExecutor()), (batched, KernelExecutor()))
            state = initialize(rng, target, initial; move, executor)
            step!(state, 2)
            sample!(state, 2)
            current_state(state)
            saved = snapshot(state)
            synchronize!(state, saved.positions, saved.logdensities)
            step!(state, rng; proposal_index=2)
        end
        step!(initialize(PureRNGs.Threefry4x64(42), logdensity, first(ensembles);
            move=DEMove()), 2)
        mixture = MoveMixture((StretchMove(), DEMove()), [0.5, 0.5])
        for initial in ensembles
            sample!(initialize(rng, logdensity, initial; move=mixture, executor=ThreadedExecutor()), 2)
        end
    end
end
