@setup_workload begin
    rng = Philox4x((42, 1))
    ensembles = (randn(rng, Float64, 3, 12), randn(rng, Float32, 3, 12))
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
            snapshot(state)
        end
        step!(initialize(Threefry4x((42, 1, 2, 3)), logdensity, first(ensembles);
            move=DEMove()), 2)
    end
end
