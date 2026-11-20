using Pkg

Pkg.activate(; temp=true)
Pkg.add([
    PackageSpec(name="KernelAbstractions", version=v"0.9.45"),
    PackageSpec(name="PrecompileTools", version=v"1.0.1"),
    PackageSpec(name="PureRNGs", version=v"0.0.2"),
    PackageSpec(name="StaticArrays", version=v"1.1.0"),
])
Pkg.develop(; path=dirname(@__DIR__), preserve=Pkg.PRESERVE_ALL)

using EnsembleMCMC, PureRNGs, Test
using EnsembleMCMC: sample

@testset "CPU kernel at minimum dependency versions" begin
    initial = [1.0 -1.0 -1.0 1.0 2.0 -2.0; 1.0 -1.0 1.0 -1.0 2.0 -2.0]
    density(x) = -sum(abs2, x) / 2
    batch!(values, positions) = map!(density, values, eachcol(positions))
    key = Philox4x64(731)
    actual = sample(key, BatchedLogDensity(density, batch!), initial, 1; executor=KernelExecutor())
    @test actual == sample(key, density, initial, 1)
end
