module EnsembleMCMCCUDAExt

import CUDA
import EnsembleMCMC

EnsembleMCMC._check_kernel_array(initial::CUDA.AnyCuArray) =
    CUDA.functional() || throw(ArgumentError("CUDA is not functional"))

# CUDA uploads require an Array rather than a host view. Keep device copies on-device.
EnsembleMCMC._kernel_copyto!(dest::CUDA.CuArray, src) =
    copyto!(dest, convert(Array{eltype(dest)}, src))
EnsembleMCMC._kernel_copyto!(dest::CUDA.CuArray, src::CUDA.AnyCuArray) = copyto!(dest, src)

# `CUDA.device` has methods only for unwrapped arrays.
_cuparent(x::CUDA.CuArray) = x
_cuparent(x) = _cuparent(parent(x))

EnsembleMCMC._with_kernel_device(f, initial::CUDA.AnyCuArray) =
    CUDA.device!(f, CUDA.device(_cuparent(initial)))

end
