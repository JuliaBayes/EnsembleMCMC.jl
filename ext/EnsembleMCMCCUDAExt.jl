module EnsembleMCMCCUDAExt

import CUDA
import EnsembleMCMC

EnsembleMCMC._check_kernel_array(initial::CUDA.AnyCuArray) =
    CUDA.functional() || throw(ArgumentError("CUDA is not functional"))

# `CUDA.device` has methods only for unwrapped arrays.
_cuparent(x::CUDA.CuArray) = x
_cuparent(x) = _cuparent(parent(x))

EnsembleMCMC._with_kernel_device(f, initial::CUDA.AnyCuArray) =
    CUDA.device!(f, CUDA.device(_cuparent(initial)))

end
