#include "cubu/internal/kernels.hpp"

namespace cubu::internal::kernels {
__global__ void
initRandomStates(curandState* states, size_t size)
{
  // *** Get the index and stride from gpu
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  // *** Init the array of random states
  for (size_t i = index; i < size; i += stride) {
    curand_init(1234, i, 0, &states[i]);
  }
}
} // namespace cubu::internal::kernels