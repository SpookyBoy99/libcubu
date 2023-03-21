#include "cubu/internal/gpu_check.hpp"
#include "cubu/internal/kernels.hpp"
#include "cubu/internal/random_states.hpp"

namespace cubu::internal {
random_states::random_states(size_t size)
  : d_randomStates_(nullptr)
  , size_(size)
{
  // *** Initialize the random states
  gpu_check cudaMalloc((void**)&d_randomStates_,
                       size_ * sizeof(d_randomStates_[0]));

  // *** Call the kernel
  kernels::initRandomStates<<<1, size_>>>(d_randomStates_, size_);

  // *** Check kernel launch
  gpu_check cudaPeekAtLastError();

  // *** Synchronise the kernel
  gpu_check cudaDeviceSynchronize();
}

random_states::~random_states()
{
  gpu_check cudaFree(d_randomStates_);
}

curandState*
random_states::data() const
{
  return d_randomStates_;
}

size_t
random_states::size() const
{
  return size_;
}
} // namespace cubu::internal