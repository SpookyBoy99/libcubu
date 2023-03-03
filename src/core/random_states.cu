#include "common/kernels.hpp"
#include "common/macros.hpp"
#include "random_states.hpp"

namespace cubu {
random_states::random_states(size_t size)
  : d_randomStates_(nullptr)
  , size_(size)
{
  // *** Initialize the random states
  gpuAssert(
    cudaMalloc((void**)&d_randomStates_, size_ * sizeof(d_randomStates_[0])));

  // *** Call the kernel
  kernels::initRandomStates<<<1, size_>>>(d_randomStates_, size_);

  // *** Check kernel launch
  gpuAssert(cudaPeekAtLastError());

  // *** Synchronise the kernel
  gpuAssert(cudaDeviceSynchronize());
}

random_states::~random_states()
{
  gpuAssert(cudaFree(d_randomStates_));
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
} // namespace cubu