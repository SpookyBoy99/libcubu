#ifndef CUBU_RANDOM_STATES_HPP
#define CUBU_RANDOM_STATES_HPP

#ifdef __NVCC__
#include "curand_kernel.h"

namespace cubu::internal {
class random_states
{
public:
  /**
   * Initializes a curandState array on the device with the given size.
   *
   * @param size Number of states to generate.
   */
  explicit random_states(size_t size = 512);

  /**
   * Frees the curandState array on the device.
   */
  ~random_states();

  /**
   * Getter for the curandState array.
   *
   * @returns Device pointer to the curandState.
   */
  [[nodiscard]] curandState* data() const;

  /**
   * Getter for the size of the curandState array.
   *
   * @returns Number of random states.
   */
  [[nodiscard]] size_t size() const;

private:
  curandState* d_randomStates_;
  size_t size_;
};
} // namespace cubu::internal
#else
namespace cubu::internal {
class random_states;
} // namespace cubu::internal
#endif
#endif // CUBU_RANDOM_STATES_HPP
