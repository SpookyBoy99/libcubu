#ifndef CUBU_EDGE_PROFILE_HPP
#define CUBU_EDGE_PROFILE_HPP

#ifndef __NVCC__
#define __host__
#define __device__
#endif

#include <array>
#include <cstddef>

namespace cubu {
namespace types {
class edge_profile
{
  typedef enum class profile_type
  {
    uniform,
    hourglass
  } profile_type_t;

public:
  __host__ __device__ float operator()(size_t i, size_t size) const;

  static edge_profile uniform(bool fixEndpoints = false);

  static edge_profile hourglass(bool fixEndpoints = false);

private:
  explicit edge_profile(profile_type_t profileType, bool fixEndpoints);

private:
  profile_type_t profileType_;
  bool fixEndpoints_;
};
}

typedef types::edge_profile edge_profile_t;
} // namespace cubu

#endif // CUBU_EDGE_PROFILE_HPP
