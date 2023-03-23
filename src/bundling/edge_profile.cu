#include <cmath>
#include "cubu/bundling.hpp"

namespace cubu {
cpu_gpu_func float
bundling::edge_profile::operator()(size_t i, size_t size) const
{
  switch (profileType_) {
    case profile_type::uniform:
      return fixEndpoints_ && (i == 0 || i == size - 1) ? 0.0f : 1.0f;
    case profile_type::hourglass:
      if (fixEndpoints_ && (i == 0 || i == size - 1)) {
        return 0.0f;
      }

      float x =
        std::abs(static_cast<float>(i) - static_cast<float>(size) / 2.0f) /
        (static_cast<float>(size) / 2.0f);

      return x > 0.7f ? std::pow((1 - x) / 0.3f, 4.0f) : 1.0f;
  }

  return 0;
}

bundling::edge_profile
bundling::edge_profile::uniform(bool fixEndpoints)
{
  return edge_profile(profile_type_t::uniform, fixEndpoints);
}

bundling::edge_profile
bundling::edge_profile::hourglass(bool fixEndpoints)
{
  return edge_profile(profile_type_t::hourglass, fixEndpoints);
}

bundling::edge_profile::edge_profile(profile_type_t profileType,
                                     bool fixEndpoints)
  : profileType_(profileType)
  , fixEndpoints_(fixEndpoints)
{
}
} // namespace cubu