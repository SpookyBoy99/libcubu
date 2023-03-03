#ifndef CUBU_EDGE_PROFILE_HPP
#define CUBU_EDGE_PROFILE_HPP

#include <array>
#include <cstddef>

namespace cubu {
class edge_profile
{
  typedef enum class profile_type
  {
    uniform,
    hourglass
  } profile_type_t;

public:
  float operator()(size_t i, size_t size) const;

  static edge_profile uniform(bool fixEndpoints = false);

  static edge_profile hourglass(bool fixEndpoints = false);

private:
  explicit edge_profile(profile_type_t profileType, bool fixEndpoints);

private:
  profile_type_t profileType_;
  bool fixEndpoints_;
};
} // namespace cubu

#endif // CUBU_EDGE_PROFILE_HPP
