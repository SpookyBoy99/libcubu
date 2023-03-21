#ifndef CUBU_INTERPOLATOR_HPP
#define CUBU_INTERPOLATOR_HPP

#include "types/graph.hpp"

namespace cubu {
class interpolator
{
public:
  typedef struct settings
  {
    bool absoluteDisplacement;

    float displacementMax;

    float directionalSeparation;

    float relaxation;

    glm::ivec2 resolution;
  } settings_t;

  static graph_t interpolate(const graph_t& originalGraph,
                           const graph_t& bundledGraph,
                           settings_t settings);
};
} // namespace cubu

#endif // CUBU_INTERPOLATOR_HPP
