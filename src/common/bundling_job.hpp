#ifndef CUBU_BUNDLING_JOB_HPP
#define CUBU_BUNDLING_JOB_HPP

#include <glm/vec4.hpp>
#include "core/edge_profile.hpp"
#include "core/graph.hpp"
#include "core/linear_resource.hpp"
#include "core/random_states.hpp"
#include "core/resource_2d.hpp"

namespace cubu {
class bundling_job
{
public:
  typedef struct settings
  {
    /** Resolution to perform the bundling at. */
    glm::ivec2 resolution{ 512 };

    /** Sampling step (pixels) of polylines. */
    float samplingStep{ 15.0f };

    /**
     * Jitter factor ([0,1]): Fraction of 'spl' that sample points are jittered
     * along a sampled edge.
     */
    float jitter{ 0.25f };

    /**
     * Flag indicating whether the fast density approximation must be used or
     * exact density maps.
     */
    bool fastDensity{ true };

    /** 1D function describing bundling strength along an edge. */
    edge_profile edgeProfile{ edge_profile::uniform() };
  } settings_t;

  bundling_job(const graph& graph, const settings_t& settings);

protected:
  static std::tuple<linear_resource<glm::vec2>,
                    linear_resource<int>,
                    linear_resource<float>>
  upload(const graph& graph, const settings_t& settings);

  static std::tuple<linear_resource<glm::vec2>, linear_resource<int>> resample(
    const linear_resource<glm::vec2>& pointsRes,
    const linear_resource<int>& edgeIndicesRes,
    const random_states& randomStates,
    const settings_t& settings);

  static resource_2d<float> generate_density_map(
    const linear_resource<glm::vec2>& pointsRes,
    const linear_resource<int>& edgeIndicesRes,
    const linear_resource<float>& edgeLengthsRes,
    const settings_t& settings);
};

} // namespace cubu

#endif // CUBU_BUNDLING_JOB_HPP
