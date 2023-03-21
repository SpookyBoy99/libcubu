#ifndef CUBU_BUNDLER_HPP
#define CUBU_BUNDLER_HPP

#include "types/edge_profile.hpp"
#include "types/graph.hpp"

namespace cubu {
class bundler
{
public:
  typedef struct settings
  {
    /** Resolution to perform the bundling at. */
    glm::ivec2 resolution{ 512 };

    /** Kernel size (pixels): controls the spatial 'scale' at which we see
     * bundles. */
    float bundlingKernelSize{ 32.0f };

    /** Sampling step (pixels) of polylines. */
    float samplingStep{ 15.0f };

    /**
     * Jitter factor ([0,1]): Fraction of 'spl' that sample points are jittered
     * along a sampled edge.
     */
    float jitter{ 0.25f };

    /**
     * Flag indicating whether to use  polyline-style bundling or classical
     * smooth bundling.
     */
    bool polylineStyle{ false };

    /**
     * Flag indicating whether the fast density approximation must be used or
     * exact density maps.
     */
    bool fastDensity{ true };

    /** 1D function describing bundling strength along an edge. */
    edge_profile_t edgeProfile{ types::edge_profile::uniform() };

    /** Laplacian smoothing kernel width ([0,1]), fraction of image width. */
    float smoothingKernelFrac{ 0.05f };

    /** Bundle smoothing ([0,1]): Controls smoothness of bundles. */
    float smoothness{ 0.2f };

    /** Number of bundling iterations. */
    size_t bundlingIterations{ 15 };
  } settings_t;

  bundler() = delete;

  static graph_t bundle(const graph_t& graph, const settings_t& settings);
};

} // namespace cubu

#endif // CUBU_BUNDLER_HPP
