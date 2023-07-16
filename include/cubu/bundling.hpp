#ifndef CUBU_BUNDLING_HPP
#define CUBU_BUNDLING_HPP

#include "graph.hpp"

#ifdef __NVCC__
#define cpu_gpu_func __host__ __device__
#else
#define cpu_gpu_func
#endif

namespace cubu {
class bundling
{
public:
  /**
   * Shape of the bundled graph edges.
   */
  class edge_profile
  {
    /**
     * Supported profile types.
     */
    typedef enum class profile_type
    {
      uniform,
      hourglass
    } profile_type_t;

  public:
    /**
     * Get the interpolated value for the i-th point of the edge.
     *
     * @param i    Index of the point in the edge.
     * @param size Number of points in the edge.
     *
     * @returns Profile value [0,1] based on the position within the edge.
     */
    cpu_gpu_func float operator()(size_t i, size_t size) const;

    /**
     * Creates a new edge profile a uniform shape.
     *
     * @param fixEndpoints Flag indicating whether the endpoints should stay
     *                     pinned.
     *
     * @returns Uniform edge profile.
     */
    static edge_profile uniform(bool fixEndpoints = false);

    /**
     * Creates a new edge profile a hourglass shape.
     *
     * @param fixEndpoints Flag indicating whether the endpoints should stay
     *                     pinned.
     *
     * @returns Hourglass edge profile.
     */
    static edge_profile hourglass(bool fixEndpoints = false);

  private:
    /**
     * Constructs a new edge profile based on the profile type.
     *
     * @param profileType  Shape of the edge profile.
     * @param fixEndpoints Flag indicating whether the endpoints should stay
     *                     pinned.
     */
    explicit edge_profile(profile_type_t profileType, bool fixEndpoints);

  private:
    profile_type_t profileType_;
    bool fixEndpoints_;
  };

public:
  /**
   * Settings for the gpu bundling.
   */
  typedef struct bundling_settings
  {
    /** Resolution to perform the bundling at. */
    int resolution{ 512 };

    /** Kernel size (pixels): controls the spatial 'scale' at which we see
     * bundles. */
    float bundlingKernelSize{ 32.0f };

    /**
     * Advection step size as a fraction of the kernel sizes [0,1]. Controls
     * speed of the bundling.
     */
    float advectionStepFactor{ 0.5f };

    /** Sampling step (pixels) of polylines. */
    float samplingStep{ 15.0f };

    /**
     * Jitter factor ([0,1]): Fraction of 'spl' that sample points are jittered
     * along a sampled edge.
     */
    float jitter{ 0.25f };

    /** 1D function describing bundling strength along an edge. */
    edge_profile edgeProfile{ edge_profile::uniform(true) };

    /** Laplacian smoothing kernel width ([0,1]), fraction of image width. */
    float smoothingKernelFrac{ 0.05f };

    /** Bundle smoothing ([0,1]): Controls smoothness of bundles. */
    float smoothness{ 0.2f };

    /** Number of bundling iterations. */
    size_t bundlingIterations{ 15 };
  } bundling_settings_t;

  /**
   * Settings for interpolating the bundled graph.
   */
  typedef struct interpolation_settings
  {
    bool absoluteDisplacement{ false };

    float displacementMax{ 0.2f };

    float relaxation{ 0 };
  } interpolation_settings_t;

public:
  /**
   * Bundles the graph on the gpu using cuda.
   *
   * @param graph    Graph to bundle.
   * @param settings Bundling settings.
   *
   * @returns Bundled graph.
   */
  static graph bundle(const graph& graph, const bundling_settings_t& settings);

  /**
   * Interpolates the bundled graph using the original graph.
   *
   * @param originalGraph Graph before bundling.
   * @param bundledGraph  Bundled graph.
   * @param settings      Interpolation settings.
   *
   * @returns Interpolated version of the bundled graph.
   */
  static graph interpolate(const graph& originalGraph,
                           const graph& bundledGraph,
                           interpolation_settings_t settings);

  /**
   * Separates bundles by displacing points from their original location based
   * on trails direction.
   *
   * @param bundledGraph Graph to separate the bundles of.
   * @param displacement Separation distance for the bundles.
   *
   * @returns Graph with displaced (separated) bundles.
   */
  static graph separate_bundles(const cubu::graph& bundledGraph, float displacement);

  // todo: Implement method for generating the density and shading maps of the
  //   interpolated graph.
  //   static < #? ? ? # > generate_maps(const graph& interpolatedGraph);

  // todo: Requires site locations:
  //    - Generate and store on just gpu, or do gpu cpu copy?
  //    - Pro copy: Easier interface, greater consistency
  //    - Con copy: Copy is not required, texture cache cannot be reused
  //    : Possiblity -> Separate type that has the site locations just on the
  //                    gpu, copies are not allowed however (managing gpu data).

  // todo: Specific density map type?
  static std::vector<float> generate_density_map(const graph& graph);

  // todo: Specific shading map type?
  static std::vector<float> generate_shading_map(const graph& graph);

  /**
   * Static class, delete constructor.
   */
  bundling() = delete;
};
} // namespace cubu

#endif // CUBU_BUNDLING_HPP
