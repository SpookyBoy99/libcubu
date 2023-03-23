#ifndef CUBU_GPU_HPP
#define CUBU_GPU_HPP

#include <glm/vec2.hpp>
#include <tuple>
#include "cubu/bundling.hpp"
#include "cubu/graph.hpp"
#include "cubu/internal/linear_resource.hpp"
#include "cubu/internal/random_states.hpp"
#include "cubu/internal/resource_2d.hpp"

namespace cubu::internal {
struct gpu
{
  /**
   * Transforms the graph into a format usable for the gpu and uploads it.
   *
   * @param graph       Graph to upload to the gpu.
   * @param offset      Offset of the non-normalized graph.
   * @param translation Translation that needs to be applied before uploading.
   * @param scale       Scaling that needs to be applied before uploading.
   *
   * @returns Tuple of linear gpu resources containing the points, indices of
   *          each line and lengths of each line.
   */
  static std::tuple<linear_resource<glm::vec2>,
                    linear_resource<int>,
                    linear_resource<float>>
  upload_graph(const graph& graph,
               const glm::vec2& offset = {},
               const glm::vec2& translation = {},
               float scale = 1.0f);

  /**
   * Downloads the point and edge data from the gpu and transforms it into a
   * graph object.
   *
   * @param pointsRes      Linear resource containing the point data.
   * @param edgeIndicesRes Linear resource containing the indices of the
   *                       starting points of each edge.
   *
   * @returns Graph object.
   */
  static graph download_graph(const linear_resource<glm::vec2>& pointsRes,
                              const linear_resource<int>& edgeIndicesRes,
                              const glm::vec2& offset = {},
                              const glm::vec2& translation = {},
                              float scale = 1.0f);

  /**
   * Resamples an existing graph.
   *
   * @param pointsRes      Linear resource containing the point data.
   * @param edgeIndicesRes Linear resource containing the indices of the
   *                       starting points of each edge.
   * @param randomStates   Cuda random states.
   * @param samplingStep   Distance between the resampled points.
   * @param jitter         Random jitter for the sampled points.
   *
   * @returns Tuple of linear gpu resources containing the resampled points and
   *          the new indices of the edge starting points.
   */
  static std::tuple<linear_resource<glm::vec2>, linear_resource<int>>
  resample_edges(const linear_resource<glm::vec2>& pointsRes,
                 const linear_resource<int>& edgeIndicesRes,
                 const random_states& randomStates,
                 float samplingStep,
                 float jitter);

  /**
   * Generates a density map image from the point data.
   *
   * @param pointsRes      Linear resource containing the point data.
   * @param edgeIndicesRes Linear resource containing the indices of the
   *                       starting points of each edge.
   * @param edgeLengthsRes Linear resource containing the length of each edge.
   * @param kernelSize     Size of the filter kernel to generate the density map
   *                       with.
   * @param resolution     Resolution of the density map (must match the
   *                       resolution of graph).
   * @param fastDensity    Use fast density approximation instead of accurate
   *                       density.
   *
   * @returns 2D gpu image of the graph density.
   */
  static resource_2d<float> generate_density_map(
    const linear_resource<glm::vec2>& pointsRes,
    const linear_resource<int>& edgeIndicesRes,
    const linear_resource<float>& edgeLengthsRes,
    float kernelSize,
    int resolution,
    bool fastDensity);

  /**
   * Advects points using the density map information.
   *
   * @param pointsRes      Linear resource containing the point data.
   * @param edgeIndicesRes Linear resource containing the indices of the
   *                       starting points of each edge.
   * @param densityMapRes  2D image of the density map.
   * @param edgeProfile    Edge shape.
   * @param kernelSize     Size of the filter kernel to advect the points with.
   *
   * @returns Linear gpu resource of the advected points.
   */
  static linear_resource<glm::vec2> advect_points(
    const linear_resource<glm::vec2>& pointsRes,
    const linear_resource<int>& edgeIndicesRes,
    const resource_2d<float>& densityMapRes,
    const bundling::edge_profile& edgeProfile,
    float kernelSize);

  /**
   * Smooths the (advected) points into proper lines.
   *
   * @param pointsRes           Linear resource containing the point data.
   * @param edgeIndicesRes      Linear resource containing the indices of the
   *                            starting points of each edge.
   * @param edgeProfile         Edge shape.
   * @param smoothingKernelFrac Percentage of smoothing [0, 1].
   * @param samplingStep        Distance between the smoothed edge points.
   * @param smoothness          Final smoothness of the edges.
   * @param resolution          Resolution of the smoothed edges (must match the
   *                            resolution of the original graph).
   *
   * @returns Linear gpu resource of the smoothed points.
   */
  static linear_resource<glm::vec2> smooth_edges(
    const linear_resource<glm::vec2>& pointsRes,
    const linear_resource<int>& edgeIndicesRes,
    const bundling::edge_profile& edgeProfile,
    float smoothingKernelFrac,
    float samplingStep,
    float smoothness,
    int resolution);

  /**
   * Static class, delete constructor.
   */
  gpu() = delete;
};
} // namespace cubu::internal

#endif // CUBU_GPU_HPP
