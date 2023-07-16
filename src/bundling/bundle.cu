#include <fstream>
#include <iostream>
#include "cubu/bundling.hpp"
#include "cubu/internal/gpu.hpp"
#include "cubu/internal/random_states.hpp"

namespace cubu {
graph
bundling::bundle(const graph& graph, const bundling_settings_t& settings)
{
  // *** Get the offset
  glm::vec2 offset = graph.bounds().min;

  // *** Get the range of the graph
  glm::vec2 range = graph.bounds().max - graph.bounds().min;

  // *** Calculate the scale
  float scale = range.x > range.y
                  ? static_cast<float>(settings.resolution) / range.x
                  : static_cast<float>(settings.resolution) / range.y;

  // *** Calculate the translation
  glm::vec2 translation = {
    (static_cast<float>(settings.resolution) - scale * range.x) / 2,
    (static_cast<float>(settings.resolution) - scale * range.y) / 2
  };

  // *** Upload the graph to the gpu
  auto [pointsRes, edgeIndicesRes, edgeLengthsRes] =
    internal::gpu::upload_graph(graph, offset, translation, scale);

  // *** Create the random states
  internal::random_states randomStates(512);

  // *** Calculate the factor with which to decrease the bundling kernel size
  // every loop
  float kernelSizeScalingFactor =
    std::pow(2.0f / settings.bundlingKernelSize,
             1.0f / static_cast<float>(settings.bundlingIterations));

  // *** Copy the kernel size from the setting as initial value
  float kernelSize = settings.bundlingKernelSize;

  // *** Iteratively bundle the edges
  for (size_t i = 0; i < settings.bundlingIterations; i++) {
    // *** Resample the graph
    std::tie(pointsRes, edgeIndicesRes) =
      internal::gpu::resample_edges(pointsRes,
                                    edgeIndicesRes,
                                    randomStates,
                                    settings.samplingStep,
                                    settings.jitter);

    // *** Generate the density map from the resampled points
    auto densityMapRes =
      internal::gpu::generate_density_map(pointsRes,
                                          edgeIndicesRes,
                                          edgeLengthsRes,
                                          kernelSize,
                                          settings.resolution);

    // *** Advect the points
    pointsRes =
      internal::gpu::advect_points(pointsRes,
                                   edgeIndicesRes,
                                   densityMapRes,
                                   settings.edgeProfile,
                                   kernelSize * settings.advectionStepFactor);

    // *** Perform smoothing
    pointsRes = internal::gpu::smooth_edges(pointsRes,
                                            edgeIndicesRes,
                                            settings.edgeProfile,
                                            settings.smoothingKernelFrac,
                                            settings.samplingStep,
                                            settings.smoothness,
                                            settings.resolution);

    // *** Decrease kernel size (coarse-to-fine bundling)
    kernelSize *= kernelSizeScalingFactor;
  }

  return internal::gpu::download_graph(
    pointsRes, edgeIndicesRes, offset, translation, scale);
}
} // namespace cubu