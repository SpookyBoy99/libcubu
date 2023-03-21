#include "cubu/bundler.hpp"
#include "cubu/internal/gpu.hpp"
#include "cubu/internal/random_states.hpp"

namespace cubu {
graph_t
bundler::bundle(const graph_t& graph, const bundler::settings_t& settings)
{
  // *** Upload the graph to the gpu
  auto [pointsRes, edgeIndicesRes, edgeLengthsRes] =
    internal::gpu::upload_graph(graph, settings.resolution);

  // *** Create the random states
  internal::random_states randomStates(512);

  // *** Calculate the factor with which to decrease the bundling kernel size
  // every loop
  float kernelSizeScalingFactor =
    pow(2.0f / settings.bundlingKernelSize,
        1.0f / static_cast<float>(settings.bundlingIterations));

  // *** Copy the kernel size from the setting as initial value
  float kernelSize = settings.bundlingKernelSize;

  // *** Iteratively bundle the edges
  for (size_t i = 0; i < settings.bundlingIterations; i++) {
    // *** Resample only the first step or when not bundling poly-lines
    if (i == 0 || !settings.polylineStyle) {
      // *** Resample the graph
      std::tie(pointsRes, edgeIndicesRes) =
        internal::gpu::resample_edges(pointsRes,
                                      edgeIndicesRes,
                                      randomStates,
                                      settings.samplingStep,
                                      settings.jitter);
    }

    // *** Generate the density map from the resampled points
    auto densityMapRes =
      internal::gpu::generate_density_map(pointsRes,
                                          edgeIndicesRes,
                                          edgeLengthsRes,
                                          kernelSize,
                                          settings.resolution,
                                          settings.fastDensity);

    std::vector<float> densityMapValues;
    densityMapRes.copy_to_host(densityMapValues);

    // *** Advect the points
    pointsRes = internal::gpu::advect_points(pointsRes,
                                             edgeIndicesRes,
                                             densityMapRes,
                                             settings.edgeProfile,
                                             kernelSize);

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

  return internal::gpu::download_graph(pointsRes, edgeIndicesRes);
}
} // namespace cubu