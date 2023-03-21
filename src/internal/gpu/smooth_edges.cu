#include "cubu/internal/gpu.hpp"
#include "cubu/internal/gpu_check.hpp"
#include "cubu/internal/kernels.hpp"

namespace cubu::internal {
linear_resource<glm::vec2>
gpu::smooth_edges(const linear_resource<glm::vec2>& pointsRes,
                  const linear_resource<int>& edgeIndicesRes,
                  const edge_profile_t& edgeProfile,
                  float smoothingKernelFrac,
                  float samplingStep,
                  float smoothness,
                  const glm::uvec2& resolution)
{

  // *** Get the edge count from the size of the edge indices resource
  size_t pointCount = pointsRes.size(), edgeCount = edgeIndicesRes.size();

  // *** Configure the kernel execution parameters
  size_t blockSize = 256;
  size_t numBlocks = (edgeCount + blockSize - 1) / blockSize;

  // *** Create a new linear resource from the resampled edges
  linear_resource<glm::vec2> smoothedPointsRes(
    pointCount,
    cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat));

  // *** Compute 1D Laplacian filter size, in #points, which corresponds to
  // 'filter_kernel' space units
  int L = static_cast<int>(
    smoothingKernelFrac *
    static_cast<float>(std::min(resolution.x, resolution.y)) / samplingStep);

  // *** Prevent doing unnecessary work
  if (L == 0) {
    // *** Simply copy the points res over to the smoothed points res if the
    // laplacian filter is zero
    gpu_check cudaMemcpy(
      smoothedPointsRes.dev_ptr(),
      pointsRes.dev_ptr(),
      pointsRes.size() * sizeof(std::decay_t<decltype(pointsRes)>::value_type),
      cudaMemcpyDeviceToDevice);

    return smoothedPointsRes;
  }

  // *** Call the kernel
  kernels::smoothEdges<<<blockSize, numBlocks>>>(
    reinterpret_cast<float2*>(smoothedPointsRes.dev_ptr()),
    pointsRes.tex(),
    edgeIndicesRes.tex(),
    pointCount,
    edgeCount,
    edgeProfile,
    L,
    smoothness);

  // *** Check kernel launch
  gpu_check cudaPeekAtLastError();

  // *** Synchronise the kernel
  gpu_check cudaDeviceSynchronize();

  return smoothedPointsRes;
}
} // namespace cubu::internal