#include "cubu/internal/gpu.hpp"
#include "cubu/internal/gpu_check.hpp"
#include "cubu/internal/kernels.hpp"

namespace cubu::internal {
linear_resource<glm::vec2>
gpu::advect_points(const linear_resource<glm::vec2>& pointsRes,
                   const linear_resource<int>& edgeIndicesRes,
                   const resource_2d<float>& densityMapRes,
                   const cubu::types::edge_profile& edgeProfile,
                   float kernelSize)
{
  // *** Get the edge count from the size of the edge indices resource
  size_t pointCount = pointsRes.size(), edgeCount = edgeIndicesRes.size();

  // *** Configure the kernel execution parameters
  size_t blockSize = 256;
  size_t numBlocks = (edgeCount + blockSize - 1) / blockSize;

  // *** Create a new linear resource from the resampled edges
  linear_resource<glm::vec2> advectedPointsRes(
    pointCount,
    cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat));

  // *** Call the kernel
  kernels::advectPoints<<<blockSize, numBlocks>>>(
    reinterpret_cast<float2*>(advectedPointsRes.dev_ptr()),
    pointsRes.tex(),
    edgeIndicesRes.tex(),
    densityMapRes.tex(),
    pointCount,
    edgeCount,
    edgeProfile,
    kernelSize);

  // *** Check kernel launch
  gpu_check cudaPeekAtLastError();

  // *** Synchronise the kernel
  gpu_check cudaDeviceSynchronize();

  return advectedPointsRes;
}
} // namespace cubu::internal