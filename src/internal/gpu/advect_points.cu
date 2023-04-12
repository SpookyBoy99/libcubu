#include "cubu/internal/gpu.hpp"
#include "cubu/internal/validate.hpp"

namespace cubu::internal {
namespace kernels {
__global__ void
advectPoints(float2* advectedPoints,
             cudaTextureObject_t pointsTex,
             cudaTextureObject_t edgeIndicesTex,
             cudaTextureObject_t densityMapTex,
             size_t pointCount,
             size_t edgeCount,
             bundling::edge_profile edgeProfile,
             float kernelSize)
{
  // *** Do nothing if no advectedPoints is specified
  if (!advectedPoints) {
    return;
  }

  // *** Get the index and stride from gpu
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  // *** Loop over the edges
  for (size_t i = index; i < edgeCount; i += stride) {
    // *** Get the index of the first point of the edge
    int pointIndexStart = tex1Dfetch<int>(edgeIndicesTex, static_cast<int>(i)),
        pointIndexEnd =
          i == edgeCount - 1
            ? static_cast<int>(pointCount)
            : tex1Dfetch<int>(edgeIndicesTex, static_cast<int>(i + 1));

    // *** Keep looping until a marker is found
    for (int j = pointIndexStart; j < pointIndexEnd; j++) {
      auto point = tex1Dfetch<float2>(pointsTex, j);

      auto v_d = tex2D<float>(densityMapTex, point.x, point.y - 1),
           v_l = tex2D<float>(densityMapTex, point.x - 1, point.y),
           v_r = tex2D<float>(densityMapTex, point.x + 1, point.y),
           v_t = tex2D<float>(densityMapTex, point.x, point.y + 1);

      auto g = make_float2(v_r - v_l, v_t - v_d);

      const float eps =
        1.0e-4; // Ensures we don't next get div by 0 for 0-length vectors
      float gn = g.x * g.x + g.y * g.y;
      if (gn < eps)
        gn = 0;
      else
        gn = rsqrtf(gn); // Robustly normalize the gradient

      float k =
        kernelSize *
        edgeProfile(j - pointIndexStart, pointIndexEnd - pointIndexStart) * gn;
      g.x *= k;
      g.x += point.x;
      g.y *= k;
      g.y += point.y;

      advectedPoints[j] = g;
    }
  }
}
} // namespace kernels

linear_resource<glm::vec2>
gpu::advect_points(const linear_resource<glm::vec2>& pointsRes,
                   const linear_resource<int>& edgeIndicesRes,
                   const resource_2d<float>& densityMapRes,
                   const bundling::edge_profile& edgeProfile,
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
  validate cudaPeekAtLastError();

  // *** Synchronise the kernel
  validate cudaDeviceSynchronize();

  return advectedPointsRes;
}
} // namespace cubu::internal