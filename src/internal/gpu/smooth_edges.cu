#include "cubu/internal/gpu.hpp"
#include "cubu/internal/gpu_check.hpp"

namespace cubu::internal {
namespace kernels {
__global__ void
smoothEdges(float2* smoothedPoints,
            cudaTextureObject_t pointsTex,
            cudaTextureObject_t edgeIndicesTex,
            size_t pointCount,
            size_t edgeCount,
            bundling::edge_profile edgeProfile,
            int laplacianFilterSize,
            float smoothness)
{
  // *** Do nothing if no advectedPoints is specified
  if (!smoothedPoints) {
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
      // *** Get the current point and create a new point to accumulate the
      // sampled points
      auto currentPoint = tex1Dfetch<float2>(pointsTex, j),
           sampledPoints = make_float2(0, 0);

      // *** Calculate the start and end points
      int samplePointIndexStart = max(j - laplacianFilterSize, pointIndexStart),
          samplePointIndexEnd = min(j + laplacianFilterSize, pointIndexEnd);

      // *** Accumulate the points
      for (int k = samplePointIndexStart; k < samplePointIndexEnd; k++) {
        auto point = tex1Dfetch<float2>(pointsTex, k);

        sampledPoints.x += point.x;
        sampledPoints.y += point.y;
      }

      // *** Calculate the smoothness according to the edge profile
      const float t =
        smoothness *
        edgeProfile(pointIndexEnd - j - 1, pointIndexEnd - pointIndexStart - 1);

      // *** Linearly interpolate
      const float k =
        t / static_cast<float>(samplePointIndexEnd - samplePointIndexStart);

      // *** Calculate the new point
      currentPoint.x *= 1 - t;
      currentPoint.x += sampledPoints.x * k;
      currentPoint.y *= 1 - t;
      currentPoint.y += sampledPoints.y * k;

      // *** Save the point
      smoothedPoints[j] = currentPoint;
    }
  }
}
} // namespace kernels

linear_resource<glm::vec2>
gpu::smooth_edges(const linear_resource<glm::vec2>& pointsRes,
                  const linear_resource<int>& edgeIndicesRes,
                  const bundling::edge_profile& edgeProfile,
                  float smoothingKernelFrac,
                  float samplingStep,
                  float smoothness,
                  int resolution)
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
  int L = static_cast<int>(smoothingKernelFrac *
                           static_cast<float>(resolution) / samplingStep);

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