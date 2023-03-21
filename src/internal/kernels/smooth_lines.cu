#include <cassert>
#include "cubu/internal/kernels.hpp"

namespace cubu::internal::kernels {
__global__ void
smoothEdges(float2* smoothedPoints,
            cudaTextureObject_t pointsTex,
            cudaTextureObject_t edgeIndicesTex,
            size_t pointCount,
            size_t edgeCount,
            edge_profile_t edgeProfile,
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
} // namespace cubu::internal::kernels