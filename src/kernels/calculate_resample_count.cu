#include "common/kernels.hpp"

namespace cubu::kernels {
__global__ void
calculateResampleCount(int* edgesSampleCount,
                       cudaTextureObject_t pointsTex,
                       cudaTextureObject_t edgeIndicesTex,
                       size_t pointCount,
                       size_t edgeCount,
                       float delta)
{
  // *** Do nothing if no output is specified
  if (!edgesSampleCount) {
    return;
  }

  // *** Get the index and stride from gpu
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  // *** Loop over the required edges
  for (size_t i = index; i < edgeCount; i += stride) {
    // *** Reset the current distance to the maximum delta distance
    float currentDistance = delta;

    // *** Get the index of the first point of the edge
    int pointIndexStart = tex1Dfetch<int>(edgeIndicesTex, static_cast<int>(i)),
        pointIndexEnd =
          i == edgeCount - 1
            ? static_cast<int>(pointCount)
            : tex1Dfetch<int>(edgeIndicesTex, static_cast<int>(i + 1));

    // *** Keep track of a counter for the current point index
    int pointIndex = pointIndexStart;

    // *** Get the first two points
    auto previousPoint = tex1Dfetch<float2>(pointsTex, pointIndex++),
         currentPoint = tex1Dfetch<float2>(pointsTex, pointIndex++);

    // *** Initialize the point count to 1 for the start point
    int sampleCount = 1;

    // *** Keep looping until the end of the edge is reached
    while (true) {
      // *** Calculate the distance between the current and (updated) previous
      // point
      float newDistance = sqrtf((previousPoint.x - currentPoint.x) *
                                  (previousPoint.x - currentPoint.x) +
                                (previousPoint.y - currentPoint.y) *
                                  (previousPoint.y - currentPoint.y));

      // *** Check if the calculated distance is smaller than the current
      // distance, in that case skip the current point, if further away add a
      // new point
      if (newDistance < currentDistance) {
        // *** Decrease the current distance with the distance between the
        // previous and current point
        currentDistance -= newDistance;

        // *** Set previous point to be the current point
        previousPoint = currentPoint;

        // *** Fetch the next point in line and set it to be the current point
        currentPoint = tex1Dfetch<float2>(pointsTex, pointIndex++);

        // *** Check if the fetched point is the end of the line
        if (pointIndex > pointIndexEnd) {
          break;
        }
      } else {
        // *** Calculate the factor of the current and calculated distances
        float t = currentDistance / newDistance;

        // *** Add a virtual point between the previous point and current point
        previousPoint.x = previousPoint.x * (1 - t) + currentPoint.x * t;
        previousPoint.y = previousPoint.y * (1 - t) + currentPoint.y * t;

        // *** Increase the number of sample points required
        sampleCount++;

        // *** Reset the current distance
        currentDistance = delta;
      }
    }

    // *** Add an extra sample point if the current distance is smaller than
    // delta
    if (currentDistance < delta) {
      sampleCount++;
    }

    // *** Store the sample points
    edgesSampleCount[i] = sampleCount == 1 ? 2 : sampleCount;
  }
}
} // namespace cubu::kernels