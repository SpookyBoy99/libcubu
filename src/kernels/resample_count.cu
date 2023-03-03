#include <cstdio>
#include <glm/geometric.hpp>
#include <glm/vec2.hpp>
#include "common/kernels.hpp"

namespace cubu::kernels {
__global__ void
calculateResampleCount(int* edgesSampleCount,
                       cudaTextureObject_t pointsTex,
                       cudaTextureObject_t edgeIndicesTex,
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
    int pointIndex = tex1Dfetch<int>(edgeIndicesTex, static_cast<int>(i));

    // *** Get the first two points
    auto previousPoint = tex1Dfetch<float2>(pointsTex, pointIndex++),
         currentPoint = tex1Dfetch<float2>(pointsTex, pointIndex++);

    // *** Initialize the point count to 2: start point and endpoint marker
    int sampleCount = 2;

    // *** Keep looping until the end of the edge is reached
    while (true) {
      // *** Calculate the distance between the current and (updated) previous
      // point
      float newDistance =
        glm::distance(glm::vec2{ previousPoint.x, previousPoint.y },
                      glm::vec2{ currentPoint.x, currentPoint.y });

      // *** Check if the calculated distance is smaller than the current
      // distance, in that case skip the current point, if further away add a
      // new point
      if (newDistance < currentDistance) {
        // *** Decrease the current distance with the distance between the
        // previous and current point
        currentDistance -= newDistance;

        // *** Set previous point to be the current point
        previousPoint = currentPoint;

        // *** Fetch the next point in line and set it o be the current point
        currentPoint = tex1Dfetch<float2>(pointsTex, pointIndex++);

        // *** Check if the fetched point is the end of the line
        if (currentPoint.x == -1 && currentPoint.y == -1) {
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
    edgesSampleCount[i] = sampleCount == 2 ? 3 : sampleCount;
  }
}
} // namespace cubu::kernels