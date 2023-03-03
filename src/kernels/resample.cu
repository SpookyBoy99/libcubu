#include <cstdio>
#include <glm/geometric.hpp>
#include <glm/vec2.hpp>
#include "common/kernels.hpp"

namespace cubu::kernels {
__global__ void
resample(float2* resampledPoints,
         cudaTextureObject_t pointsTex,
         cudaTextureObject_t edgeIndicesTex,
         cudaTextureObject_t resampledEdgeIndicesTex,
         size_t edgeCount,
         size_t resampledEdgeCount,
         float delta,
         float jitter,
         curandState* state)
{
  // *** Do nothing if no output is specified
  if (!resampledPoints) {
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

    // *** Get the number of new points
    uint resampledIndexStart =
      tex1Dfetch<int>(resampledEdgeIndicesTex, static_cast<int>(i));

    // *** Get the new number of points for the next line
    uint resampledIndexEnd =
      i == edgeCount - 1
        ? resampledEdgeCount
        : tex1Dfetch<int>(resampledEdgeIndicesTex, static_cast<int>(i + 1));

    // *** Set j to start at the point index start
    uint j = resampledIndexStart;

    // *** Add the 1 point to the resampled line
    resampledPoints[j++] = previousPoint;

    // *** Cache random generator for speed, since we'll modify it locally
    curandState lState = state[threadIdx.x];

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
          // *** Add the previous point of the edge to the line
          resampledPoints[j++] = previousPoint;

          break;
        }
      } else {
        // *** Calculate the factor of the current and calculated distances
        float t = currentDistance / newDistance;

        // *** Generate a random number in range [-1, 1]
        float r = curand_uniform(&lState) * 2 - 1;

        // *** Jitter currently-generated point
        float rt = t * (1 + r * jitter);

        // *** Set the potion based on the previous and next points and the
        //     jitter and add the resampled point to the output
        resampledPoints[j++] =
          make_float2(previousPoint.x * (1 - rt) + currentPoint.x * rt,
                      previousPoint.y * (1 - rt) + currentPoint.y * rt);

        // *** Check if the number of resampled points has been reached
        if (j == resampledIndexEnd) {
          printf("Reached %s %d: Why?", __FILE__, __LINE__);

          // *** Add the last point of the edge to the line
          resampledPoints[j++] = currentPoint;

          break;
        }

        // *** Set non-jittered point as new previous point
        previousPoint.x = previousPoint.x * (1 - t) + currentPoint.x * t;
        previousPoint.y = previousPoint.y * (1 - t) + currentPoint.y * t;

        // *** Reset the current distance
        currentDistance = delta;
      }
    }

    // fixme: When does this occur?
    if (j < resampledIndexEnd - 1) {
      printf("Reached %s %d: Why?", __FILE__, __LINE__);
      // *** Copy the previous point
      resampledPoints[j++] = resampledPoints[j - 2];
    }

    // *** Add the marker
    resampledPoints[j++] = make_float2(-1.0f, -1.0f);

    // *** Check if the correct number of points are generated
    assert(j == resampledIndexEnd);

    // *** Update random generator
    state[threadIdx.x] = lState;
  }
}
} // namespace cubu::kernels