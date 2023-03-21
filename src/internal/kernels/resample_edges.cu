#include "cubu/internal/kernels.hpp"

namespace cubu::internal::kernels {
__global__ void
resampleEdges(float2* resampledPoints,
              cudaTextureObject_t pointsTex,
              cudaTextureObject_t edgeIndicesTex,
              cudaTextureObject_t resampledEdgeIndicesTex,
              size_t pointCount,
              size_t resampledPointCount,
              size_t edgeCount,
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

    // *** Get the number of new points
    uint resampledIndexStart =
           tex1Dfetch<int>(resampledEdgeIndicesTex, static_cast<int>(i)),
         resampledIndexEnd = i == edgeCount - 1
                               ? resampledPointCount
                               : tex1Dfetch<int>(resampledEdgeIndicesTex,
                                                 static_cast<int>(i + 1));

    // *** Set j to start at the point index start
    uint j = resampledIndexStart;

    // *** Add the 1 point to the resampled line
    resampledPoints[j++] = previousPoint;

    // *** Cache random generator for speed, since we'll modify it locally
    curandState lState = state[threadIdx.x];

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

        // *** Fetch the next point in line and set it o be the current point
        currentPoint = tex1Dfetch<float2>(pointsTex, pointIndex++);

        // *** Check if the fetched point is the end of the line
        if (pointIndex > pointIndexEnd) {
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

        // *** Stop if the number of required resampled points is reached
        if (j == resampledIndexEnd - 1) {
          break;
        }

        // *** Set non-jittered point as new previous point
        previousPoint.x = previousPoint.x * (1 - t) + currentPoint.x * t;
        previousPoint.y = previousPoint.y * (1 - t) + currentPoint.y * t;

        // *** Reset the current distance
        currentDistance = delta;
      }
    }

    // *** Add the last point as well
    resampledPoints[j++] = tex1Dfetch<float2>(pointsTex, pointIndexEnd - 1);

    // *** Update random generator
    state[threadIdx.x] = lState;
  }
}
} // namespace cubu::internal::kernels