#include "cubu/internal/gpu.hpp"
#include "cubu/internal/validate.hpp"

namespace cubu::internal {
namespace kernels {
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
} // namespace kernels

std::tuple<linear_resource<glm::vec2>, linear_resource<int>>
gpu::resample_edges(const linear_resource<glm::vec2>& pointsRes,
                    const linear_resource<int>& edgeIndicesRes,
                    const random_states& randomStates,
                    float samplingStep,
                    float jitter)
{
  // *** Get the edge count from the size of the edge indices resource
  size_t pointCount = pointsRes.size(), edgeCount = edgeIndicesRes.size();

  // *** Configure the kernel execution parameters
  size_t blockSize = 256;
  size_t numBlocks = (edgeCount + blockSize - 1) / blockSize;

  // *** Allocate the edges sample count output array
  int* d_edgesSampleCount;
  validate cudaMalloc((void**)&d_edgesSampleCount,
                      edgeCount * sizeof(d_edgesSampleCount[0]));

  // *** Call the kernel
  kernels::calculateResampleCount<<<numBlocks, blockSize>>>(
    d_edgesSampleCount,
    pointsRes.tex(),
    edgeIndicesRes.tex(),
    pointCount,
    edgeCount,
    samplingStep);

  // *** Check kernel launch
  validate cudaPeekAtLastError();

  // *** Synchronise the kernel
  validate cudaDeviceSynchronize();

  // *** Allocate a vector for the edge sample points
  std::vector<std::decay<decltype(*d_edgesSampleCount)>::type>
    h_edgesSampleCounts(edgeCount);

  // *** Copy the edges to
  validate cudaMemcpy(h_edgesSampleCounts.data(),
                      d_edgesSampleCount,
                      edgeCount *
                        sizeof(decltype(h_edgesSampleCounts)::value_type),
                      cudaMemcpyDeviceToHost);

  // *** Free the device sample points
  validate cudaFree(d_edgesSampleCount);

  // *** Get the count for the previous edge
  auto previousSampleCount = h_edgesSampleCounts.front();

  // *** Reset the first sample count to zero
  h_edgesSampleCounts.front() = 0;

  // *** Loop over all the edges sample counts
  for (size_t i = 1; i < edgeCount; i++) {
    // *** Get the sample count for the current edge
    auto currentSampleCount = h_edgesSampleCounts[i];

    // *** Update it to be equal to the previous value + previous sample count
    h_edgesSampleCounts[i] = h_edgesSampleCounts[i - 1] + previousSampleCount;

    // *** Set the previous to the current
    previousSampleCount = currentSampleCount;
  }

  // *** Get the total sample count
  size_t resampledPointCount = h_edgesSampleCounts.back() + previousSampleCount;

  // *** Create a new linear resource for the sample points
  linear_resource resampledEdgeIndices(
    h_edgesSampleCounts,
    cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned));

  // *** Create a new linear resource from the resampled edges
  linear_resource<glm::vec2> resampledPointsRes(
    resampledPointCount,
    cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat));

  // *** Call the kernel
  kernels::resampleEdges<<<numBlocks, blockSize>>>(
    reinterpret_cast<float2*>(resampledPointsRes.dev_ptr()),
    pointsRes.tex(),
    edgeIndicesRes.tex(),
    resampledEdgeIndices.tex(),
    pointCount,
    resampledPointCount,
    edgeCount,
    samplingStep,
    jitter,
    randomStates.data());

  // *** Check kernel launch
  validate cudaPeekAtLastError();

  // *** Synchronise the kernel
  validate cudaDeviceSynchronize();

  return { std::move(resampledPointsRes), std::move(resampledEdgeIndices) };
}
} // namespace cubu::internal