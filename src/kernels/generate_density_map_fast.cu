#include <cstdio>
#include "common/kernels.hpp"

namespace cubu::kernels {
__global__ void
generateDensityMapFast(float* densityOutput,
                       cudaTextureObject_t pointsTex,
                       cudaTextureObject_t edgeIndicesTex,
                       cudaTextureObject_t edgeLengthsTex,
                       size_t pointCount,
                       size_t edgeCount,
                       size_t pitch)
{
  // *** Do nothing if no output is specified
  if (!densityOutput) {
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

    // *** Keep track of a counter for the current point index
    int pointIndex = pointIndexStart;

    // *** Get the length of the current edge
    auto edgeLength = tex1Dfetch<float>(edgeLengthsTex, static_cast<int>(i));

    // *** Keep looping until a marker is found
    while (true) {
      auto point = tex1Dfetch<float2>(pointsTex, pointIndex++);

      // *** Check if the fetched point is the end of the line
      if (pointIndex > pointIndexEnd) {
        break;
      }

      // *** Calculate the index for the density image
      int siteIndex = static_cast<int>(point.y) * static_cast<int>(pitch) +
                      static_cast<int>(point.x);

      // *** Add the edge length to the density input
      densityOutput[siteIndex] += edgeLength;
    }
  }
}
} // namespace cubu::kernels