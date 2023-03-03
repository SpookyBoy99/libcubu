#include "common/kernels.hpp"

namespace cubu::kernels {
__global__ void
generateDensityMapFast(float* densityOutput,
                       cudaTextureObject_t pointsTex,
                       cudaTextureObject_t edgeIndicesTex,
                       cudaTextureObject_t edgeLengthsTex,
                       size_t edgeCount,
                       int width)
{
  // *** Do nothing if no output is specified
  if (!densityOutput) {
    return;
  }

  // *** Get the index and stride from gpu
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  // *** Loop over the points
  for (size_t i = index; i < edgeCount; i += stride) {
    // *** Get the index of the first point of the edge
    int pointIndex = tex1Dfetch<int>(edgeIndicesTex, static_cast<int>(i));

    // *** Get the length of the current edge
    auto edgeLength = tex1Dfetch<float>(edgeLengthsTex, static_cast<int>(i));

    // *** Keep looping until a marker is found
    while (true) {
      auto point = tex1Dfetch<float2>(pointsTex, pointIndex++);

      // *** Stop on a marker
      if (point.x == -1 && point.y == -1) {
        break;
      }

      // *** Calculate the index for the density image
      int siteIndex =
        static_cast<int>(point.y) * width + static_cast<int>(point.x);

      // *** Add the edge length to the density input
      densityOutput[siteIndex] += edgeLength;
    }
  }
}
} // namespace cubu::kernels