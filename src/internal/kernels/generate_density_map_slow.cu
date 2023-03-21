#include "cubu/internal/kernels.hpp"

namespace cubu::internal::kernels {
__global__ void
generateDensityMapCount(uint* countsOutput,
                        cudaTextureObject_t pointsTex,
                        cudaTextureObject_t edgeIndicesTex,
                        size_t pointCount,
                        size_t edgeCount,
                        size_t pitch)
{
  // *** Do nothing if no countsOutput is specified
  if (!countsOutput) {
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

    // *** Keep looping until the last point is reached
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
      atomicAdd(&countsOutput[siteIndex], 1);
    }
  }
}

__global__ void
convertDensityMapToFloat(float* densityOutput,
                         cudaTextureObject_t densityCountsTex,
                         int width,
                         int height)
{
  // *** Do nothing if no countsOutput is specified
  if (!densityOutput) {
    return;
  }

  // *** Get the index and stride from gpu
  size_t index_x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride_x = blockDim.x * gridDim.x;
  size_t index_y = blockIdx.y * blockDim.y + threadIdx.y;
  size_t stride_y = blockDim.y * gridDim.y;

  for (size_t x = index_x; x < width; x += stride_x) {
    for (size_t y = index_y; y < height; y += stride_y) {
      densityOutput[y * width + x] =
        static_cast<float>(tex1Dfetch<uint>(densityCountsTex, y * width + x));
    }
  }
}
} // namespace cubu::internal::kernels