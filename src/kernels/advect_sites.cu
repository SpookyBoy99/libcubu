#include <cstdio>
#include "common/kernels.hpp"

namespace cubu::kernels {
__global__ void
advectSites(float2* advectedPoints,
            cudaTextureObject_t pointsTex,
            cudaTextureObject_t edgeIndicesTex,
            cudaTextureObject_t densityMapTex,
            size_t pointCount,
            size_t edgeCount,
            edge_profile edgeProfile,
            float kernelSize)
{
  // *** Do nothing if no advectedPoints is specified
  if (!advectedPoints) {
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
      auto point = tex1Dfetch<float2>(pointsTex, j);

      auto v_d = tex2D<float>(densityMapTex, point.x, point.y - 1),
           v_l = tex2D<float>(densityMapTex, point.x - 1, point.y),
           v_r = tex2D<float>(densityMapTex, point.x + 1, point.y),
           v_t = tex2D<float>(densityMapTex, point.x, point.y + 1);

      auto g = make_float2(v_r - v_l, v_t - v_d);

      const float eps =
        1.0e-4; // Ensures we don't next get div by 0 for 0-length vectors
      float gn = g.x * g.x + g.y * g.y;
      if (gn < eps)
        gn = 0;
      else
        gn = rsqrtf(gn); // Robustly normalize the gradient

      float k =
        kernelSize *
        edgeProfile(j - pointIndexStart, pointIndexEnd - pointIndexStart) * gn;
      g.x *= k;
      g.x += point.x;
      g.y *= k;
      g.y += point.y;

      advectedPoints[j] = g;
    }
  }
}
} // namespace cubu::kernels