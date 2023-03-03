#include "common/kernels.hpp"

namespace cubu::kernels {
__global__ void
advectSites(float2* output,
            cudaTextureObject_t texSites,
            cudaTextureObject_t texDensity,
            size_t pointCount,
            float kernelSize)
{
  // *** Do nothing if no output is specified
  if (!output) {
    return;
  }

  // *** Get the index and stride from gpu
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  // *** Loop over the points
  for (size_t i = index; i < pointCount; i += stride) {
    auto point = tex1Dfetch<float2>(texSites, static_cast<int>(i));

    // *** Markers can just be copied, don't advect it
    if (point.x == -1 && point.y == -1) {
      // *** Add the previous point of the edge to the line
      output[i] = point;
    } else {
      auto v_d = tex2D<float>(texDensity, point.x, point.y - 1),
           v_l = tex2D<float>(texDensity, point.x - 1, point.y),
           v_r = tex2D<float>(texDensity, point.x + 1, point.y),
           v_t = tex2D<float>(texDensity, point.x, point.y + 1);

      auto g = make_float2(v_r - v_l, v_t - v_d);

      const float eps =
        1.0e-4; // Ensures we don't next get div by 0 for 0-length vectors
      float gn = g.x * g.x + g.y * g.y;
      if (gn < eps)
        gn = 0;
      else
        gn = rsqrtf(gn); // Robustly normalize the gradient

      //      float k = kernelSize * point.z * gn;
      //      g.x *= k;
      //      g.x += point.x;
      //      g.y *= k;
      //      g.y += point.y;

      output[i] = g;
    }
  }
}
} // namespace cubu::kernels