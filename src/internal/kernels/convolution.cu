#include "cubu/internal/kernels.hpp"

namespace cubu::internal::kernels {
__global__ void
convolutionKernel(float* output,
                  cudaTextureObject_t densityTex,
                  cudaTextureObject_t convolutionKernelTex,
                  int width,
                  int height,
                  size_t pitch,
                  int kernelRadius,
                  convolution_direction_t direction)
{
  size_t index_x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride_x = blockDim.x * gridDim.x;
  size_t index_y = blockIdx.y * blockDim.y + threadIdx.y;
  size_t stride_y = blockDim.y * gridDim.y;

  for (size_t x = index_x; x < width; x += stride_x) {
    for (size_t y = index_y; y < height; y += stride_y) {
      const float pixel_x = static_cast<float>(x) + 0.5f;
      const float pixel_y = static_cast<float>(y) + 0.5f;

      float sum = 0;

      for (int k = -kernelRadius; k <= kernelRadius; ++k) {
        sum +=
          tex2D<float>(
            densityTex,
            pixel_x + static_cast<float>((direction == convolveRows) * k),
            pixel_y + static_cast<float>((direction == convolveColumns) * k)) *
          tex1Dfetch<float>(convolutionKernelTex, kernelRadius - k);
      }

      output[y * pitch + x] = sum;
    }
  }
}
} // namespace cubu::internal::kernels