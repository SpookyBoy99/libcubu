#include "cubu/internal/gpu.hpp"
#include "cubu/internal/gpu_check.hpp"
#include "cubu/internal/kernels.hpp"

namespace cubu::internal {
resource_2d<float>
gpu::generate_density_map(const linear_resource<glm::vec2>& pointsRes,
                          const linear_resource<int>& edgeIndicesRes,
                          const linear_resource<float>& edgeLengthsRes,
                          float kernelSize,
                          const glm::ivec2& resolution,
                          bool fastDensity)
{
  // *** Get the edge count from the size of the edge indices resource
  size_t pointCount = pointsRes.size(), edgeCount = edgeIndicesRes.size();

  // *** Create a resource for the density map
  resource_2d<float> densityMapRes(resolution.x, resolution.y);

  // *** Set to all zeros
  gpu_check cudaMemset2D(densityMapRes.dev_ptr(),
                         densityMapRes.pitch() *
                           sizeof(decltype(densityMapRes)::value_type),
                         0x0,
                         densityMapRes.width(),
                         densityMapRes.height());

  {
    // *** If fast density is used, no atomic operations are used in the kernel
    // which will result in lower values than using the accurate approach
    if (fastDensity) {
      // *** Configure the kernel execution parameters
      size_t blockSize = 256;
      size_t numBlocks = (edgeCount + blockSize - 1) / blockSize;

      // *** Call the kernel
      kernels::generateDensityMapFast<<<blockSize, numBlocks>>>(
        densityMapRes.dev_ptr(),
        pointsRes.tex(),
        edgeIndicesRes.tex(),
        edgeLengthsRes.tex(),
        pointCount,
        edgeCount,
        densityMapRes.pitch());

      // *** Check kernel launch
      gpu_check cudaPeekAtLastError();

      // *** Synchronise the kernel
      gpu_check cudaDeviceSynchronize();
    } else {
      // *** Create a texture for keeping track of the density count
      resource_2d<uint> densityCountsRes(resolution.x, resolution.y);

      {
        // *** Configure the kernel execution parameters
        size_t blockSize = 256;
        size_t numBlocks = (edgeCount + blockSize - 1) / blockSize;

        // *** Set to all zeros
        gpu_check cudaMemset2D(densityCountsRes.dev_ptr(),
                               densityCountsRes.pitch() *
                                 sizeof(decltype(densityMapRes)::value_type),
                               0x0,
                               densityCountsRes.width(),
                               densityCountsRes.height());

        kernels::generateDensityMapCount<<<blockSize, numBlocks>>>(
          densityCountsRes.dev_ptr(),
          pointsRes.tex(),
          edgeIndicesRes.tex(),
          pointsRes.size(),
          edgeIndicesRes.size(),
          densityCountsRes.pitch());

        gpu_check cudaDeviceSynchronize();
      }

      {
        // *** Configure the kernel execution parameters
        dim3 blockSize(256, 256);
        dim3 numBlocks((resolution.x + blockSize.x - 1) / blockSize.x,
                       (resolution.y + blockSize.y - 1) / blockSize.y);

        kernels::convertDensityMapToFloat<<<blockSize, numBlocks>>>(
          densityMapRes.dev_ptr(),
          densityCountsRes.tex(),
          resolution.x,
          resolution.y);

        gpu_check cudaDeviceSynchronize();
      }
    }
  }

  {
    // *** Generate the parabolic filter kernel
    const int kernelRadius = static_cast<int>(kernelSize / 2.0f) * 2;
    const int kernelLength = 2 * kernelRadius + 1;

    std::vector<float> h_parabolicFilterKernel;
    h_parabolicFilterKernel.reserve(kernelLength);

    for (size_t i = 0; i < kernelLength; i++) {
      auto x = static_cast<float>(i) / static_cast<float>(kernelLength - 1);
      x = std::abs(x - 0.5f) / 0.5f;
      h_parabolicFilterKernel.emplace_back(1.0f - x * x);
    }

    // *** Copy the kernel to a texture
    linear_resource parabolicFilterKernelRes(h_parabolicFilterKernel);

    // *** Configure the kernel execution parameters
    dim3 blockSize(256, 256);
    dim3 numBlocks((resolution.x + blockSize.x - 1) / blockSize.x,
                   (resolution.y + blockSize.y - 1) / blockSize.y);

    {
      // *** Create a resource for the vertically convoluted density map
      resource_2d<float> convDensityMapRes(resolution.x, resolution.y);

      kernels::convolutionKernel<<<blockSize, numBlocks>>>(
        convDensityMapRes.dev_ptr(),
        densityMapRes.tex(),
        parabolicFilterKernelRes.tex(),
        resolution.x,
        resolution.y,
        convDensityMapRes.pitch(),
        kernelRadius,
        convolveRows);

      // *** Check kernel launch
      gpu_check cudaPeekAtLastError();

      // *** Synchronise the kernel
      gpu_check cudaDeviceSynchronize();

      // *** Move the convoluted density map over to the density map
      densityMapRes = std::move(convDensityMapRes);
    }

    {
      // *** Create a resource for the vertically convoluted density map
      resource_2d<float> convDensityMapRes(resolution.x, resolution.y);

      kernels::convolutionKernel<<<blockSize, numBlocks>>>(
        convDensityMapRes.dev_ptr(),
        densityMapRes.tex(),
        parabolicFilterKernelRes.tex(),
        resolution.x,
        resolution.y,
        convDensityMapRes.pitch(),
        kernelRadius,
        convolveColumns);

      // *** Check kernel launch
      gpu_check cudaPeekAtLastError();

      // *** Synchronise the kernel
      gpu_check cudaDeviceSynchronize();

      // *** Move the convoluted density map over to the density map
      densityMapRes = std::move(convDensityMapRes);
    }
  }

  return densityMapRes;
}
} // namespace cubu::internal