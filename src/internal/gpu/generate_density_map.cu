#include "cubu/internal/gpu.hpp"
#include "cubu/internal/gpu_check.hpp"

namespace cubu::internal {
typedef enum convolution_direction
  : char
{
  convolveColumns = 0,
  convolveRows
} convolution_direction_t;

namespace kernels {
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
      densityOutput[siteIndex] += edgeLength;
    }
  }
}

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

__global__ void
convolveDensityMap(float* output,
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
} // namespace kernels

resource_2d<float>
gpu::generate_density_map(const linear_resource<glm::vec2>& pointsRes,
                          const linear_resource<int>& edgeIndicesRes,
                          const linear_resource<float>& edgeLengthsRes,
                          float kernelSize,
                          int resolution,
                          bool fastDensity)
{
  // *** Get the edge count from the size of the edge indices resource
  size_t pointCount = pointsRes.size(), edgeCount = edgeIndicesRes.size();

  // *** Create a resource for the density map
  resource_2d<float> densityMapRes(resolution, resolution);

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
      resource_2d<uint> densityCountsRes(resolution, resolution);

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
        dim3 numBlocks((resolution + blockSize.x - 1) / blockSize.x,
                       (resolution + blockSize.y - 1) / blockSize.y);

        kernels::convertDensityMapToFloat<<<blockSize, numBlocks>>>(
          densityMapRes.dev_ptr(),
          densityCountsRes.tex(),
          resolution,
          resolution);

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
    dim3 numBlocks((resolution + blockSize.x - 1) / blockSize.x,
                   (resolution + blockSize.y - 1) / blockSize.y);

    {
      // *** Create a resource for the vertically convoluted density map
      resource_2d<float> convDensityMapRes(resolution, resolution);

      kernels::convolveDensityMap<<<blockSize, numBlocks>>>(
        convDensityMapRes.dev_ptr(),
        densityMapRes.tex(),
        parabolicFilterKernelRes.tex(),
        resolution,
        resolution,
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
      resource_2d<float> convDensityMapRes(resolution, resolution);

      kernels::convolveDensityMap<<<blockSize, numBlocks>>>(
        convDensityMapRes.dev_ptr(),
        densityMapRes.tex(),
        parabolicFilterKernelRes.tex(),
        resolution,
        resolution,
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