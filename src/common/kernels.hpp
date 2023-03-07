#ifndef CUBU_KERNELS_HPP
#define CUBU_KERNELS_HPP

#ifdef __NVCC__
#include <cstdio>
#include <curand_kernel.h>
#include "core/edge_profile.hpp"

namespace cubu {
typedef enum convolution_direction
  : char
{
  convolveColumns = 0,
  convolveRows
} convolution_direction_t;

namespace kernels {
__global__ void
initRandomStates(curandState* states, size_t size);

__global__ void
generateDensityMapFast(float* densityOutput,
                       cudaTextureObject_t pointsTex,
                       cudaTextureObject_t edgeIndicesTex,
                       cudaTextureObject_t edgeLengthsTex,
                       size_t pointCount,
                       size_t edgeCount,
                       size_t pitch);

__global__ void
calculateResampleCount(int* edgesSampleCount,
                       cudaTextureObject_t pointsTex,
                       cudaTextureObject_t edgeIndicesTex,
                       size_t pointCount,
                       size_t edgeCount,
                       float delta);

__global__ void
resample(float2* resampledPoints,
         cudaTextureObject_t pointsTex,
         cudaTextureObject_t edgeIndicesTex,
         cudaTextureObject_t resampledEdgeIndicesTex,
         size_t pointCount,
         size_t resampledPointCount,
         size_t edgeCount,
         float delta,
         float jitter,
         curandState* state);

__global__ void
advectSites(float2* advectedPoints,
            cudaTextureObject_t pointsTex,
            cudaTextureObject_t edgeIndicesTex,
            cudaTextureObject_t densityMapTex,
            size_t pointCount,
            size_t edgeCount,
            edge_profile edgeProfile,
            float kernelSize);

__global__ void
smoothLines(float2* smoothedPoints,
            cudaTextureObject_t pointsTex,
            cudaTextureObject_t edgeIndicesTex,
            size_t pointCount,
            size_t edgeCount,
            edge_profile edgeProfile,
            int laplacianFilterSize,
            float smoothness);

__global__ void
convolutionKernel(float* output,
                  cudaTextureObject_t densityTex,
                  cudaTextureObject_t convolutionKernelTex,
                  int width,
                  int height,
                  int kernelRadius,
                  convolution_direction_t direction);
} // namespace kernels
} // namespace cubu
#endif // __NVCC__

#endif // CUBU_KERNELS_HPP
