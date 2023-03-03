#ifndef CUBU_KERNELS_HPP
#define CUBU_KERNELS_HPP

#ifdef __NVCC__
#include <curand_kernel.h>

namespace cubu::kernels {
__global__ void
initRandomStates(curandState* states, size_t size);

__global__ void
generateDensityMapFast(float* densityOutput,
                       cudaTextureObject_t pointsTex,
                       cudaTextureObject_t edgeIndicesTex,
                       cudaTextureObject_t edgeLengthsTex,
                       size_t edgeCount,
                       int width);

__global__ void
calculateResampleCount(int* edgesSampleCount,
                       cudaTextureObject_t pointsTex,
                       cudaTextureObject_t edgeIndicesTex,
                       size_t edgeCount,
                       float delta);

__global__ void
resample(float2* resampledPoints,
         cudaTextureObject_t pointsTex,
         cudaTextureObject_t edgeIndicesTex,
         cudaTextureObject_t resampledEdgeIndicesTex,
         size_t edgeCount,
         size_t resampledEdgeCount,
         float delta,
         float jitter,
         curandState* state);

__global__ void
advectSites(float2* output,
            cudaTextureObject_t texSites,
            cudaTextureObject_t texDensity,
            cudaTextureObject_t texEdgeLengths,
            size_t pointCount,
            float kernelSize);
} // namespace cubu::kernels
#endif // __NVCC__

#endif // CUBU_KERNELS_HPP
