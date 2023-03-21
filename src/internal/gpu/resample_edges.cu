#include "cubu/internal/gpu.hpp"
#include "cubu/internal/gpu_check.hpp"
#include "cubu/internal/kernels.hpp"

namespace cubu::internal {
std::tuple<linear_resource<glm::vec2>, linear_resource<int>>
gpu::resample_edges(const linear_resource<glm::vec2>& pointsRes,
                    const linear_resource<int>& edgeIndicesRes,
                    const random_states& randomStates,
                    float samplingStep,
                    float jitter)
{
  // *** Get the edge count from the size of the edge indices resource
  size_t pointCount = pointsRes.size(), edgeCount = edgeIndicesRes.size();

  // *** Configure the kernel execution parameters
  size_t blockSize = 256;
  size_t numBlocks = (edgeCount + blockSize - 1) / blockSize;

  // *** Allocate the edges sample count output array
  int* d_edgesSampleCount;
  gpu_check cudaMalloc((void**)&d_edgesSampleCount,
                       edgeCount * sizeof(d_edgesSampleCount[0]));

  // *** Call the kernel
  kernels::calculateResampleCount<<<numBlocks, blockSize>>>(
    d_edgesSampleCount,
    pointsRes.tex(),
    edgeIndicesRes.tex(),
    pointCount,
    edgeCount,
    samplingStep);

  // *** Check kernel launch
  gpu_check cudaPeekAtLastError();

  // *** Synchronise the kernel
  gpu_check cudaDeviceSynchronize();

  // *** Allocate a vector for the edge sample points
  std::vector<std::decay<decltype(*d_edgesSampleCount)>::type>
    h_edgesSampleCounts(edgeCount);

  // *** Copy the edges to
  gpu_check cudaMemcpy(h_edgesSampleCounts.data(),
                       d_edgesSampleCount,
                       edgeCount *
                         sizeof(decltype(h_edgesSampleCounts)::value_type),
                       cudaMemcpyDeviceToHost);

  // *** Free the device sample points
  gpu_check cudaFree(d_edgesSampleCount);

  // *** Get the count for the previous edge
  auto previousSampleCount = h_edgesSampleCounts.front();

  // *** Reset the first sample count to zero
  h_edgesSampleCounts.front() = 0;

  // *** Loop over all the edges sample counts
  for (size_t i = 1; i < edgeCount; i++) {
    // *** Get the sample count for the current edge
    auto currentSampleCount = h_edgesSampleCounts[i];

    // *** Update it to be equal to the previous value + previous sample count
    h_edgesSampleCounts[i] = h_edgesSampleCounts[i - 1] + previousSampleCount;

    // *** Set the previous to the current
    previousSampleCount = currentSampleCount;
  }

  // *** Get the total sample count
  size_t resampledPointCount = h_edgesSampleCounts.back() + previousSampleCount;

  // *** Create a new linear resource for the sample points
  linear_resource resampledEdgeIndices(
    h_edgesSampleCounts,
    cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned));

  // *** Create a new linear resource from the resampled edges
  linear_resource<glm::vec2> resampledPointsRes(
    resampledPointCount,
    cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat));

  // *** Call the kernel
  kernels::resampleEdges<<<numBlocks, blockSize>>>(
    reinterpret_cast<float2*>(resampledPointsRes.dev_ptr()),
    pointsRes.tex(),
    edgeIndicesRes.tex(),
    resampledEdgeIndices.tex(),
    pointCount,
    resampledPointCount,
    edgeCount,
    samplingStep,
    jitter,
    randomStates.data());

  // *** Check kernel launch
  gpu_check cudaPeekAtLastError();

  // *** Synchronise the kernel
  gpu_check cudaDeviceSynchronize();

  return { std::move(resampledPointsRes), std::move(resampledEdgeIndices) };
}

} // namespace cubu::internal