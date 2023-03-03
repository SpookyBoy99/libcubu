#include <iostream>
#include "bundling_job.hpp"
#include "common/kernels.hpp"
#include "common/macros.hpp"

namespace cubu {
bundling_job::bundling_job(const graph& graph,
                           const bundling_job::settings_t& settings)
{
  // *** Upload the graph to the gpu
  auto [pointsRes, edgeIndicesRes, edgeLengthsRes] = upload(graph, settings);

  // *** Create the random states
  random_states randomStates(512);

  // *** Resample the graph
  std::tie(pointsRes, edgeIndicesRes) =
    resample(pointsRes, edgeIndicesRes, randomStates, settings);

  // *** Generate the density map from the resampled points
  auto densityMapRes =
    generate_density_map(pointsRes, edgeIndicesRes, edgeLengthsRes, settings);

  std::vector<float> d_densityMap;
  densityMapRes.copy_to_host(d_densityMap);

  for (const auto& d : d_densityMap) {
    std::cout << d << std::endl;
  }
}

std::tuple<linear_resource<glm::vec2>,
           linear_resource<int>,
           linear_resource<float>>
bundling_job::upload(const graph& graph, const settings_t& settings)
{
  // *** Calculate the edge and point counts
  size_t pointCount = graph.point_count() + graph.edges().size(),
         edgeCount = graph.edges().size();

  // *** Create a new vector for all the points
  std::vector<glm::vec2> h_points;

  // *** Allocate the memory for all the points and end of line markers
  h_points.reserve(pointCount);

  // *** Create a new vector for all the edge indices
  std::vector<int> h_edgeIndices;

  // *** Allocate the memory for the edge indices
  h_edgeIndices.reserve(edgeCount);

  // *** Create a new vector for all the edge lengths
  std::vector<float> h_edgeLengths;

  // *** Allocate the memory for the edge indices
  h_edgeLengths.reserve(edgeCount);

  // *** Get the range of the graph
  glm::vec2 range = graph.bounds().max - graph.bounds().min;

  // *** Calculate the scale
  float scale = range.x > range.y
                  ? static_cast<float>(settings.resolution.x) / range.x
                  : static_cast<float>(settings.resolution.y) / range.y;

  // *** Calculate the translation
  glm::vec2 translation = {
    (static_cast<float>(settings.resolution.x) - scale * range.x) / 2,
    (static_cast<float>(settings.resolution.y) - scale * range.y) / 2
  };

  // *** Loop over all the poly lines
  for (const auto& line : graph.edges()) {
    // *** Add the starting point of the next polyline to the list of edge
    // indices
    h_edgeIndices.emplace_back(h_points.size());

    // *** Add the length of the edge to the list of edge lengths
    h_edgeLengths.emplace_back(line->length());

    // *** Loop over all the points in the line
    for (const auto& point : line->points()) {
      h_points.emplace_back((point - graph.bounds().min) * scale + translation);
    }

    // *** Add end of polyline marker
    h_points.emplace_back(-1, -1);
  }

  // *** Create a linear texture containing all the points and data
  linear_resource pointsRes(
    h_points, cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat));

  // *** Create a linear texture containing the indices of the starting point of
  // each edge in the points texture
  linear_resource edgeIndicesRes(
    h_edgeIndices,
    cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned));

  // *** Create a linear texture containing the lengths of each edge
  linear_resource edgeLengthsRes(
    h_edgeLengths,
    cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat));

  return { std::move(pointsRes),
           std::move(edgeIndicesRes),
           std::move(edgeLengthsRes) };
}

std::tuple<linear_resource<glm::vec2>, linear_resource<int>>
bundling_job::resample(const linear_resource<glm::vec2>& pointsRes,
                       const linear_resource<int>& edgeIndicesRes,
                       const random_states& randomStates,
                       const settings_t& settings)
{
  // *** Get the edge count from the size of the edge indices resource
  size_t edgeCount = edgeIndicesRes.size();

  // *** Configure the kernel execution parameters
  size_t blockSize = 256;
  size_t numBlocks = (edgeCount + blockSize - 1) / blockSize;

  // *** Allocate the edges sample count output array
  int* d_edgesSampleCount;
  gpuAssert(cudaMalloc((void**)&d_edgesSampleCount,
                       edgeCount * sizeof(d_edgesSampleCount[0])));

  // *** Call the kernel
  kernels::calculateResampleCount<<<numBlocks, blockSize>>>(
    d_edgesSampleCount,
    pointsRes.tex(),
    edgeIndicesRes.tex(),
    edgeCount,
    settings.samplingStep);

  // *** Check kernel launch
  gpuAssert(cudaPeekAtLastError());

  // *** Synchronise the kernel
  gpuAssert(cudaDeviceSynchronize());

  // *** Allocate a vector for the edge sample points
  std::vector<std::decay<decltype(*d_edgesSampleCount)>::type>
    h_edgesSampleCounts(edgeCount);

  // *** Copy the edges to
  gpuAssert(cudaMemcpy(h_edgesSampleCounts.data(),
                       d_edgesSampleCount,
                       edgeCount * sizeof(d_edgesSampleCount[0]),
                       cudaMemcpyDeviceToHost));

  // *** Free the device sample points
  gpuAssert(cudaFree(d_edgesSampleCount));

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
  kernels::resample<<<numBlocks, blockSize>>>(
    reinterpret_cast<float2*>(resampledPointsRes.dev_ptr()),
    pointsRes.tex(),
    edgeIndicesRes.tex(),
    resampledEdgeIndices.tex(),
    edgeCount,
    resampledPointCount,
    settings.samplingStep,
    settings.jitter,
    randomStates.data());

  // *** Check kernel launch
  gpuAssert(cudaPeekAtLastError());

  // *** Synchronise the kernel
  gpuAssert(cudaDeviceSynchronize());

  return { std::move(resampledPointsRes), std::move(resampledEdgeIndices) };
}

resource_2d<float>
bundling_job::generate_density_map(const linear_resource<glm::vec2>& pointsRes,
                                   const linear_resource<int>& edgeIndicesRes,
                                   const linear_resource<float>& edgeLengthsRes,
                                   const settings_t& settings)
{
  // *** Get the edge count from the size of the edge indices resource
  size_t edgeCount = edgeIndicesRes.size();

  // *** Configure the kernel execution parameters
  size_t blockSize = 256;
  size_t numBlocks = (edgeCount + blockSize - 1) / blockSize;

  // *** Create a resource for the density map
  resource_2d<float> densityMapRes(
    settings.resolution.x,
    settings.resolution.y,
    cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat));

  // *** Set to all zeros
  gpuAssert(cudaMemset(densityMapRes.dev_ptr(),
                       0x0,
                       densityMapRes.size() *
                         sizeof(decltype(densityMapRes)::value_type)));

  // *** If fast density is used, no atomic operations are used in the kernel
  // which will result in lower values than using the accurate approach
  if (settings.fastDensity) {
    // *** Call the kernel
    // fixme: Pitched memory access
    kernels::generateDensityMapFast<<<blockSize, numBlocks>>>(
      densityMapRes.dev_ptr(),
      pointsRes.tex(),
      edgeIndicesRes.tex(),
      edgeLengthsRes.tex(),
      edgeCount,
      settings.resolution.x);

    // *** Check kernel launch
    gpuAssert(cudaPeekAtLastError());

    // *** Synchronise the kernel
    gpuAssert(cudaDeviceSynchronize());
  }

  return densityMapRes;
}
} // namespace cubu