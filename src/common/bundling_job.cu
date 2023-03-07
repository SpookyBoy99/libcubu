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

  // todo: wrap in loop for iterative resampling and advection
  {
    // *** Resample the graph
    std::tie(pointsRes, edgeIndicesRes) =
      resample(pointsRes, edgeIndicesRes, randomStates, settings);

    // *** Generate the density map from the resampled points
    auto densityMapRes =
      generate_density_map(pointsRes, edgeIndicesRes, edgeLengthsRes, settings);

    std::vector<float> densityMapValues;
    densityMapRes.copy_to_host(densityMapValues);

    // *** Advect the points
    pointsRes =
      advect_sites(pointsRes, edgeIndicesRes, densityMapRes, settings);

    // *** Perform smoothing
    pointsRes = smooth_lines(pointsRes, edgeIndicesRes, settings);

    std::vector<glm::vec2> smoothedPoints;
    pointsRes.copy_to_host(smoothedPoints);

    for (const auto& p : smoothedPoints) {
      printf("%f %f\n", p.x, p.y);
    }
  }
}

std::tuple<linear_resource<glm::vec2>,
           linear_resource<int>,
           linear_resource<float>>
bundling_job::upload(const graph& graph, const settings_t& settings)
{
  // *** Calculate the edge and point counts
  size_t pointCount = graph.point_count(), edgeCount = graph.edges().size();

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
  size_t pointCount = pointsRes.size(), edgeCount = edgeIndicesRes.size();

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
    pointCount,
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
  gpuAssert(
    cudaMemcpy(h_edgesSampleCounts.data(),
               d_edgesSampleCount,
               edgeCount * sizeof(decltype(h_edgesSampleCounts)::value_type),
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
    pointCount,
    resampledPointCount,
    edgeCount,
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
  size_t pointCount = pointsRes.size(), edgeCount = edgeIndicesRes.size();

  // *** Create a resource for the density map
  resource_2d<float> densityMapRes(settings.resolution.x,
                                   settings.resolution.y);

  // *** Set to all zeros
  gpuAssert(cudaMemset2D(densityMapRes.dev_ptr(),
                         densityMapRes.pitch() *
                           sizeof(decltype(densityMapRes)::value_type),
                         0x0,
                         densityMapRes.width(),
                         densityMapRes.height()));

  {
    // *** Configure the kernel execution parameters
    size_t blockSize = 256;
    size_t numBlocks = (edgeCount + blockSize - 1) / blockSize;

    // *** If fast density is used, no atomic operations are used in the kernel
    // which will result in lower values than using the accurate approach
    if (settings.fastDensity) {
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
      gpuAssert(cudaPeekAtLastError());

      // *** Synchronise the kernel
      gpuAssert(cudaDeviceSynchronize());
    } else {
      // todo: Add accurate density map generation
    }
  }

  {
    // *** Generate the parabolic filter kernel
    const int kernelRadius = static_cast<int>(settings.samplingStep / 2.0f) * 2;
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
    dim3 numBlocks((settings.resolution.x + blockSize.x - 1) / blockSize.x,
                   (settings.resolution.y + blockSize.y - 1) / blockSize.y);

    {
      // *** Create a resource for the vertically convoluted density map
      resource_2d<float> convDensityMapRes(settings.resolution.x,
                                           settings.resolution.y);

      kernels::convolutionKernel<<<blockSize, numBlocks>>>(
        convDensityMapRes.dev_ptr(),
        densityMapRes.tex(),
        parabolicFilterKernelRes.tex(),
        settings.resolution.x,
        settings.resolution.y,
        kernelRadius,
        convolveRows);

      // *** Check kernel launch
      gpuAssert(cudaPeekAtLastError());

      // *** Synchronise the kernel
      gpuAssert(cudaDeviceSynchronize());

      // *** Move the convoluted density map over to the density map
      densityMapRes = std::move(convDensityMapRes);
    }

    {
      // *** Create a resource for the vertically convoluted density map
      resource_2d<float> convDensityMapRes(settings.resolution.x,
                                           settings.resolution.y);

      kernels::convolutionKernel<<<blockSize, numBlocks>>>(
        convDensityMapRes.dev_ptr(),
        densityMapRes.tex(),
        parabolicFilterKernelRes.tex(),
        settings.resolution.x,
        settings.resolution.y,
        kernelRadius,
        convolveColumns);

      // *** Check kernel launch
      gpuAssert(cudaPeekAtLastError());

      // *** Synchronise the kernel
      gpuAssert(cudaDeviceSynchronize());

      // *** Move the convoluted density map over to the density map
      densityMapRes = std::move(convDensityMapRes);
    }
  }

  return densityMapRes;
}

linear_resource<glm::vec2>
bundling_job::advect_sites(const linear_resource<glm::vec2>& pointsRes,
                           const linear_resource<int>& edgeIndicesRes,
                           const resource_2d<float>& densityMapRes,
                           const bundling_job::settings_t& settings)
{
  // *** Get the edge count from the size of the edge indices resource
  size_t pointCount = pointsRes.size(), edgeCount = edgeIndicesRes.size();

  // *** Configure the kernel execution parameters
  size_t blockSize = 256;
  size_t numBlocks = (edgeCount + blockSize - 1) / blockSize;

  // *** Create a new linear resource from the resampled edges
  linear_resource<glm::vec2> advectedPointsRes(
    pointCount,
    cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat));

  // todo: Remove; this is just a demonstration for the barriers which will be
  //  deprecated soon
  cudaMemset(advectedPointsRes.dev_ptr(),
             0x0,
             advectedPointsRes.size() * sizeof(float2));
  // todo: Add support for directional advection

  // *** Call the kernel
  kernels::advectSites<<<blockSize, numBlocks>>>(
    reinterpret_cast<float2*>(advectedPointsRes.dev_ptr()),
    pointsRes.tex(),
    edgeIndicesRes.tex(),
    densityMapRes.tex(),
    pointCount,
    edgeCount,
    settings.edgeProfile,
    settings.advectKernelSize);

  // *** Check kernel launch
  gpuAssert(cudaPeekAtLastError());

  // *** Synchronise the kernel
  gpuAssert(cudaDeviceSynchronize());

  return advectedPointsRes;
}

linear_resource<glm::vec2>
bundling_job::smooth_lines(const linear_resource<glm::vec2>& pointsRes,
                           const linear_resource<int>& edgeIndicesRes,
                           const bundling_job::settings_t& settings)
{

  // *** Get the edge count from the size of the edge indices resource
  size_t pointCount = pointsRes.size(), edgeCount = edgeIndicesRes.size();

  // *** Configure the kernel execution parameters
  size_t blockSize = 256;
  size_t numBlocks = (edgeCount + blockSize - 1) / blockSize;

  // *** Create a new linear resource from the resampled edges
  linear_resource<glm::vec2> smoothedPointsRes(
    pointCount,
    cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat));

  // todo: Remove; this is just a demonstration for the barriers which will be
  //  deprecated soon
  cudaMemset(smoothedPointsRes.dev_ptr(),
             0x0,
             smoothedPointsRes.size() * sizeof(float2));

  // *** Compute 1D Laplacian filter size, in #points, which corresponds to
  // 'filter_kernel' space units
  int L = static_cast<int>(settings.smoothingKernelFrac *
                           static_cast<float>(settings.resolution.x) /
                           settings.samplingStep);

  // *** Prevent doing unnecessary work
  if (L == 0) {
    // *** Simply copy the points res over to the smoothed points res if the
    // laplacian filter is zero
    gpuAssert(cudaMemcpy(
      smoothedPointsRes.dev_ptr(),
      pointsRes.dev_ptr(),
      pointsRes.size() * sizeof(std::decay_t<decltype(pointsRes)>::value_type),
      cudaMemcpyDeviceToDevice));

    return smoothedPointsRes;
  }

  // *** Call the kernel
  kernels::smoothLines<<<blockSize, numBlocks>>>(
    reinterpret_cast<float2*>(smoothedPointsRes.dev_ptr()),
    pointsRes.tex(),
    edgeIndicesRes.tex(),
    pointCount,
    edgeCount,
    settings.edgeProfile,
    L,
    settings.smoothness);

  // *** Check kernel launch
  gpuAssert(cudaPeekAtLastError());

  // *** Synchronise the kernel
  gpuAssert(cudaDeviceSynchronize());

  return smoothedPointsRes;
}
} // namespace cubu