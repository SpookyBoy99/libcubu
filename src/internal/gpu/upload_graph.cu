#include "cubu/internal/gpu.hpp"

namespace cubu::internal {
std::tuple<linear_resource<glm::vec2>,
           linear_resource<int>,
           linear_resource<float>>
gpu::upload_graph(const graph_t& graph, const glm::uvec2& resolution)
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
  float scale = range.x > range.y ? static_cast<float>(resolution.x) / range.x
                                  : static_cast<float>(resolution.y) / range.y;

  // *** Calculate the translation
  glm::vec2 translation = {
    (static_cast<float>(resolution.x) - scale * range.x) / 2,
    (static_cast<float>(resolution.y) - scale * range.y) / 2
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
} // namespace cubu::internal