#include "cubu/internal/gpu.hpp"

namespace cubu::internal {
graph_t
gpu::download_graph(const linear_resource<glm::vec2>& pointsRes,
                    const linear_resource<int>& edgeIndicesRes)
{
  // *** Copy the points and the edges over to the host
  std::vector<glm::vec2> h_points;
  std::vector<int> h_edgeIndices;

  pointsRes.copy_to_host(h_points);
  edgeIndicesRes.copy_to_host(h_edgeIndices);

  // *** Create a set for the new lines
  std::vector<std::unique_ptr<polyline_t>> lines;

  // *** Loop over all the edges
  for (size_t i = 0; i < h_edgeIndices.size(); i++) {
    // *** Calculate the start and end points
    int pointIndexStart = h_edgeIndices[i],
        pointIndexEnd = i == h_edgeIndices.size() - 1
                          ? static_cast<int>(h_points.size())
                          : h_edgeIndices[i + 1];

    lines.emplace_back(std::make_unique<polyline_t>(std::vector<point_t>{
      h_points.begin() + pointIndexStart, h_points.begin() + pointIndexEnd }));
  }

  // *** Create a new graph from lines
  return graph_t(std::move(lines));
}
} // namespace cubu::internal