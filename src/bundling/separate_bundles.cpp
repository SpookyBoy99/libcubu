#include <glm/geometric.hpp>
#include <glm/vec3.hpp>
#include "cubu/bundling.hpp"

namespace cubu {
graph
bundling::separate_bundles(const cubu::graph& bundledGraph, float displacement)
{
  // *** Create a vector for all the edges of the interpolated graph
  std::vector<std::unique_ptr<polyline>> edges;

  // *** Reserve enough space for all the edges
  edges.reserve(bundledGraph.edges().size());

  // *** Loop over all the edges in the source and bundled graph
  for (const auto& edge : bundledGraph.edges()) {
    // *** Create a copy of all the points
    std::vector<point_t> points(edge->points());

    // *** Get the previous point
    auto previousPoint = points.front();

    // *** Accumulate distance from starting point to current point
    float distanceToStart = 0;

    // *** Loop over all points in the edge except first and last
    for (size_t i = 1; i < edge->points().size() - 1; i++) {
      // *** Get the next point
      auto currentPoint = edge->at(i);

      // *** Calculate the distance between the previous and current point and
      // add it to the accumulated distance from the start
      distanceToStart += glm::distance(previousPoint, currentPoint);

      // *** Calculate the arc-length of the i-th point of the target edge [0,1]
      const float t = distanceToStart / edge->length();

      // *** Separate bundles going in different directions
      auto tan = glm::normalize(glm::vec3{ currentPoint - previousPoint, 0 });
      auto delta = glm::cross(tan, { 0, 0, 1 });

      // *** Calculate the displacement and add it to the point
      points[i] += glm::vec2(delta.x, delta.y) * displacement *
                   std::pow(1.0f - 2.0f * std::abs(t - 0.5f), 0.4f);

      // *** Update the previous point
      previousPoint = currentPoint;
    }

    // *** Create the new polyline from the points and displacements
    edges.emplace_back(
      std::make_unique<polyline>(std::move(points), edge->displacement()));
  }

  return graph{ std::move(edges) };
}
} // namespace cubu