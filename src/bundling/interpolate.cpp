#include <glm/geometric.hpp>
#include "cubu/bundling.hpp"

namespace cubu {
graph
bundling::interpolate(const graph& originalGraph,
                      const graph& bundledGraph,
                      interpolation_settings_t settings)
{
  // *** Create a vector for all the edges of the interpolated graph
  std::vector<polyline> edges;

  // *** Allocate the memory beforehand
  edges.reserve(bundledGraph.size());

  // *** Loop over all the edges in the source and bundled graph
  for (size_t i = 0; i < originalGraph.size(); i++) {
    // *** Get the edge from both the original and bundeld graph
    const auto& originalEdge = originalGraph[i];
    const auto& bundledEdge = bundledGraph[i];

    // *** Calculate the max displacement
    const float maxDisplacement =
      settings.absoluteDisplacement
        ? settings.displacementMax
        : settings.displacementMax * originalEdge.length();

    // *** Create a vector for all the points of the edge and the displacements
    std::vector<point_t> points(bundledEdge.size());
    std::vector<float> displacements(bundledEdge.size());

    // *** Copy the first and last points from the original target edge as they
    // are not interpolated
    std::tie(points.front(), points.back()) = bundledEdge.endpoints();

    // *** Get the previous point
    auto previousPoint = points.front();

    // *** Accumulate distance from starting point to current point
    float distanceToStart = 0;

    // *** Loop over all points in the bundled edge except first and last
    for (size_t j = 1; j < bundledEdge.size() - 1; j++) {
      // *** Get the next point
      const auto& currentPoint = bundledEdge.point_at(j);

      // *** Calculate the distance between the previous and current point and
      // add it to the accumulated distance from the start
      distanceToStart += glm::distance(previousPoint, currentPoint);

      // *** Calculate the arc-length of the i-th point of the target edge [0,1]
      const float t = distanceToStart / bundledEdge.length();

      // *** Calculate the corresponding index for the source (resampled target
      // line has more points)
      auto k =
        static_cast<size_t>(t * static_cast<float>(originalEdge.size() - 1));

      // *** Calculate how far between the current and next point from the
      // source it is (assuming linear uniform sampling of the target)
      const float u =
        t * static_cast<float>(originalEdge.size() - 1) - static_cast<float>(k);

      // *** Calculate the interpolated point position
      glm::vec2 p =
        originalEdge.point_at(k) * (1 - u) + originalEdge.point_at(k + 1) * u;

      // *** Store the displacement distance
      const float displacementDistance = glm::distance(currentPoint, p);

      // *** Effective displacement as a result of the interpolation
      float effectiveRelaxation = 1 - settings.relaxation;

      // *** Clamp the displacement
      if (effectiveRelaxation * displacementDistance > maxDisplacement) {
        effectiveRelaxation = maxDisplacement / displacementDistance;
      }

      // *** Update the current point
      points[j] =
        currentPoint * effectiveRelaxation + p * (1 - effectiveRelaxation);

      // *** Calculate the displacement value of the point
      displacements[j] =
        effectiveRelaxation * displacementDistance / maxDisplacement;

      // *** Update the previous point
      previousPoint = currentPoint;
    }

    // *** Create the new polyline from the points and displacements
    edges.emplace_back(std::move(points), std::move(displacements));
  }

  return graph{ std::move(edges) };
}
} // namespace cubu