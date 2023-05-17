#include <glm/geometric.hpp>
#include <glm/vec3.hpp>
#include "cubu/bundling.hpp"

namespace cubu {
graph
bundling::interpolate(const graph& originalGraph,
                      const graph& bundledGraph,
                      interpolation_settings_t settings)
{
  // *** Create a vector for all the edges of the interpolated graph
  std::vector<std::unique_ptr<polyline>> edges(bundledGraph.edges().size());

  // *** Loop over all the edges in the source and bundled graph
  for (size_t i = 0; i < originalGraph.edges().size(); i++) {
    const auto& originalEdge = originalGraph.edges()[i];
    const auto& bundledEdge = bundledGraph.edges()[i];

    const float maxDisplacement =
      settings.absoluteDisplacement
        ? settings.displacementMax
        : settings.displacementMax * originalEdge->length();

    // *** Create a vector for all the points of the edge and the displacements
    std::vector<point_t> points(bundledEdge->points().size());
    std::vector<float> displacements(bundledEdge->points().size());

    // *** Copy the first and last points from the original target edge as they
    // are not interpolated
    points.front() = bundledEdge->points().front();
    points.back() = bundledEdge->points().back();

    // *** Get the previous point
    auto previousPoint = points.front();

    // *** Accumulate distance from starting point to current point
    float distanceToStart = 0;

    for (size_t j = 1; j < bundledEdge->points().size() - 1; j++) {
      // *** Get the next point
      auto currentPoint = bundledEdge->at(j);

      // *** Calculate the distance between the previous and current point and
      // add it to the accumulated distance from the start
      distanceToStart += glm::distance(previousPoint, currentPoint);

      // *** Calculate the arc-length of the i-th point of the target edge [0,1]
      const float t = distanceToStart / bundledEdge->length();

      // *** Calculate the corresponding index for the source (resampled target
      // line has more points)
      auto k = static_cast<size_t>(
        t * static_cast<float>(originalEdge->points().size() - 1));

      // *** Calculate how far between the current and next point from the
      // source it is (assuming linear uniform sampling of the target)
      const float u = t * static_cast<float>(originalEdge->points().size() - 1) -
                      static_cast<float>(k);

      // *** Calculate the interpolated point position
      glm::vec2 p =
        originalEdge->points()[k] * (1 - u) + originalEdge->points()[k + 1] * u;

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
    edges[i] =
      std::make_unique<polyline>(std::move(points), std::move(displacements));
  }

  return graph{ std::move(edges) };
}
} // namespace cubu