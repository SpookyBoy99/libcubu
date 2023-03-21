#include "cubu/interpolator.hpp"
#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

namespace cubu {
graph_t
interpolator::interpolate(const graph_t& originalGraph,
                          const graph_t& bundledGraph,
                          settings_t settings)
{
  // *** Create a vector for all the edges of the interpolated graph
  std::vector<std::unique_ptr<polyline_t>> edges(bundledGraph.edges().size());

  // *** Loop over all the edges in the source and bundled graph
  for (size_t i = 0; i < originalGraph.edges().size(); i++) {
    const auto& sourceEdge = originalGraph.edges()[i];
    const auto& targetEdge = bundledGraph.edges()[i];

    float maxDisplacement =
      settings.absoluteDisplacement
        ? settings.displacementMax *
            static_cast<float>(
              std::min(settings.resolution.x, settings.resolution.y))
        : settings.displacementMax * sourceEdge->length();

    // *** Create a vector for all the points of the edge and the displacements
    std::vector<point_t> points(targetEdge->points().size());
    std::vector<float> displacements(targetEdge->points().size());

    // *** Get the previous point
    auto previousPoint = targetEdge->points().front();

    // *** Accumulate distance from starting point to current point
    float distanceToStart = 0;

    for (size_t j = 1; j < targetEdge->points().size() - 1; j++) {
      // *** Get the next point
      auto currentPoint = targetEdge->at(j);

      // *** Calculate the distance between the previous and current point and
      // add it to the accumulated distance from the start
      distanceToStart += glm::distance(previousPoint, currentPoint);

      // *** Calculate the arc-length of the i-th point of the target edge [0,1]
      const float t = distanceToStart / targetEdge->length();

      // *** Calculate the corresponding index for the source (resampled target
      // line has more points)
      auto k = static_cast<size_t>(
        t * static_cast<float>(sourceEdge->points().size() - 1));

      // *** Calculate how far between the current and next point from the
      // source it is (assuming linear uniform sampling of the target)
      const float u = t * static_cast<float>(sourceEdge->points().size() - 1) -
                      static_cast<float>(k);

      // *** Calculate the interpolated point position
      glm::vec2 p =
        targetEdge->points()[j] * (1 - u) + targetEdge->points()[j + 1] * u;

      // *** Store the displacement distance
      const float displacementDistance = glm::distance(currentPoint, p);

      // *** Effective displacement as a result of the interpolation
      float effectiveDisplacement =
        (1 - settings.relaxation) * displacementDistance;

      // *** Clamp the displacement
      if (effectiveDisplacement > maxDisplacement) {
        effectiveDisplacement = maxDisplacement / displacementDistance;
      }

      // *** Update the current point
      points[j] =
        currentPoint * effectiveDisplacement + p * (1 - effectiveDisplacement);

      // *** Calculate the displacement value of the point
      displacements[j] =
        effectiveDisplacement * displacementDistance / maxDisplacement;

      // *** Separate bundles going in different directions
      if (settings.directionalSeparation != 0) {
        auto tan = glm::normalize(glm::vec3{ currentPoint - previousPoint, 0 });
        auto delta = glm::cross(tan, { 0, 0, 1 });

        // *** max shift should be proportional with edge length, more
        // precisely with length / resolution
        currentPoint += glm::vec2(delta.x, delta.y) *
                        settings.directionalSeparation *
                        std::pow(1.0f - 2.0f * std::abs(t - 0.5f), 0.4f);
      }

      // *** Update the previous point
      previousPoint = currentPoint;
    }

    // *** Create the new polyline from the points and displacements
    edges[i] =
      std::make_unique<polyline_t>(std::move(points), std::move(displacements));
  }

  return graph_t{ std::move(edges) };
}
} // namespace cubu