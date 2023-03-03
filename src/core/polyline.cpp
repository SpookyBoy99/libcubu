#include "polyline.hpp"
#include <glm/geometric.hpp>

namespace cubu {
polyline::polyline()
  : length_{ 0 }
  , bounds_{ glm::vec2{ std::numeric_limits<float>::max() },
             glm::vec2{ std::numeric_limits<float>::min() } }
{
}

polyline::polyline(std::vector<point_t> points)
  : points_(std::move(points))
  , length_{ 0 }
  , bounds_{ glm::vec2{ std::numeric_limits<float>::max() },
             glm::vec2{ std::numeric_limits<float>::min() } }
{
  for (size_t i = 1; i < points_.size(); i++) {
    length_ += glm::distance(points_[i - 1], points_[i]);

    bounds_.min.x = std::min(bounds_.min.x, points_[i].x);
    bounds_.min.y = std::min(bounds_.min.y, points_[i].y);
    bounds_.max.x = std::max(bounds_.max.x, points_[i].x);
    bounds_.max.y = std::max(bounds_.max.y, points_[i].y);
  }
}

polyline::polyline(std::initializer_list<point_t> points)
  : polyline(std::vector(points))
{
}

void
polyline::add_point(const point_t& point)
{
  // *** Update the bounds
  bounds_.min.x = std::min(bounds_.min.x, point.x);
  bounds_.min.y = std::min(bounds_.min.y, point.y);
  bounds_.max.x = std::max(bounds_.max.x, point.x);
  bounds_.max.y = std::max(bounds_.max.y, point.y);

  // *** Add the point
  points_.emplace_back(point);

  // *** Update the length if needed
  if (points_.size() > 1) {
    length_ += glm::distance(points_.at(points_.size() - 2),
                             points_.at(points_.size() - 1));
  }
}

void
polyline::add_point(float x, float y)
{
  add_point({ x, y });
}

float
polyline::length() const
{
  return length_;
}

polyline::bounds_t
polyline::bounds() const
{
  return bounds_;
}

std::vector<point_t>
polyline::endpoints() const
{
  return { points_.front(), points_.back() };
}

const std::vector<point_t>&
polyline::points() const
{
  return points_;
}
} // namespace cubu