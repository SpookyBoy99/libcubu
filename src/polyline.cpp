#include "cubu/polyline.hpp"
#include <glm/geometric.hpp>

namespace cubu {
polyline::polyline()
  : length_{ 0 }
  , bounds_{ glm::vec2{ std::numeric_limits<float>::max() },
             glm::vec2{ std::numeric_limits<float>::min() } }
{
}

polyline::polyline(std::vector<point_t> points, std::vector<float> displacement)
  : points_(std::move(points))
  , displacement_(std::move(displacement))
  , length_{ 0 }
  , bounds_{ glm::vec2{ std::numeric_limits<float>::max() },
             glm::vec2{ std::numeric_limits<float>::min() } }
{
  // *** Check if arrays are equal in size
  assert(("Mismatch in point and displacement vectors.",
          points_.size() == displacement_.size()));

  // *** Calculate the bounds and length of the line
  calculate_properties();
}

polyline::polyline(std::vector<point_t> points)
  : polyline(std::move(points), std::vector<float>(points.size()))
{
}

polyline::polyline(std::initializer_list<point_t> points)
  : polyline(std::vector(points))
{
}

const point_t&
polyline::operator[](size_t i) const
{
  return at(i);
}

void
polyline::add_point(const point_t& point, float displacement)
{
  // *** Add the point
  points_.emplace_back(point);
  displacement_.emplace_back(displacement);

  // *** Update the bounds
  bounds_.min.x = std::min(bounds_.min.x, points_.back().x);
  bounds_.min.y = std::min(bounds_.min.y, points_.back().y);
  bounds_.max.x = std::max(bounds_.max.x, points_.back().x);
  bounds_.max.y = std::max(bounds_.max.y, points_.back().y);

  // *** Update the length if needed
  if (points_.size() > 1) {
    length_ += glm::distance(points_.at(points_.size() - 2),
                             points_.at(points_.size() - 1));
  }
}

void
polyline::add_point(float x, float y, float displacement)
{
  return add_point({ x, y }, displacement);
}

void
polyline::set_point(size_t i, const point_t& point)
{
  points_.at(i) = point;
  recalculate_properties();
}

void
polyline::set_point(size_t i, const point_t& point, float displacement)
{
  set_point(i, point);
  set_displacement(i, displacement);
}

void
polyline::set_point(size_t i, float x, float y)
{
  return set_point(i, { x, y });
}

void
polyline::set_point(size_t i, float x, float y, float displacement)
{
  return set_point(i, { x, y }, displacement);
}

void
polyline::set_displacement(size_t i, float displacement)
{
  displacement_.at(i) = displacement;
}

const point_t&
polyline::at(size_t i) const
{
  return points_.at(i);
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

std::tuple<point_t, point_t>
polyline::endpoints() const
{
  return { points_.front(), points_.back() };
}

const std::vector<point_t>&
polyline::points() const
{
  return points_;
}

float
polyline::displacement(size_t i) const
{
  return displacement_.at(i);
}

const std::vector<float>&
polyline::displacement() const
{
  return displacement_;
}

void
polyline::recalculate_properties()
{
  reset_properties();
  calculate_properties();
}

void
polyline::reset_properties()
{
  length_ = 0;
  bounds_ = { glm::vec2{ std::numeric_limits<float>::max() },
              glm::vec2{ std::numeric_limits<float>::min() } };
}

void
polyline::calculate_properties()
{
  for (size_t i = 1; i < points_.size(); i++) {
    length_ += glm::distance(points_[i - 1], points_[i]);

    bounds_.min.x = std::min(bounds_.min.x, points_[i].x);
    bounds_.min.y = std::min(bounds_.min.y, points_[i].y);
    bounds_.max.x = std::max(bounds_.max.x, points_[i].x);
    bounds_.max.y = std::max(bounds_.max.y, points_[i].y);
  }
}
} // namespace cubu