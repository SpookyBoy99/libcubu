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
  , displacements_(std::move(displacement))
  , length_{ 0 }
  , bounds_{ glm::vec2{ std::numeric_limits<float>::max() },
             glm::vec2{ std::numeric_limits<float>::min() } }
{
  // *** Check if arrays are equal in size
  assert(("Mismatch in point and get_displacement vectors.",
          points_.size() == displacements_.size()));

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

void
polyline::recalculate_properties()
{
  reset_properties();
  calculate_properties();
}

std::pair<point_t&, float&>
polyline::operator[](size_t i)
{
  return at(i);
}

std::pair<const point_t&, float>
polyline::operator[](size_t i) const
{
  return at(i);
}

std::pair<point_t&, float&>
polyline::at(size_t i)
{
  return { points_.at(i), displacements_.at(i) };
}

std::pair<const point_t&, float>
polyline::at(size_t i) const
{
  return { points_.at(i), displacements_.at(i) };
}

point_t&
polyline::point_at(size_t i)
{
  return points_.at(i);
}

const point_t&
polyline::point_at(size_t i) const
{
  return points_.at(i);
}

float&
polyline::displacement_at(size_t i)
{
  return displacements_.at(i);
}

float
polyline::displacement_at(size_t i) const
{
  return displacements_.at(i);
}

std::pair<point_t, point_t>
polyline::endpoints() const
{
  return { points_.front(), points_.back() };
}

void
polyline::points(std::vector<point_t> points, std::vector<float> displacements)
{
  polyline::points(std::move(points));
  polyline::displacements(std::move(displacements));
}

void
polyline::points(std::vector<point_t> points)
{
  points_ = std::move(points);
  recalculate_properties();
}

const std::vector<point_t>&
polyline::points() const
{
  return points_;
}

void
polyline::displacements(std::vector<float> displacements)
{
  assert(displacements.size() == points_.size() || displacements.size() == 0);
  displacements_ = std::move(displacements);
}

const std::vector<float>&
polyline::displacements() const
{
  return displacements_;
}

float
polyline::length() const
{
  return length_;
}

const polyline::bounds_t&
polyline::bounds() const
{
  return bounds_;
}

size_t
polyline::size() const
{
  return points_.size();
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