#include "cubu/graph.hpp"
#include <fstream>
#include <optional>

namespace cubu {
graph::graph()
  : open_{ false }
  , bounds_{ glm::vec2{ std::numeric_limits<float>::max() },
             glm::vec2{ std::numeric_limits<float>::min() } }
  , range_{ std::numeric_limits<float>::max(),
            std::numeric_limits<float>::min() }
  , pointCount_{ 0 }
{
}

graph::graph(const std::string& path, bool endpointsOnly)
  : graph()
{
  open(path, endpointsOnly);
}

graph::graph(std::vector<polyline> edges)
  : open_{ true }
  , bounds_{ glm::vec2{ std::numeric_limits<float>::max() },
             glm::vec2{ std::numeric_limits<float>::min() } }
  , range_{ std::numeric_limits<float>::max(),
            std::numeric_limits<float>::min() }
  , pointCount_{ 0 }
  , edges_(std::move(edges))
{
  calculate_limits();
}

graph::graph(std::initializer_list<polyline> edges)
  : graph(std::vector(edges))
{
}

bool
graph::open(const std::string& path, bool endpointsOnly)
{
  // *** If a file is already opened, fail
  if (is_open()) {
    return false;
  }

  // *** Open the file in a file stream
  std::ifstream fs(path);

  // *** Check if the file opening succeeded
  if (!fs.is_open()) {
    return false;
  }

  // *** Set open to true
  open_ = true;

  // *** Optional list of points, will be null_opt if no line is being loaded
  std::optional<std::vector<point_t>> points;

  // *** Check if the end of the file has been reached
  bool eof = false;

  // *** Loop over the file token by token
  std::string token;

  // *** Read until no tokens are left
  while (!eof) {
    // *** Read the next token
    eof = !(fs >> token);

    // *** Check if the end of the line (or file) has been reached
    if (eof || token.ends_with(':')) {
      // *** Check if we are working on constructing a polyline
      if (points) {
        // *** Convert into polyline
        polyline line(std::move(points.value()));

        // *** Reset the optional
        points.reset();

        // *** Update the point count
        pointCount_ += line.size();

        // *** Update the bounds
        bounds_.min.x = std::min(bounds_.min.x, line.bounds().min.x);
        bounds_.min.y = std::min(bounds_.min.y, line.bounds().min.y);
        bounds_.max.x = std::max(bounds_.max.x, line.bounds().max.x);
        bounds_.max.y = std::max(bounds_.max.y, line.bounds().max.y);

        // *** Update the min max lengths found
        range_.min = std::min(range_.min, line.length());
        range_.max = std::max(range_.max, line.length());

        // *** Move the polyline into the list of lines
        edges_.emplace_back(std::move(line));
      }

      // *** Start on constructing a new poly line if the end hasn't been
      // reached yet
      if (!eof && std::all_of(token.begin(), token.end() - 1, ::isdigit)) {
        points = std::make_optional<std::vector<point_t>>();
      }
    } else if (points && !token.starts_with('(') && !token.starts_with('<')) {
      // *** Store the x of the point before reading the next point
      float x = std::stof(token);

      // *** Read the next token, stop loading if it fails
      if (!(fs >> token)) {
        // *** Close the file and reset if failure occurs
        return !close();
      }

      // *** Parse the y coordinate
      float y = std::stof(token);

      // *** Check if endpoints only is disabled, or the two points hasn't been
      // reached yet
      if (!endpointsOnly || points->size() < 2) {
        // *** Add the point
        points->emplace_back(x, y);
      } else {
        // *** Update the endpoints position
        points->at(1) = { x, y };
      }
    }
  }

  return true;
}

bool
graph::close()
{
  // *** If a file is not opened, fail
  if (!is_open()) {
    return false;
  }

  // *** Set open to false
  open_ = false;

  // *** Set point count back to zero
  pointCount_ = 0;

  // *** Remove all the lines
  edges_.clear();

  // *** Reset the calculated limits
  reset_limits();

  return true;
}

void
graph::recalculate_limits()
{
  reset_limits();
  calculate_limits();
}

bool
graph::is_open() const
{
  return open_;
}

polyline&
graph::operator[](size_t i)
{
  return at(i);
}

const polyline&
graph::operator[](size_t i) const
{
  return at(i);
}

polyline&
graph::at(size_t i)
{
  return edges_.at(i);
}

const polyline&
graph::at(size_t i) const
{
  return edges_.at(i);
}

void
graph::edges(std::vector<polyline> edges)
{
  edges_ = std::move(edges);
  recalculate_limits();
}

const std::vector<polyline>&
graph::edges() const
{
  return edges_;
}

size_t
graph::size() const
{
  return edges_.size();
}

size_t
graph::point_count() const
{
  return pointCount_;
}

graph::bounds_t
graph::bounds() const
{
  return bounds_;
}

graph::range_t
graph::range() const
{
  return range_;
}

void
graph::reset_limits()
{
  // *** Reset the min and max values
  bounds_ = { glm::vec2{ std::numeric_limits<float>::max() },
              glm::vec2{ std::numeric_limits<float>::min() } };
  range_ = { std::numeric_limits<float>::max(),
             std::numeric_limits<float>::min() };
}

void
graph::calculate_limits()
{
  for (const auto& line : edges_) {
    // *** Update the point count
    pointCount_ += line.size();

    // *** Update the bounds
    bounds_.min.x = std::min(bounds_.min.x, line.bounds().min.x);
    bounds_.min.y = std::min(bounds_.min.y, line.bounds().min.y);
    bounds_.max.x = std::max(bounds_.max.x, line.bounds().max.x);
    bounds_.max.y = std::max(bounds_.max.y, line.bounds().max.y);

    // *** Update the min max lengths found
    range_.min = std::min(range_.min, line.length());
    range_.max = std::max(range_.max, line.length());
  }
}
} // namespace cubu