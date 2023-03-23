#include "cubu/graph.hpp"
#include <fstream>

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

graph::graph(std::vector<std::unique_ptr<polyline>> lines)
  : open_{ true }
  , bounds_{ glm::vec2{ std::numeric_limits<float>::max() },
             glm::vec2{ std::numeric_limits<float>::min() } }
  , range_{ std::numeric_limits<float>::max(),
            std::numeric_limits<float>::min() }
  , pointCount_{ 0 }
  , lines_(std::move(lines))
{
  calculate_limits();
}

graph::graph(const std::string& path, bool endpointsOnly)
  : graph()
{
  open(path, endpointsOnly);
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

  // *** Pointer to the line that is currently being loaded
  std::unique_ptr<polyline> line;

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
      if (line) {
        // *** Update the polyline to only keep the beginning and endpoint
        if (endpointsOnly) {
          const auto& [start, end] = line->endpoints();
          line = std::make_unique<polyline>(std::vector{ start, end });
        }

        // *** Update the point count
        pointCount_ += line->points().size();

        // *** Update the bounds
        bounds_.min.x = std::min(bounds_.min.x, line->bounds().min.x);
        bounds_.min.y = std::min(bounds_.min.y, line->bounds().min.y);
        bounds_.max.x = std::max(bounds_.max.x, line->bounds().max.x);
        bounds_.max.y = std::max(bounds_.max.y, line->bounds().max.y);

        // *** Update the min max lengths found
        range_.min = std::min(range_.min, line->length());
        range_.max = std::max(range_.max, line->length());

        // *** Move the polyline into the list of lines
        lines_.emplace_back(std::move(line));

        // *** Reset the pointer
        line.reset();
      }

      // *** Start on constructing a new poly line if the end hasn't been
      // reached yet
      if (!eof && std::all_of(token.begin(), token.end() - 1, ::isdigit)) {
        line = std::make_unique<polyline>();
      }
    } else if (line && !token.starts_with('(') && !token.starts_with('<')) {
      // *** Store the x of the point before reading the next point
      float x = std::stof(token);

      // *** Read the next token, stop loading if it fails
      if (!(fs >> token)) {
        // *** Close the file and reset if failure occurs
        return !close();
      }

      // *** Parse the y coordinate
      float y = std::stof(token);

      // *** Add the point
      line->add_point(x, y);
    }
  }

  printf("Edges: %zu, points: %zu\n", lines_.size(), pointCount_);
  printf("Min: %f, max: %f\n", range_.min, range_.max);

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
  lines_.clear();

  // *** Reset the calculated limits
  reset_limits();

  return true;
}

bool
graph::is_open() const
{
  return open_;
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

size_t
graph::point_count() const
{
  return pointCount_;
}

const std::vector<std::unique_ptr<polyline>>&
graph::edges() const
{
  return lines_;
}

void
graph::recalculate_limits()
{
  reset_limits();
  calculate_limits();
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
  for (const auto& line : lines_) {
    // *** Update the point count
    pointCount_ += line->points().size();

    // *** Update the bounds
    bounds_.min.x = std::min(bounds_.min.x, line->bounds().min.x);
    bounds_.min.y = std::min(bounds_.min.y, line->bounds().min.y);
    bounds_.max.x = std::max(bounds_.max.x, line->bounds().max.x);
    bounds_.max.y = std::max(bounds_.max.y, line->bounds().max.y);

    // *** Update the min max lengths found
    range_.min = std::min(range_.min, line->length());
    range_.max = std::max(range_.max, line->length());
  }
}
} // namespace cubu