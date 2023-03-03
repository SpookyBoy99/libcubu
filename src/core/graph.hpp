#ifndef CUBU_GRAPH_HPP
#define CUBU_GRAPH_HPP

#include <memory>
#include <string>
#include "polyline.hpp"

namespace cubu {
class graph
{
public:
  typedef struct bounds
  {
    glm::vec2 min, max;
  } bounds_t;

  typedef struct range
  {
    float min, max;
  } range_t;

  explicit graph();

  explicit graph(const std::string& path, bool endpointsOnly = false);

  bool open(const std::string& path, bool endpointsOnly = false);

  bool close();

  [[nodiscard]] bool is_open() const;

  [[nodiscard]] bounds_t bounds() const;

  [[nodiscard]] range_t range() const;

  [[nodiscard]] size_t point_count() const;

  [[nodiscard]] const std::vector<std::unique_ptr<polyline>>& edges() const;

private:
  bool open_;

  bounds_t bounds_;
  range_t range_;
  size_t pointCount_;
  std::vector<std::unique_ptr<polyline>> lines_;
};
} // namespace cubu

#endif // CUBU_GRAPH_HPP
