#ifndef CUBU_POLYLINE_HPP
#define CUBU_POLYLINE_HPP

#include <glm/vec2.hpp>
#include <vector>

namespace cubu {
typedef glm::vec2 point_t;

class polyline
{
public:
  typedef struct bounds
  {
    glm::vec2 min, max;
  } bounds_t;

public:
  explicit polyline();

  explicit polyline(std::vector<point_t> points);

  polyline(std::initializer_list<point_t> points);

  void add_point(const point_t& point);

  void add_point(float x, float y);

  [[nodiscard]] float length() const;

  [[nodiscard]] bounds_t bounds() const;

  [[nodiscard]] std::vector<point_t> endpoints() const;

  [[nodiscard]] const std::vector<point_t>& points() const;

private:
  std::vector<point_t> points_;
  float length_;
  bounds_t bounds_;
};
} // namespace cubu

#endif // CUBU_POLYLINE_HPP
