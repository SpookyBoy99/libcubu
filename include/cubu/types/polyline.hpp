#ifndef CUBU_POLYLINE_HPP
#define CUBU_POLYLINE_HPP

#include <glm/vec2.hpp>
#include <tuple>
#include <vector>

namespace cubu {
/**
 * Point consists of a floating point x and y position.
 */
typedef glm::vec2 point_t;

namespace types {
/**
 * Container for an edge of the graph.
 */
class polyline
{
public:
  /**
   * Bounding box encapsulating the edge.
   */
  typedef struct bounds
  {
    glm::vec2 min, max;
  } bounds_t;

public:
  /**
   * Creates an empty polyline.
   */
  explicit polyline();

  /**
   * Constructs a polyline from a set of points and the displacement for each
   * point. Calculates properties such as bounding box and length.
   *
   * @param points       Points of the polyline.
   * @param displacement Displacement of each point (with respect to arbitrary
   * points).
   */
  explicit polyline(std::vector<point_t> points,
                    std::vector<float> displacement);

  /**
   * Constructs a polyline from a set of points. Calculates properties such as
   * bounding box and length.
   *
   * @param points Points of the polyline.
   */
  explicit polyline(std::vector<point_t> points);

  /**
   * Constructs a polyline from an initializer list of points. Calculates
   * properties such as bounding box and length.
   *
   * @param points Points of the polyline.
   */
  polyline(std::initializer_list<point_t> points);

  /**
   * Alias for polyline::at(size_t i).
   *
   * @param i Index of the point to retrieve.
   *
   * @returns Const. reference to i-th the point.
   */
  const point_t& operator[](size_t i) const;

  /**
   * Adds a new point to the polyline and updates the bounding box and length of
   * the polyline.
   *
   * @param point        Position of the point.
   * @param displacement Optional displacement of the point.
   */
  void add_point(const point_t& point, float displacement = 0);

  /**
   * Adds a new point to the polyline and updates the bounding box and length of
   * the polyline.
   *
   * @param x            Horizontal position of the point.
   * @param y            Vertical position of the point.
   * @param displacement Optional displacement of the point.
   */
  void add_point(float x, float y, float displacement = 0);

  /**
   * Updates the position of a point and recalculates the bounding box and
   * length of the polyline.
   *
   * @param i     Index of the point to update.
   * @param point New position of the point.
   */
  void set_point(size_t i, const point_t& point);

  /**
   * Updates the position of a point and its displacement and recalculates the
   * bounding box and length of the polyline.
   *
   * @param i            Index of the point to update.
   * @param point        New position of the point.
   * @param displacement New displacement of the point.
   */
  void set_point(size_t i, const point_t& point, float displacement);

  /**
   * Updates the position of a point and recalculates the bounding box and
   * length of the polyline.
   *
   * @param i     Index of the point to update.
   * @param x     New horizontal position of the point.
   * @param y     New vertical position of the point.
   */
  void set_point(size_t i, float x, float y);

  /**
   * Updates the position of a point and its displacement and recalculates the
   * bounding box and length of the polyline.
   *
   * @param i            Index of the point to update.
   * @param x            New horizontal position of the point.
   * @param y            New vertical position of the point.
   * @param displacement New displacement of the point.
   */
  void set_point(size_t i, float x, float y, float displacement);

  /**
   * Updates the displacement of a point. No recalculations are performed.
   *
   * @param i            Index of the point to update.
   * @param displacement New displacement of the point.
   */
  void set_displacement(size_t i, float displacement);

  /**
   * Retrieves the i-th point of the polyline.
   *
   * @param i Index of the point to retrieve.
   *
   * @returns Const. reference to i-th the point.
   */
  [[nodiscard]] const point_t& at(size_t i) const;

  /**
   * Getter for the length of the polyline.
   *
   * @returns Polyline length.
   */
  [[nodiscard]] float length() const;

  /**
   * Getter for the bounding box of the polyline.
   *
   * @returns Polyline bounding box.
   */
  [[nodiscard]] bounds_t bounds() const;

  /**
   * Getter for the first and last point of the polyline.
   *
   * @returns Polyline endpoints.
   */
  [[nodiscard]] std::tuple<point_t, point_t> endpoints() const;

  /**
   * Getter for the points of the polyline.
   *
   * @returns Polyline points.
   */
  [[nodiscard]] const std::vector<point_t>& points() const;

  /**
   * Getter for the displacement of the i-th point.
   *
   * @param i Index of the point to retrieve the displacement of.
   *
   * @returns Displacement of the i-th point.
   */
  [[nodiscard]] float displacement(size_t i) const;

  /**
   * Getter for the point displacements of the polyline.
   *
   * @return Polyline point displacements.
   */
  [[nodiscard]] const std::vector<float>& displacement() const;

protected:
  /**
   * Recalculates the bounding box and length of the polyline.
   */
  void recalculate_properties();

private:
  void reset_properties();

  void calculate_properties();

private:
  std::vector<point_t> points_;
  std::vector<float> displacement_;
  float length_;
  bounds_t bounds_;
};
}

typedef types::polyline polyline_t;
} // namespace cubu

#endif // CUBU_POLYLINE_HPP
