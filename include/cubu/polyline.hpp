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
   * Constructs a polyline from a set of points and the get_displacement for
   * each point. Calculates properties such as bounding box and length.
   *
   * @param points       Points of the polyline.
   * @param displacement Displacement of each point (with respect to arbitrary
   *                     points).
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
   * Recalculates the bounding box and length of the polyline.
   */
  void recalculate_properties();

  /**
   * Alias for polyline::at(size_t i).
   *
   * @param i Index of the point and displacement to retrieve.
   *
   * @returns Reference to the i-th point in the polyline and it's displacement.
   */
  std::pair<point_t&, float&> operator[](size_t i);

  /**
   * Alias for polyline::at(size_t i).
   *
   * @param i Index of the point and displacement to retrieve.
   *
   * @returns Const. reference to the i-th point in the polyline and it's
   *          displacement.
   */
  std::pair<const point_t&, float> operator[](size_t i) const;

  /**
   * Retrieves the i-th point and displacement of the polyline.
   *
   * @param i Index of the point and displacement to retrieve.
   *
   * @returns Reference to the i-th point in the polyline and it's displacement.
   */
  [[nodiscard]] std::pair<point_t&, float&> at(size_t i);

  /**
   * Retrieves the i-th point and displacement of the polyline.
   *
   * @param i Index of the point and displacement to retrieve.
   *
   * @returns Const. reference to the i-th point in the polyline and it's
   *          displacement.
   */
  [[nodiscard]] std::pair<const point_t&, float> at(size_t i) const;

  /**
   * Retrieves the i-th point of the polyline.
   *
   * @param i Index of the point to retrieve.
   *
   * @returns Reference to the i-th point in the polyline.
   */
  [[nodiscard]] point_t& point_at(size_t i);

  /**
   * Retrieves the i-th point of the polyline.
   *
   * @param i Index of the point to retrieve.
   *
   * @returns Const. reference to the i-th point in the polyline.
   */
  [[nodiscard]] const point_t& point_at(size_t i) const;

  /**
   * Retrieves the displacement of the i-th point of the polyline.
   *
   * @param i Index of the point to retrieve the displacement of.
   *
   * @returns Reference to the displacement of the i-th point in the polyline.
   */
  [[nodiscard]] float& displacement_at(size_t i);

  /**
   * Retrieves the displacement of the i-th point of the polyline.
   *
   * @param i Index of the point to retrieve the displacement of.
   *
   * @returns Displacement of the i-th point in the polyline.
   */
  [[nodiscard]] float displacement_at(size_t i) const;

  /**
   * Getter for the first and last point of the polyline.
   *
   * @returns Polyline endpoints.
   */
  [[nodiscard]] std::pair<point_t, point_t> endpoints() const;

  /**
   * Sets the points of the polyline and the displacements for these points and
   * recalculates its properties
   *
   * @param points       Points of the polyline.
   * @param displacement Displacement of each point (with respect to arbitrary
   *                     points).
   */
  void points(std::vector<point_t> points, std::vector<float> displacements);

  /**
   * Sets the points of the polyline and recalculates its properties.
   *
   * @param points Points of the polyline.
   */
  void points(std::vector<point_t> points);

  /**
   * Getter for the points of the polyline.
   *
   * @returns Polyline points.
   */
  [[nodiscard]] const std::vector<point_t>& points() const;

  /**
   * Sets the displacements for the polyline points.
   *
   * @param displacement Displacement of each point (with respect to arbitrary
   *                     points).
   */
  void displacements(std::vector<float> displacements);

  /**
   * Getter for the point displacements of the polyline.
   *
   * @return Polyline point displacements.
   */
  [[nodiscard]] const std::vector<float>& displacements() const;

  /**
   * Getter for the number of points of the polyline.
   *
   * @returns Number of polyline points.
   */
  [[nodiscard]] size_t size() const;

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
  [[nodiscard]] const bounds_t& bounds() const;

private:
  void reset_properties();

  void calculate_properties();

private:
  std::vector<point_t> points_;
  std::vector<float> displacements_;
  float length_;
  bounds_t bounds_;
};
} // namespace cubu

#endif // CUBU_POLYLINE_HPP
