#ifndef CUBU_GRAPH_HPP
#define CUBU_GRAPH_HPP

#include <memory>
#include <string>
#include "polyline.hpp"

namespace cubu {
/**
 * Container for an undirected graph consisting of a set of polylines (edges).
 */
class graph
{
public:
  /**
   * Bounding box encapsulating the graph.
   */
  typedef struct bounds
  {
    glm::vec2 min, max;
  } bounds_t;

  /**
   * Range of lengths of the edges.
   */
  typedef struct range
  {
    float min, max;
  } range_t;

  /**
   * Creates an empty graph.
   */
  explicit graph();

  /**
   * Creates a graph from a set of edges.
   *
   * @param lines Edges of the graph.
   */
  explicit graph(std::vector<std::unique_ptr<polyline>> lines);

  /**
   * Loads a graph from a text file.
   *
   * @param path          Path to the file containing the graph.
   * @param endpointsOnly Flag indicating if only the endpoints of each edge
   *                      should be kept.
   */
  explicit graph(const std::string& path, bool endpointsOnly = false);

  /**
   * Loads a graph from a text file. Fails if a graph has already been opened.
   *
   * @param path          Path to the file containing the graph.
   * @param endpointsOnly Flag indicating if only the endpoints of each edge
   *                      should be kept.
   *
   * @returns True if the graph has been loaded successfully.
   */
  bool open(const std::string& path, bool endpointsOnly = false);

  /**
   * Close an existing graph and reset all parameters. Fails if no graph was
   * open.
   *
   * @returns True if the graph has been closed successfully.
   */
  bool close();

  /**
   * Recalculates the bounds and range of lengths of the graph.
   */
  void recalculate_limits();

  /**
   * Checks if a graph has been opened.
   *
   * @returns True if a graph is opened.
   */
  [[nodiscard]] bool is_open() const;

  /**
   * Returns the bounding box encapsulating all edges in the graph.
   *
   * @returns Bounding box of the graph.
   */
  [[nodiscard]] bounds_t bounds() const;

  /**
   * Returns the range of the lengths of edges in the graph.
   *
   * @returns Min and max length of the edges in the graph.
   */
  [[nodiscard]] range_t range() const;

  /**
   * Returns all the points across all the edges in the graph.
   *
   * @returns Number of points in the graph.
   */
  [[nodiscard]] size_t point_count() const;

  /**
   * Returns const reference to the underlying vector of edges.
   *
   * @returns Const reference to the vector of edges.
   */
  [[nodiscard]] const std::vector<std::unique_ptr<polyline>>& edges() const;

private:
  /**
   * Resets the bounds and ranges to the default values.
   */
  void reset_limits();

  /**
   * Loops over all the edges and updates the bounding box and range values of
   * the graph.
   */
  void calculate_limits();

private:
  bool open_;

  bounds_t bounds_;
  range_t range_;
  size_t pointCount_;
  std::vector<std::unique_ptr<polyline>> lines_;
};
} // namespace cubu

#endif // CUBU_GRAPH_HPP
