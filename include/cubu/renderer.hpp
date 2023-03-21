#ifndef CUBU_RENDERER_HPP
#define CUBU_RENDERER_HPP

#include <EGL/egl.h>
#include "types/graph.hpp"

namespace cubu {
class renderer
{
public:
  typedef enum class color_mode
  {
    grayscale,
    rainbow,
    inverse_rainbow,
    directional,
    flat,
    density_map,
    dispaclacement
  } color_mode_t;

  typedef struct settings
  {
    /** Enable drawing of the bundled edges. */
    bool drawEdges{ true };

    /** Enable drawing of the (end)points. */
    bool drawPoints{ false };

    /** Color profile of the edges. */
    color_mode_t colorMode{ color_mode::rainbow };
  } settings_t;

  explicit renderer();

  ~renderer();

  renderer(const renderer&) = delete;            // disable copy constructor
  renderer& operator=(const renderer&) = delete; // disable copy assignment
  renderer(renderer&&) = delete;                 // disable move constructor
  renderer& operator=(renderer&&) = delete;      // disable move assignment

  void render_graph(const graph_t& graph, const settings_t& settings);

private:
  EGLDisplay eglDisplay_;
  EGLint major_, minor_;
  EGLint numConfigs_;
  EGLConfig eglConfig_;
  EGLContext eglContext_;
};
} // namespace cubu

#endif // CUBU_RENDERER_HPP
