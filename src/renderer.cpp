#include "cubu/renderer.hpp"
#include <glm/gtx/color_space.hpp>
#include <glm/gtx/vector_angle.hpp>
#include <glm/vec4.hpp>
#include <stdexcept>
#include "glad/glad.h"

static const EGLint configAttribs[] = { EGL_SURFACE_TYPE,
                                        EGL_PBUFFER_BIT,
                                        EGL_BLUE_SIZE,
                                        8,
                                        EGL_GREEN_SIZE,
                                        8,
                                        EGL_RED_SIZE,
                                        8,
                                        EGL_DEPTH_SIZE,
                                        8,
                                        EGL_RENDERABLE_TYPE,
                                        EGL_OPENGL_BIT,
                                        EGL_NONE };

namespace cubu {
renderer::renderer()
  : eglDisplay_{}
  , major_{}
  , minor_{}
  , numConfigs_{}
  , eglConfig_{}
  , eglContext_{}
{
  // *** Get a "virtual" EGL display
  eglDisplay_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);

  // *** Initialize EGL
  eglInitialize(eglDisplay_, &major_, &minor_);

  // *** Select an appropriate configuration
  eglChooseConfig(eglDisplay_, configAttribs, &eglConfig_, 1, &numConfigs_);

  // *** Bind the API
  eglBindAPI(EGL_OPENGL_API);

  // *** Create a context
  eglContext_ =
    eglCreateContext(eglDisplay_, eglConfig_, EGL_NO_CONTEXT, nullptr);

  // *** Make the context current
  eglMakeCurrent(eglDisplay_, EGL_NO_SURFACE, EGL_NO_SURFACE, eglContext_);

  // *** Load GL (extensions)
  if (!gladLoadGL()) {
    throw std::runtime_error("Failed to initialize OpenGL Context.");
  }
}

renderer::~renderer()
{
  eglTerminate(eglDisplay_);
}

void
renderer::render_graph(const graph_t& graph, const settings_t& settings)
{
  // *** Create the buffers for the points and edge colors
  std::vector<glm::vec2> vertexBuffer(graph.point_count());
  std::vector<glm::vec4> colorBuffer(settings.drawEdges ? graph.point_count()
                                                        : 0);

//  const float rhoMax = settings.densityMax ? *settings.densityMax : 1;

  // *** Keep track of the current point index
  size_t pointIndex = 0;

  // *** Loop over all the points in the graph
  for (const auto& line : graph.edges()) {
    // *** Normalize the line length
    float normalizedLength = (line->length() - graph.range().min) /
                             (graph.range().max - graph.range().min);

    // *** Create the base color
    glm::vec4 color;

    // *** Update the color based on the selected profile if the color is the
    // same for the entire line:
    switch (settings.colorMode) {
      case color_mode::flat: {
        color = { 0.9f, 0.9f, 0.9f, color.a };
        break;
      }
      case color_mode::grayscale: {
        color = {
          normalizedLength, normalizedLength, normalizedLength, color.a
        };
        break;
      }
      case color_mode::directional: {
        auto [from, to] = line->endpoints();
        glm::vec3 hsv{ glm::angle(from, to),
                       std::pow(normalizedLength, 0.3f),
                       1.0f };

        // *** Update the color
        color = { glm::rgbColor(hsv), color.a };
        break;
      }
      case color_mode::rainbow:
      case color_mode::inverse_rainbow: {
        const float dx = 0.8f,
                    l = settings.colorMode == color_mode::rainbow
                          ? normalizedLength
                          : (1 - normalizedLength),
                    v = (6 - 2 * dx) * l + dx;

        // *** Update the color
        color = { std::max(0.0f, (3 - std::abs(v - 4) - std::abs(v - 5)) / 2),
                  std::max(0.0f, (4 - std::abs(v - 2) - std::abs(v - 4)) / 2),
                  std::max(0.0f, (3 - std::abs(v - 1) - std::abs(v - 2)) / 2),
                  color.a };
      }
    }

    //    // 2. Determine edge transparency
    //    float2alpha(1 - val, alpha); // Edge base-alpha maps edge length
    //    alpha *= global_alpha;       // Modulate above with global
    //    transparency

    for (const auto& point : line->points()) {
      // *** Only process colors if the edges are drawn
      if (settings.drawEdges) {
        // *** Update the color if it is is a per vertex based color
        switch (settings.colorMode) {
          case color_mode::density_map:
            // float2rgb rho <- densityMap[x, y] / maxrho
            break;
          case color_mode::dispaclacement:
            // float2rgb: line.displacement(i)
            break;
        }
      }

      // *** Add the points to the vertex buffer
      vertexBuffer[pointIndex] = point;

      // *** Increment the point index
      pointIndex++;
    }
  }

  // *** Enable blending and line smoothing
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_LINE_SMOOTH);
  glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
  glShadeModel(GL_SMOOTH);
  glEnable(GL_POINT_SMOOTH);

  // *** Set line and point sizes
  glLineWidth(1.0);
  glPointSize(1.5);

  // *** Specify the vertex attributes (e.g. positions and colors)
  glVertexPointer(2, GL_FLOAT, 0, vertexBuffer.data());
  glColorPointer(4, GL_UNSIGNED_BYTE, 0, colorBuffer.data());
}
} // namespace cubu