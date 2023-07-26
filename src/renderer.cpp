#include "cubu/renderer.hpp"
#include <fcntl.h>
#include <glad/glad.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/color_space.hpp>
#include <glm/gtx/vector_angle.hpp>
#include <glm/vec4.hpp>
#include <png++/png.hpp>
#include <queue>
#include <stdexcept>

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
renderer::renderer(glm::ivec2 resolution)
  : resolution_{ resolution }
  , eglDisplay_{}
  , major_{}
  , minor_{}
  , numConfigs_{}
  , eglConfig_{}
  , eglContext_{}
  , frameBuffer_{}
  , colorBuffer_{}
  , depthBuffer_{}
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

  // *** Generate the frame buffers and textures
  glGenFramebuffers(1, &frameBuffer_);
  glGenTextures(1, &colorBuffer_);
  glGenRenderbuffers(1, &depthBuffer_);

  // *** Bind the generated frame buffer
  glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer_);

  // *** Create a texture to bind the framebuffer to
  glBindTexture(GL_TEXTURE_2D, colorBuffer_);
  glTexImage2D(GL_TEXTURE_2D,
               0,
               GL_RGB8,
               resolution_.x,
               resolution_.y,
               0,
               GL_RGBA,
               GL_UNSIGNED_BYTE,
               nullptr);

  // *** Enable linear filtering for the texture
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  // *** Attach the frame buffer to the image
  glFramebufferTexture2D(
    GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorBuffer_, 0);

  // *** Attach the frame buffer to the depth image
  glBindRenderbuffer(GL_RENDERBUFFER, depthBuffer_);
  glRenderbufferStorage(
    GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, resolution_.x, resolution_.y);
  glFramebufferRenderbuffer(
    GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthBuffer_);

  // *** Check the status of the framebuffer
  switch (glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER)) {
    case GL_FRAMEBUFFER_COMPLETE:
      break;
    case GL_FRAMEBUFFER_UNSUPPORTED:
      throw std::runtime_error("Unsupported framebuffer type.");
    default:
      throw std::runtime_error("Framebuffer Error");
  }
}

renderer::~renderer()
{
  eglTerminate(eglDisplay_);
}

png::image<png::rgba_pixel>
renderer::render_graph(const graph& graph, const settings_t& settings) const
{
  glm::vec3 pointColor{ 1.0f, 0.0f, 0.0f };

  // *** Get the offset
  glm::vec2 offset = graph.bounds().min;

  // *** Get the range of the graph
  glm::vec2 range = graph.bounds().max - graph.bounds().min;

  // *** Calculate the scale
  float scale = range.x > range.y ? 2.0f / range.x : 2.0f / range.y;

  // *** Calculate the translation
  glm::vec2 translation = { (2.0f - scale * range.x) / 2 - 1.0f,
                            (2.0f - scale * range.y) / 2 - 1.0f };

  // *** Calculate the edge and point counts
  size_t pointCount = graph.point_count(), edgeCount = graph.edges().size();

  // *** Create a new vector for all the vertices
  std::vector<glm::vec2> vertexBuffer;

  // *** Allocate the memory for all the points and end of line markers
  vertexBuffer.reserve(pointCount);

  // *** Create a new vector for all the vertex colors
  std::vector<glm::vec4> colorBuffer;

  // *** Only reserve memory for the colors if the edges are drawn
  if (settings.drawEdges) {
    // *** Allocate the memory for all the points and end of line markers
    colorBuffer.reserve(pointCount);
  }

  // *** Create a new vector for all the edge indices
  std::vector<int> edgeIndices;

  // *** Allocate the memory for the edge indices
  edgeIndices.reserve(edgeCount);

  // *** Create a new vector for all the draw order
  std::vector<std::tuple<float, size_t>> drawOrder;

  // *** Allocate the memory for the edge indices
  drawOrder.reserve(edgeCount);

  //  const float rhoMax = settings.densityMax ? *settings.densityMax : 1;

  // *** Loop over all the points in the graph
  for (const auto& line : graph.edges()) {
    // *** Add the line to the draw order based on length
    drawOrder.emplace_back(line.length(), edgeIndices.size());

    // *** Add the starting point of the next polyline to the list of edge
    // indices
    edgeIndices.emplace_back(vertexBuffer.size());

    // *** Normalize the line length
    float normalizedLength = (line.length() - graph.range().min) /
                             (graph.range().max - graph.range().min);

    // *** Create the base color
    glm::vec4 color{ 1.0f };

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
        // *** Get the endpoints of the trail to calculate the general direction of the trail
        const auto &[from, to] = line.endpoints();

        // *** Calculate the angle between the vector from the start point to
        // the end point and the (inverted) x-axis
        auto angle = glm::degrees(
          glm::orientedAngle(glm::normalize(to - from), glm::vec2{ -1, 0 }));

        // *** Convert the angle range from [-180, 180] to [0, 360] for the hue,
        // and set the saturation and value as well
        glm::vec3 hsv{ 180 - angle, std::pow(normalizedLength, 0.3f), 1.0f };

        // *** Convert the color from hsv to rgb and apply
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

    for (size_t i = 0; i < line.size(); i++) {
      // *** Get the point
      const auto& point = line[i];

      // *** Calculate t
      auto t =
        static_cast<float>(i) / static_cast<float>(line.size() - 1);
      assert(t <= 1);
      t = 0.2f + 0.8f * std::pow(1.0f - 2.0f * std::abs(t - 0.5f), 0.5f);
      t = normalizedLength + (1.0f - normalizedLength) * 0.5f * t;

      // *** Only process colors if the edges are drawn
      if (settings.drawEdges) {
        // *** Update the color if it is a per vertex based color
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
      vertexBuffer.emplace_back((point - offset) * scale + translation);
      colorBuffer.emplace_back(
        color *
        glm::vec4{ glm::vec3{ t },
                   (1.0f - normalizedLength) + normalizedLength * t * 0.5f });
    }
  }

  // *** Create a vector for the draw order
  std::priority_queue<decltype(drawOrder)::value_type,
                      std::vector<decltype(drawOrder)::value_type>,
                      std::greater<>>
    drawOrderQueue(std::greater<>(), std::move(drawOrder));

  glBindTexture(GL_TEXTURE_2D, 0);
  glEnable(GL_TEXTURE_2D);
  glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer_);

  glViewport(0, 0, resolution_.x, resolution_.y);

  glClearColor(1, 1, 1, 1);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(-1, 1, -1, 1, -1, 1);
  glScalef(1, -1, 1);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // *** Enable blending and line smoothing
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_LINE_SMOOTH);
  glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
  glShadeModel(GL_SMOOTH);
  glEnable(GL_POINT_SMOOTH);

  // *** Set line and point sizes
  glLineWidth(1.0f);
  glPointSize(1.5f);

  // *** Set the buffer pointers
  glVertexPointer(2, GL_FLOAT, 0, vertexBuffer.data());
  glColorPointer(4, GL_FLOAT, 0, colorBuffer.data());

  glEnableClientState(GL_VERTEX_ARRAY);

  // *** Loop over all the values in the priority queue
  while (!drawOrderQueue.empty()) {
    // *** Get the index of the edge index
    const size_t i = std::get<1>(drawOrderQueue.top());

    // *** Pop the value
    drawOrderQueue.pop();

    // *** Get the index of the first point of the edge
    const int pointIndexStart = edgeIndices[i],
              pointIndexEnd = i == edgeCount - 1 ? static_cast<int>(pointCount)
                                                 : edgeIndices[i + 1];

    // *** Draw the edge
    if (settings.drawEdges) {
      glEnableClientState(GL_COLOR_ARRAY);
      glDrawArrays(
        GL_LINE_STRIP, pointIndexStart, pointIndexEnd - pointIndexStart);
    }

    // *** Draw the points
    if (settings.drawPoints) {
      glDisableClientState(GL_COLOR_ARRAY);
      glColor3fv(glm::value_ptr(pointColor));
      glDrawArrays(GL_POINTS, pointIndexStart, pointIndexEnd - pointIndexStart);
    }
  }

  // *** Create a buffer for the pixels
  std::vector<unsigned char> image(resolution_.x * resolution_.y * 4);

  // *** Read the screen image into the buffer
  glReadPixels(0,
               0,
               resolution_.x,
               resolution_.y,
               GL_RGBA,
               GL_UNSIGNED_BYTE,
               image.data());

  // *** Create a new empty png image
  png::image<png::rgba_pixel> pngImage(resolution_.x, resolution_.y);

  // *** Copy all the pixels to the image
  for (int y = 0; y < resolution_.y; y++) {
    for (int x = 0; x < resolution_.x; x++) {
      // *** Calculate the index of the pixel
      size_t i = (y * resolution_.x + x) * 4;

      // *** Update said pixel
      pngImage.set_pixel(
        x,
        y,
        png::rgba_pixel(image[i], image[i + 1], image[i + 2], image[i + 3]));
    }
  }

  // *** Disable the vertex and color arrays
  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);

  // *** Disable the blending and line smoothing and reset all the values
  glDisable(GL_BLEND);
  glDisable(GL_LINE_SMOOTH);
  glLineWidth(1.0f);
  glPointSize(1.f);
  glDisable(GL_POINT_SMOOTH);

  // *** Return the png image object
  return pngImage;
}
} // namespace cubu