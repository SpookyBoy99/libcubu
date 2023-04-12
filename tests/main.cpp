#include <iostream>
#include "cubu/graph.hpp"
#include "cubu/internal/gpu.hpp"
#include "cubu/renderer.hpp"

int
main()
{
  cubu::renderer renderer;

  cubu::graph graph("data/3.US-1");

  if (!graph.is_open())
    throw std::runtime_error("Failed to open graph");

  cubu::bundling::bundling_settings bundlingSettings;
  bundlingSettings.edgeProfile = cubu::bundling::edge_profile::uniform(true);

  cubu::graph bundledGraph = cubu::bundling::bundle(graph, bundlingSettings);

  cubu::bundling::interpolation_settings interpolationSettings;

  // *** Create an interpolated graph
  cubu::graph interpolatedGraph =
    cubu::bundling::interpolate(graph, bundledGraph, interpolationSettings);

  // todo: generate density map
  // todo: generate shading map

  //    // *** Dump the interpolated graph data
  //    for (const auto& line : interpolatedGraph.edges()) {
  //      std::cout << "--- [ Line ] ---" << std::endl;
  //
  //      for (size_t i = 0; i < line->points().size(); i++) {
  //        const auto& p = line->at(i);
  //        const auto& d = line->displacement(i);
  //
  //        std::cout << "pos = (x: " << p.x << ", y: " << p.y << "); displ = "
  //        << d << std::endl;
  //      }
  //    }

  cubu::renderer::settings_t renderSettings;
  renderSettings.colorMode = cubu::renderer::color_mode::grayscale;

  renderer.render_graph(bundledGraph, renderSettings);
}
