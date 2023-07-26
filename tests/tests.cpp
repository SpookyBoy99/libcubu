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

  printf("Edges: %zu, points: %zu\n", graph.size(), graph.point_count());
  printf("Min: %f, max: %f\n", graph.range().min, graph.range().max);

  cubu::bundling::bundling_settings bundlingSettings;
  bundlingSettings.edgeProfile = cubu::bundling::edge_profile::uniform(true);

  cubu::graph bundledGraph = cubu::bundling::bundle(graph, bundlingSettings);

  cubu::bundling::interpolation_settings interpolationSettings;

  // *** Create an interpolated graph
  cubu::graph interpolatedGraph =
    cubu::bundling::interpolate(graph, bundledGraph, interpolationSettings);

  // *** Displace the graph
  interpolatedGraph = cubu::bundling::separate_bundles(interpolatedGraph, 0.02f);

  // todo: generate density map
  // todo: generate shading map

  cubu::renderer::settings_t renderSettings;
  renderSettings.colorMode = cubu::renderer::color_mode::directional;

  auto renderedGraph = renderer.render_graph(interpolatedGraph, renderSettings);

  renderedGraph.write("/tmp/screenshot.png");
}
