#include "../include/cubu/bundler.hpp"
#include "../include/cubu/interpolator.hpp"
#include "../include/cubu/renderer.hpp"
#include "../include/cubu/types/graph.hpp"
#include "cubu/internal/gpu.hpp"

int
main()
{
  cubu::renderer renderer;

  cubu::graph_t graph("data/3.US-1");

  if (!graph.is_open())
    throw std::runtime_error("Failed to open graph");

  cubu::bundler::settings_t bundleSettings;
  bundleSettings.edgeProfile = cubu::types::edge_profile::uniform(true);
  bundleSettings.fastDensity = false;

  cubu::graph_t bundledGraph = cubu::bundler::bundle(graph, bundleSettings);

  cubu::interpolator::settings_t interpolationSettings;

  // *** Create an interpolated graph
  cubu::graph_t interpolatedGraph =
    cubu::interpolator::interpolate(graph, bundledGraph, interpolationSettings);

  cubu::internal::gpu::generate_density_map();
  // todo: generate density map
  // todo: generate shading map

  cubu::renderer::settings_t renderSettings;

  renderer.render_graph(interpolatedGraph, renderSettings);
}
