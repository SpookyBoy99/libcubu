#include <iostream>
#include "common/bundling_job.hpp"
#include "core/graph.hpp"

int
main()
{
  cubu::graph graph("data/3.US-1");

  if (!graph.is_open())
    throw std::runtime_error("Failed to open graph");

  cubu::bundling_job::settings_t settings;
  settings.edgeProfile = cubu::edge_profile::uniform(true);

  cubu::bundling_job job(graph, settings);
}
