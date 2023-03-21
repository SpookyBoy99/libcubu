cmake_minimum_required(VERSION 3.22)
project(CUBu VERSION "0.0.1" LANGUAGES CXX CUDA)

# -----------------------------------
# Find packages
# -----------------------------------

find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED COMPONENTS EGL)

# -----------------------------------
# Add the glad subdirectory
# -----------------------------------

add_subdirectory(lib/glad)

# -----------------------------------
# Include the source directory
# -----------------------------------

include_directories(src)

# -----------------------------------
# Set sources and libraries
# -----------------------------------

set(SOURCES
        src/internal/gpu/advect_points.cu
        src/internal/gpu/download_graph.cu
        src/internal/gpu/generate_density_map.cu
        src/internal/gpu/resample_edges.cu
        src/internal/gpu/smooth_edges.cu
        src/internal/gpu/upload_graph.cu
        src/internal/kernels/advect_sites.cu
        src/internal/kernels/convolution.cu
        src/internal/kernels/init_random_states.cu
        src/internal/kernels/generate_density_map_fast.cu
        src/internal/kernels/generate_density_map_slow.cu
        src/internal/kernels/resample_edges.cu
        src/internal/kernels/calculate_resample_count.cu
        src/internal/kernels/smooth_lines.cu
        src/internal/random_states.cu
        src/types/edge_profile.cu
        src/types/graph.cpp
        src/types/polyline.cpp
        src/bundler.cu
        src/interpolator.cpp
        src/renderer.cpp
        tests/main.cpp
        )

set(LIBS
        OpenGL::EGL
        glad
        )

# -----------------------------------
# Build binary and/or library
# -----------------------------------

# Add the executable and link it
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${LIBS})
target_include_directories(${PROJECT_NAME} PUBLIC include)


# -----------------------------------
# Compiler settings
# -----------------------------------

# Set version to C++17
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(${PROJECT_NAME} PRIVATE GLM_FORCE_CUDA CUDA_VERSION=12000)


