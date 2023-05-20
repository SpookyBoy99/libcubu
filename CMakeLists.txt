cmake_minimum_required(VERSION 3.22)
project(CUBu VERSION "0.0.1" LANGUAGES CXX CUDA)

# -----------------------------------
# Find packages
# -----------------------------------

find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED COMPONENTS EGL)
find_package(PNG REQUIRED)

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
        src/bundling/bundle.cu
        src/bundling/edge_profile.cu
        src/bundling/interpolate.cpp
        src/bundling/separate_bundles.cpp
        src/internal/gpu/advect_points.cu
        src/internal/gpu/download_graph.cu
        src/internal/gpu/generate_density_map.cu
        src/internal/gpu/resample_edges.cu
        src/internal/gpu/smooth_edges.cu
        src/internal/gpu/upload_graph.cu
        src/internal/random_states.cu
        src/graph.cpp
        src/polyline.cpp
        src/renderer.cpp
        tests/main.cpp
        )

set(LIBS
        OpenGL::EGL
        PNG::PNG
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
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --diag-suppress 20012
        --use_fast_math
        >)


