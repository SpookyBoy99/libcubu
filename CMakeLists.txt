cmake_minimum_required(VERSION 3.22)
project(CUBu VERSION "0.0.1" LANGUAGES CXX CUDA)

# -----------------------------------
# Find packages
# -----------------------------------

find_package(CUDA REQUIRED)

# -----------------------------------
# Include the source directory
# -----------------------------------

include_directories(src)

# -----------------------------------
# Set sources and libraries
# -----------------------------------

set(SOURCES
        src/common/bundling_job.cu
        src/core/edge_profile.cu
        src/core/graph.cpp
        src/core/polyline.cpp
        src/core/random_states.cu
        src/kernels/advect_sites.cu
        src/kernels/convolution.cu
        src/kernels/init_random_states.cu
        src/kernels/generate_density_map_fast.cu
        src/kernels/resample.cu
        src/kernels/calculate_resample_count.cu
        src/kernels/smooth_lines.cu
        src/main.cpp
        )

# -----------------------------------
# Build binary and/or library
# -----------------------------------

# Add the executable and link it
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${LIBS})

# -----------------------------------
# Compiler settings
# -----------------------------------

# Set version to C++17
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(${PROJECT_NAME} PRIVATE GLM_FORCE_CUDA CUDA_VERSION=12000)


