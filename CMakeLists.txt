cmake_minimum_required(VERSION 3.22)
project(CUBu VERSION "0.0.1" LANGUAGES CXX CUDA)

# -----------------------------------
# Preprocessing
# -----------------------------------

string(TOUPPER ${PROJECT_NAME} PROJECT_PREFIX)

# -----------------------------------
# Set default build to release
# -----------------------------------

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug" FORCE)
endif ()

# -----------------------------------
# Options
# -----------------------------------

option(${PROJECT_PREFIX}_BUILD_SHARED "Build shared library" OFF)
option(${PROJECT_PREFIX}_BUILD_TESTS "Build the unit tests" OFF)

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
        )

set(LIBS
        OpenGL::EGL
        PNG::PNG
        glad
        )

# -----------------------------------
# Build binary and/or library
# -----------------------------------

if (${PROJECT_PREFIX}_BUILD_SHARED OR BUILD_SHARED_LIBS)
    # Compile the sources a shared for the project
    add_library(${PROJECT_NAME} SHARED ${SOURCES})
else ()
    # Compile the sources as static for the project
    add_library(${PROJECT_NAME} STATIC ${SOURCES})
endif ()

# Add the alias
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# Link the libraries
target_link_libraries(${PROJECT_NAME} PUBLIC ${LIBS})

# Add the includes to the project
target_include_directories(${PROJECT_NAME} PUBLIC include)

# -----------------------------------
# Compiler settings
# -----------------------------------

# Set version to C++20
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(${PROJECT_NAME} PRIVATE GLM_FORCE_CUDA CUDA_VERSION=12000)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --diag-suppress 20012
        --use_fast_math
        >)

# -----------------------------------
# Testing
# -----------------------------------

# Check if testing is enabled
if (${PROJECT_PREFIX}_BUILD_TESTS)
    enable_testing()

    # Add the tests folder
    add_subdirectory(tests)
endif ()
