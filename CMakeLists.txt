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
        src/graph.cpp
        src/main.cu
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
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)


