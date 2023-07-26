# -----------------------------------
# Find packages
# -----------------------------------

### None

# -----------------------------------
# Set sources and libraries
# -----------------------------------

# Add source files to the executable
set(TEST_SOURCES
        tests.cpp
        )

# Set link libraries
set(TEST_LIBS
        ${PROJECT_NAME}
        )

# -----------------------------------
# Build and add test
# -----------------------------------

# Add the executable
add_executable(${PROJECT_NAME}_tests ${TEST_SOURCES})

# Register the test
add_test(NAME ${PROJECT_NAME}_tests COMMAND ${PROJECT_NAME}_tests)

# Link the libraries
target_link_libraries(${PROJECT_NAME}_tests PRIVATE ${TEST_LIBS})

# -----------------------------------
# Compiler settings
# -----------------------------------

# Set version to C++20
set_property(TARGET ${PROJECT_NAME}_tests PROPERTY CXX_STANDARD 20)

# Compile options for all
target_compile_options(${PROJECT_NAME}_tests PRIVATE
        -Wall -Wextra -pedantic -Werror
        $<$<CONFIG:DEBUG>:-O0 -g3>
        $<$<CONFIG:RELEASE>:-O3>
        )
