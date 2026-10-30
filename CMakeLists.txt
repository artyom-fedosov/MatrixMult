cmake_minimum_required(VERSION 4.0)

# `import std;` is still behind an experimental gate in CMake. The UUID is tied
# to the CMake release; if configuring fails, look it up in
# Help/dev/experimental.rst of your CMake version.
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "451f2fe2-a8a2-47c3-bc32-94786d8fc91b")

project(MatrixMult
    VERSION 1.0.0
    DESCRIPTION "Matrix multiplication showcasing OpenMP with C++23 modules"
    LANGUAGES CXX)

if(NOT CMAKE_GENERATOR MATCHES "Ninja")
    message(FATAL_ERROR "C++ modules require the Ninja generator (cmake -G Ninja)")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_MODULE_STD ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(MATMUL_NATIVE "Optimize for the build machine (-march=native)" ON)
option(MATMUL_ENABLE_OFFLOAD "Let `target` regions use configured offload devices" OFF)

find_package(OpenMP 4.5 REQUIRED COMPONENTS CXX)
message(STATUS "OpenMP ${OpenMP_CXX_VERSION} (${OpenMP_CXX_FLAGS})")

add_library(matmul_core)
target_sources(matmul_core
    PUBLIC
        FILE_SET CXX_MODULES
        BASE_DIRS src
        FILES
            src/matmul.cppm
            src/matrix.cppm
            src/timer.cppm
            src/verify.cppm
            src/runtime.cppm
            src/kernels.cppm
            src/kernels/serial.cppm
            src/kernels/parallel_for.cppm
            src/kernels/collapse_schedule.cppm
            src/kernels/simd.cppm
            src/kernels/tiled.cppm
            src/kernels/manual_spmd.cppm
            src/kernels/tasks_recursive.cppm
            src/kernels/taskloop.cppm
            src/kernels/task_depend.cppm
            src/kernels/nested.cppm
            src/kernels/offload.cppm
)
target_link_libraries(matmul_core PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(matmul_core PRIVATE -Wall -Wextra -Wpedantic)

if(MATMUL_NATIVE)
    target_compile_options(matmul_core PUBLIC $<$<CONFIG:Release>:-march=native>)
endif()

if(MATMUL_ENABLE_OFFLOAD)
    target_compile_definitions(matmul_core PUBLIC MATMUL_OFFLOAD=1)
else()
    # `target` regions are still compiled, but always run on the host.
    target_compile_options(matmul_core PUBLIC -foffload=disable)
endif()

add_executable(matmul src/main.cpp)
target_link_libraries(matmul PRIVATE matmul_core)
target_compile_options(matmul PRIVATE -Wall -Wextra -Wpedantic)
