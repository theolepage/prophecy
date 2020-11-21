cmake_minimum_required(VERSION 3.17)
project(prophecy LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)
find_package(Threads)
find_package(GTest)

message(STATUS "---")

# ---
# Set paths to sources
# ---

set(SRC
    src/main.cc
)

set(SRC_CUDA
    src/kernel.cu
)

set(SRC_TESTS
    tests/test_example.cc
    tests/test_tensor.cc
)

# ---
# Set compilation flags and C++ standard
# ---

set(OPT_FLAGS "-Ofast -march=native -flto")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPT_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3 -fsanitize=address")

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2")

# ---
# Detect build type (release or debug)
# ---

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")

    string(REGEX REPLACE ".*/build_" "" BUILD_DIR_NAME ${CMAKE_BINARY_DIR})
    if(${BUILD_DIR_NAME} STREQUAL "debug")
        set(CMAKE_BUILD_TYPE "Debug")
    endif()
endif(NOT CMAKE_BUILD_TYPE)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ---
# Handle CUDA: enable if nvcc has been found
# ---

if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA enabled: computations will be run on the GPU")
    enable_language(CUDA)

    add_compile_definitions(CUDA_ENABLED)
    set(SRC ${SRC} ${SRC_CUDA})
else()
    message(STATUS "CUDA disabled: computations will be run on the CPU")
endif()

# ---
# Create target `prophecy`
# ---

add_executable(${PROJECT_NAME} ${SRC})

# ---
# Create targets for unit tests
# ---

if(${GTEST_FOUND})
    message(STATUS "GTest enabled: unit tests will be generated")
    enable_testing()

    # For each test file create an executable of test. Launched using ctest
    foreach(f IN LISTS SRC_TESTS)
        # Find the test_name by deleting the extension and the parent_dir
        string(REGEX REPLACE "\\.[^.]*$" "" F_WITHOUT_EXT ${f})
        string(REGEX REPLACE "[^/]*/" "" TEST_NAME ${F_WITHOUT_EXT})

        add_executable(${TEST_NAME} ${f})
        target_link_libraries(${TEST_NAME}
            PRIVATE
            ${GTEST_BOTH_LIBRARIES}
            ${CMAKE_THREAD_LIBS_INIT})

        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
        set_tests_properties(${TEST_NAME} PROPERTIES TIMEOUT 2)
    endforeach()
else()
    message(STATUS "GTest disabled: unit tests will not be generated")
endif()

message(STATUS "---")