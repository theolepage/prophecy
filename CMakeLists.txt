cmake_minimum_required(VERSION 3.17)
project(prophecy LANGUAGES CXX)

include(CheckLanguage)
find_package(Threads)
find_package(GTest)

set(SRC
    src/main.cc
)

set(SRC_CUDA
    src/kernel.cu
    src/cuda_tools/cuda_memory.cu
)

set(SRC_TESTS
    tests/test_example.cc
    tests/test_tensor.cc
    # FIXME
)

# ---
# Prophecy
# ---

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    message("CUDA has been found")
    enable_language(CUDA)
    add_compile_definitions(CUDA_ENABLED)
    set(SRC ${SRC} ${SRC_CUDA})
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wextra -pedantic")

add_executable(${PROJECT_NAME} ${SRC})

# ---
# Unit tests
# ---

if (${GTEST_FOUND})
    enable_testing()

    # For each test file create an executable of test. Launched using ctest
    foreach(f IN LISTS SRC_TESTS)
        # Find the test_name by deleting the extension and the parent_dir
        string(REGEX REPLACE "\\.[^.]*$" "" F_WITHOUT_EXT ${f})
        string(REGEX REPLACE "[^/]*/" "" TEST_NAME ${F_WITHOUT_EXT})

        add_executable(${TEST_NAME} ${f})
        target_link_libraries(${TEST_NAME} PRIVATE ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
        set_tests_properties(${TEST_NAME} PROPERTIES TIMEOUT 2)
    endforeach()
else()
    message(WARNING "Unit tests are disabled because GTest was not found.")
endif()