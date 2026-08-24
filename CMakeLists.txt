cmake_minimum_required(VERSION 3.18)
project(scgm_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(LIB_SOURCES
    src/graph.cpp
    src/features.cpp
    src/cosine_similarity.cpp
    src/candidate_generation.cpp
    src/csr_matrix.cpp
    src/lapjv.cpp
    src/gpu_matching.cu
    src/evaluation.cpp
)

add_library(scgm_lib STATIC ${LIB_SOURCES})
target_include_directories(scgm_lib PUBLIC ${INCLUDE_DIR})
target_link_libraries(scgm_lib PUBLIC CUDA::cudart)
set_target_properties(scgm_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

add_executable(scgm_cuda src/main.cpp)
target_link_libraries(scgm_cuda PRIVATE scgm_lib)

add_executable(toy_scgm examples/toy_scgm.cpp)
target_link_libraries(toy_scgm PRIVATE scgm_lib)

enable_testing()

set(TEST_SOURCES
    tests/test_features.cpp
    tests/test_permutation.cpp
    tests/test_csr.cpp
    tests/test_topk.cpp
    tests/test_lapjv.cpp
    tests/test_cpu_gpu_parity.cpp
    tests/test_toy_scgm.cpp
)

foreach(test_file ${TEST_SOURCES})
    get_filename_component(test_name ${test_file} NAME_WE)
    add_executable(${test_name} ${test_file})
    target_link_libraries(${test_name} PRIVATE scgm_lib)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")