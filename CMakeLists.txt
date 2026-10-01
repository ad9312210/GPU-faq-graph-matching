cmake_minimum_required(VERSION 3.18)
project(scgm_cuda LANGUAGES CXX CUDA)

option(SCGM_BUILD_TESTS "Build native SCGM tests" ON)
option(SCGM_ENABLE_WARNINGS "Enable compiler warnings" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(SCGM_CUDA_ARCHITECTURES "native" CACHE STRING
    "CUDA architectures, e.g. native or 89; use empty for compiler default")
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR
   CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "${SCGM_CUDA_ARCHITECTURES}")
endif()

find_package(CUDAToolkit REQUIRED)
find_path(NVML_INCLUDE_DIR nvml.h)
find_library(NVML_LIBRARY NAMES nvidia-ml)

#set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
# the reason being teh headers like graph.hpp, common.hpp, gpu_matching.hpp are in the same directory as the source files, so we can just include the current source directory
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(LIB_SOURCES
    #src/graph.cpp
    src/graph.cpp

    #src/features.cpp
    src/features.cpp

    #src/cosine_similarity.cpp
    src/cosine_similarity.cpp

    #src/candidate_generation.cpp
    src/candidate_generation.cpp
    src/bipartite_matching.cpp
    src/sparse_sinkhorn.cpp
    src/device_memory_sampler.cpp

    #src/csr_matrix.cpp
    src/csr_matrix.cpp
    #src/lapjv.cpp
    src/lapjv.cpp
    cuda/gpu_matching.cu
    #src/evaluation.cpp
    src/evaluation.cpp
)

add_library(scgm_lib STATIC ${LIB_SOURCES})
target_include_directories(scgm_lib PUBLIC ${INCLUDE_DIR})
target_link_libraries(scgm_lib PUBLIC CUDA::cudart)
if(NVML_INCLUDE_DIR AND NVML_LIBRARY)
    target_include_directories(scgm_lib PRIVATE ${NVML_INCLUDE_DIR})
    target_link_libraries(scgm_lib PUBLIC ${NVML_LIBRARY})
    target_compile_definitions(scgm_lib PUBLIC SCGM_HAS_NVML=1)
    message(STATUS "NVML per-process memory sampling enabled")
else()
    message(STATUS "NVML not found; per-process GPU peak memory will be unavailable")
endif()
if(SCGM_ENABLE_WARNINGS)
    target_compile_options(scgm_lib PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
endif()
set_target_properties(scgm_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

#add_executable(scgm_cuda src/main.cpp)
add_executable(scgm_cuda apps/main.cpp)
target_link_libraries(scgm_cuda PRIVATE scgm_lib)

#add_executable(toy_scgm examples/toy_scgm.cpp)

add_executable(toy_scgm apps/toy_scgm.cpp)
target_link_libraries(toy_scgm PRIVATE scgm_lib)

if(SCGM_BUILD_TESTS)
    enable_testing()

    set(TEST_SOURCES
        tests/test_features.cpp
        tests/test_permutation.cpp
        tests/test_csr.cpp
        tests/test_topk.cpp
        tests/test_lapjv.cpp
        tests/test_sparse_sinkhorn.cpp
        tests/test_cpu_gpu_parity.cpp
        tests/test_toy_scgm.cpp
    )

    foreach(test_file ${TEST_SOURCES})
        get_filename_component(test_name ${test_file} NAME_WE)
        add_executable(${test_name} ${test_file})
        target_link_libraries(${test_name} PRIVATE scgm_lib)
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")