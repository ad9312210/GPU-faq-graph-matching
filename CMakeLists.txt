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
find_package(Threads REQUIRED)
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
    src/candidate_repair.cpp
    src/sparse_sinkhorn.cpp
    src/device_memory_sampler.cpp

    #src/csr_matrix.cpp
    src/csr_matrix.cpp
    #src/lapjv.cpp
    src/lapjv.cpp
    cuda/gpu_matching.cu
    #src/evaluation.cpp
    src/evaluation.cpp

    #src/structural_index.cpp
    src/structural_index.cpp

    cuda/structural_index.cu
    cuda/csr_matrix.cu
    cuda/sparse_sinkhorn.cu
    cuda/gpu_assignment.cu
    cuda/assignment_components.cu
    cuda/assignment_scheduler.cu
    cuda/multi_gpu_matching.cu
)

add_library(scgm_lib STATIC ${LIB_SOURCES})

# Cooperative selection of pre-scored candidate chunks.
target_sources(scgm_lib PRIVATE cuda/topk.cu cuda/gpu_candidate_scoring.cu)

target_include_directories(scgm_lib PUBLIC ${INCLUDE_DIR})
target_link_libraries(scgm_lib PUBLIC CUDA::cudart Threads::Threads)
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
    add_executable(test_multi_gpu_matching tests/test_multi_gpu_matching.cpp)
    target_link_libraries(test_multi_gpu_matching PRIVATE scgm_lib)
    add_test(NAME test_multi_gpu_matching COMMAND test_multi_gpu_matching)
    set_tests_properties(test_multi_gpu_matching PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(test_assignment_scheduler tests/test_assignment_scheduler.cpp)
    target_link_libraries(test_assignment_scheduler PRIVATE scgm_lib)
    add_test(NAME test_assignment_scheduler COMMAND test_assignment_scheduler)
    set_tests_properties(test_assignment_scheduler PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(test_assignment_components tests/test_assignment_components.cpp)
    target_link_libraries(test_assignment_components PRIVATE scgm_lib)
    add_test(NAME test_assignment_components COMMAND test_assignment_components)
    set_tests_properties(test_assignment_components PROPERTIES SKIP_RETURN_CODE 77)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME test_correspondence_quality
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_correspondence_quality.py)
        add_test(NAME test_candidate_repair_cli
            COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_candidate_repair_cli.py
                $<TARGET_FILE:scgm_cuda>)
    endif()

    add_executable(test_gpu_sinkhorn tests/test_gpu_sinkhorn.cpp)
    target_link_libraries(test_gpu_sinkhorn PRIVATE scgm_lib)
    add_test(NAME test_gpu_sinkhorn COMMAND test_gpu_sinkhorn)
    set_tests_properties(test_gpu_sinkhorn PROPERTIES SKIP_RETURN_CODE 77)


    add_executable(benchmark_gpu_topk_streaming tests/benchmark_gpu_topk_streaming.cpp)
    target_link_libraries(benchmark_gpu_topk_streaming PRIVATE scgm_lib)

    # Input-driven bridge for independent Python reference validation.
    # Not a standalone CTest: the caller supplies a sparse problem on stdin.
    add_executable(assignment_solver_driver tests/assignment_solver_driver.cpp)
    target_link_libraries(assignment_solver_driver PRIVATE scgm_lib)

    add_executable(test_gpu_device_costs tests/test_gpu_device_costs.cpp)
    target_link_libraries(test_gpu_device_costs PRIVATE scgm_lib)

    add_test(
        NAME test_gpu_device_costs
        COMMAND test_gpu_device_costs
    )

    add_executable(
        test_gpu_resident_pipeline
        tests/test_gpu_resident_pipeline.cpp
    )

    target_link_libraries(
        test_gpu_resident_pipeline
        PRIVATE scgm_lib
    )

    add_test(
        NAME test_gpu_resident_pipeline
        COMMAND test_gpu_resident_pipeline
    )

    set_tests_properties(
        test_gpu_resident_pipeline
        PROPERTIES SKIP_RETURN_CODE 77
    )

    add_executable(test_gpu_assignment tests/test_gpu_assignment.cpp)
    target_link_libraries(test_gpu_assignment PRIVATE scgm_lib)

    add_test(NAME test_gpu_assignment COMMAND test_gpu_assignment)
    set_tests_properties(test_gpu_assignment PROPERTIES SKIP_RETURN_CODE 77)


    set(TEST_SOURCES
        tests/test_candidate_repair.cpp
        tests/test_features.cpp
        tests/test_permutation.cpp
        tests/test_csr.cpp
        tests/test_device_csr.cpp
        tests/test_structural_index.cpp
        tests/test_topk.cpp
        tests/test_lapjv.cpp
        tests/test_sparse_sinkhorn.cpp
        tests/test_cpu_gpu_parity.cpp
        tests/test_toy_scgm.cpp
        tests/test_gpu_structural_index.cpp
        tests/test_gpu_topk_chunks.cpp
        tests/test_gpu_topk_merge.cpp
        tests/test_gpu_topk_streaming.cpp
        tests/test_gpu_topk.cpp
    )

    foreach(test_file ${TEST_SOURCES})
        get_filename_component(test_name ${test_file} NAME_WE)
        add_executable(${test_name} ${test_file})
        target_link_libraries(${test_name} PRIVATE scgm_lib)
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()

    # A missing GPU is a skipped test, not a successful correctness check.
    set_tests_properties(
        test_gpu_topk_chunks
        PROPERTIES SKIP_RETURN_CODE 77
    )

    set_tests_properties(
        test_gpu_topk_merge
        PROPERTIES SKIP_RETURN_CODE 77
    )

    set_tests_properties(test_gpu_topk_streaming test_gpu_topk test_cpu_gpu_parity test_device_csr
        PROPERTIES SKIP_RETURN_CODE 77)

    # Exit 77 means GPU hardware was unavailable, not a successful parity check.
    set_tests_properties(test_gpu_structural_index PROPERTIES SKIP_RETURN_CODE 77)


    set_tests_properties(
        test_gpu_device_costs
        PROPERTIES SKIP_RETURN_CODE 77
    )

endif()

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
