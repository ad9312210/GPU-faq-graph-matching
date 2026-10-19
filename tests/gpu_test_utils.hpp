#ifndef SCGM_GPU_TEST_UTILS_HPP
#define SCGM_GPU_TEST_UTILS_HPP
#include "common.hpp"

// Only known lack-of-device/driver conditions are skips. Unexpected CUDA
// failures are test failures, never evidence of successful parity.
inline int gpu_test_availability() {
    int count = 0;
    const auto error = cudaGetDeviceCount(&count);
    if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver ||
        (error == cudaSuccess && count == 0)) {
        std::cout << "SKIPPED: no usable CUDA GPU\n";
        return 77;
    }
    if (error != cudaSuccess) {
        std::cerr << "GPU availability failed: " << cudaGetErrorString(error) << '\n';
        return 1;
    }
    return 0;
}
struct GPUTestContext {
    GPUContext value;
    GPUTestContext() {
        CUDA_CHECK(cudaSetDevice(value.device_id));
        CUDA_CHECK(cudaStreamCreate(&value.stream));
    }
    ~GPUTestContext() { if (value.stream) cudaStreamDestroy(value.stream); }
    GPUTestContext(const GPUTestContext&) = delete;
    GPUTestContext& operator=(const GPUTestContext&) = delete;
};
inline void gpu_test_require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
#endif
