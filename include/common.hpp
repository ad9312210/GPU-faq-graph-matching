#ifndef SCGM_COMMON_HPP
#define SCGM_COMMON_HPP

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <functional>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                         \
    do {                                                                          \
        cudaError_t err = (call);                                                 \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s (error code %d: %s)\n",     \
                    __FILE__, __LINE__, #call, (int)err,                          \
                    cudaGetErrorString(err));                                     \
            throw std::runtime_error(                                             \
                std::string("CUDA error: ") + cudaGetErrorString(err) +           \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));              \
        }                                                                         \
    } while (0)

static constexpr float SCGM_INF = 1e30f;
static constexpr float SCGM_EPS = 1e-12f;
static constexpr double SCGM_TOL = 1e-6;

struct Timer {
    std::chrono::high_resolution_clock::time_point start_time;
    std::string label;

    Timer(const std::string& lbl) : label(lbl) {
        start_time = std::chrono::high_resolution_clock::now();
    }

    double elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_time).count();
    }

    void report() const {
        std::cout << "  " << label << ": " << std::fixed << std::setprecision(3)
                  << elapsed_ms() << " ms" << std::endl;
    }
};

struct GPUContext {
    int device_id;
    cudaStream_t stream;

    GPUContext() : device_id(0), stream(nullptr) {}
};

// Explicit multi-device requests are strict: never silently drop a device.
inline void validate_gpu_device_ids(const std::vector<int>& ids) {
    if (ids.empty()) throw std::invalid_argument("GPU device list must not be empty");
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    std::set<int> unique;
    for (int id : ids)
        if (id < 0 || id >= count || !unique.insert(id).second)
            throw std::invalid_argument("GPU IDs must be available, nonnegative, and unique");
}

// Thread-local execution owner. Destroy device buffers before this owner.
class GPUContextOwner {
    GPUContext context_;
public:
    explicit GPUContextOwner(int device) {
        CUDA_CHECK(cudaSetDevice(device));
        context_.device_id = device;
        CUDA_CHECK(cudaStreamCreate(&context_.stream));
    }
    GPUContextOwner(const GPUContextOwner&) = delete;
    GPUContextOwner& operator=(const GPUContextOwner&) = delete;
    ~GPUContextOwner() noexcept {
        if (!context_.stream) return;
        int previous = -1;
        const bool restore = cudaGetDevice(&previous) == cudaSuccess;
        if (cudaSetDevice(context_.device_id) == cudaSuccess) {
            (void)cudaStreamSynchronize(context_.stream);
            (void)cudaStreamDestroy(context_.stream);
        }
        if (restore && previous != context_.device_id) (void)cudaSetDevice(previous);
    }
    const GPUContext& get() const noexcept { return context_; }
};

inline std::vector<GPUContext> initialize_gpus(const std::vector<int>& device_ids = {0}) {
    std::vector<GPUContext> contexts;
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "WARNING: No CUDA devices found. GPU features disabled." << std::endl;
        return contexts;
    }
    for (int did : device_ids) {
        if (did >= device_count) {
            std::cerr << "WARNING: Device " << did << " not available (have "
                      << device_count << " devices)." << std::endl;
            continue;
        }
        GPUContext ctx;
        ctx.device_id = did;
        CUDA_CHECK(cudaSetDevice(did));
        CUDA_CHECK(cudaStreamCreate(&ctx.stream));
        contexts.push_back(ctx);
    }
    return contexts;
}

inline void cleanup_gpus(std::vector<GPUContext>& contexts) {
    for (auto& ctx : contexts) {
        if (ctx.stream) {
            cudaSetDevice(ctx.device_id);
            cudaStreamDestroy(ctx.stream);
            ctx.stream = nullptr;
        }
    }
    contexts.clear();
}

#endif // SCGM_COMMON_HPP
