#include "device_memory_sampler.hpp"

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#ifdef SCGM_HAS_NVML
#include <nvml.h>
#include <unistd.h>
#endif

struct DeviceMemorySampler::Impl {
    std::atomic<bool> running{false};
    std::atomic<size_t> peak_bytes{0};
    std::thread worker;
    bool initialized = false;
#ifdef SCGM_HAS_NVML
    nvmlDevice_t device{};
    unsigned int pid = 0;

    void sample_once() {
        unsigned int count = 0;
        nvmlReturn_t status = nvmlDeviceGetComputeRunningProcesses_v3(device, &count, nullptr);
        if (status != NVML_SUCCESS && status != NVML_ERROR_INSUFFICIENT_SIZE) return;
        if (count == 0) return;
        std::vector<nvmlProcessInfo_t> processes(count);
        status = nvmlDeviceGetComputeRunningProcesses_v3(device, &count, processes.data());
        if (status != NVML_SUCCESS) return;
        for (unsigned int index = 0; index < count; ++index) {
            const auto& process = processes[index];
            if (process.pid != pid || process.usedGpuMemory == static_cast<unsigned long long>(NVML_VALUE_NOT_AVAILABLE)) continue;
            const size_t bytes = static_cast<size_t>(process.usedGpuMemory);
            size_t observed = peak_bytes.load(std::memory_order_relaxed);
            while (bytes > observed && !peak_bytes.compare_exchange_weak(
                       observed, bytes, std::memory_order_relaxed)) {}
        }
    }
#endif
};

DeviceMemorySampler::DeviceMemorySampler(int device_index) : impl_(new Impl) {
#ifdef SCGM_HAS_NVML
    if (device_index < 0 || nvmlInit_v2() != NVML_SUCCESS) return;
    if (nvmlDeviceGetHandleByIndex_v2(static_cast<unsigned int>(device_index),
                                      &impl_->device) != NVML_SUCCESS) return;
    impl_->pid = static_cast<unsigned int>(getpid());
    impl_->initialized = true;
    impl_->sample_once();
    impl_->running.store(true, std::memory_order_relaxed);
    impl_->worker = std::thread([state = impl_.get()] {
        while (state->running.load(std::memory_order_relaxed)) {
            state->sample_once();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        state->sample_once();
    });
#else
    (void)device_index;
#endif
}

DeviceMemorySampler::~DeviceMemorySampler() {
    stop();
}

void DeviceMemorySampler::stop() {
    if (!impl_) return;
#ifdef SCGM_HAS_NVML
    if (impl_->worker.joinable()) {
        impl_->running.store(false, std::memory_order_relaxed);
        impl_->worker.join();
    }
#endif
}

bool DeviceMemorySampler::available() const {
    return impl_ && impl_->initialized && impl_->peak_bytes.load(std::memory_order_relaxed) > 0;
}

size_t DeviceMemorySampler::peak_process_bytes() const {
    return available() ? impl_->peak_bytes.load(std::memory_order_relaxed) : 0;
}
