#ifndef SCGM_DEVICE_MEMORY_SAMPLER_HPP
#define SCGM_DEVICE_MEMORY_SAMPLER_HPP

#include <cstddef>
#include <memory>

// Samples this process's CUDA memory through NVML while a GPU pipeline runs.
// Returns no value when NVML is unavailable or cannot identify this process.
class DeviceMemorySampler {
public:
    explicit DeviceMemorySampler(int device_index = 0);
    ~DeviceMemorySampler();
    DeviceMemorySampler(const DeviceMemorySampler&) = delete;
    DeviceMemorySampler& operator=(const DeviceMemorySampler&) = delete;

    void stop();
    bool available() const;
    size_t peak_process_bytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif
