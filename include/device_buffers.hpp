
#ifndef SCGM_DEVICE_BUFFERS_HPP
#define SCGM_DEVICE_BUFFERS_HPP

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace scgm_device_detail {

inline void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

// Cleanup cannot throw. Report failures without hiding an active exception.
inline void report_cleanup_error(
    cudaError_t status, const char* operation) noexcept
{
    if (status != cudaSuccess) {
        std::fprintf(
            stderr, "DeviceBuffer cleanup: %s: %s\n",
            operation, cudaGetErrorString(status));
    }
}

// Temporarily select the allocation's device and restore the caller's device.
class ScopedDevice {
public:
    explicit ScopedDevice(int device_id) {
        check_cuda(cudaGetDevice(&previous_), "cudaGetDevice");

        if (previous_ != device_id) {
            check_cuda(cudaSetDevice(device_id), "cudaSetDevice");
            changed_ = true;
        }
    }

    ~ScopedDevice() noexcept {
        if (changed_) {
            report_cleanup_error(
                cudaSetDevice(previous_), "restore CUDA device");
        }
    }

    ScopedDevice(const ScopedDevice&) = delete;
    ScopedDevice& operator=(const ScopedDevice&) = delete;

private:
    int previous_ = -1;
    bool changed_ = false;
};

} // namespace scgm_device_detail

template <typename T>
class DeviceBuffer {
    static_assert(
        std::is_trivially_copyable<T>::value,
        "DeviceBuffer requires a trivially copyable element type");
    static_assert(
        !std::is_const<T>::value && !std::is_volatile<T>::value,
        "DeviceBuffer element type must be mutable and non-volatile");

public:
    // Owns one CUDA allocation. Moving transfers ownership.
    // Borrowed pointers never release this allocation.
    // Operations using this buffer must finish before its owner is destroyed.
    DeviceBuffer() noexcept = default;

    explicit DeviceBuffer(std::size_t count, int device_id)
        : device_id_(device_id)
    {
        if (device_id < 0) {
            throw std::invalid_argument(
                "DeviceBuffer: device ID must be nonnegative");
        }

        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::overflow_error(
                "DeviceBuffer: allocation size overflow");
        }

        // Empty buffers are valid and do not allocate CUDA memory.
        if (count == 0) {
            return;
        }

        scgm_device_detail::ScopedDevice device(device_id);

        void* allocation = nullptr;
        scgm_device_detail::check_cuda(
            cudaMalloc(&allocation, count * sizeof(T)),
            "DeviceBuffer cudaMalloc");

        pointer_ = static_cast<T*>(allocation);
        count_ = count;
    }

    ~DeviceBuffer() noexcept {
        reset();
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept {
        swap(other);
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            swap(other);
        }
        return *this;
    }

    void swap(DeviceBuffer& other) noexcept {
        using std::swap;
        swap(pointer_, other.pointer_);
        swap(count_, other.count_);
        swap(device_id_, other.device_id_);
    }

    T* data() noexcept {
        return pointer_;
    }

    const T* data() const noexcept {
        return pointer_;
    }

    std::size_t size() const noexcept {
        return count_;
    }

    std::size_t bytes() const noexcept {
        return count_ * sizeof(T);
    }

    bool empty() const noexcept {
        return count_ == 0;
    }

    int device_id() const noexcept {
        return device_id_;
    }

    // Blocking host-to-device copy on the supplied stream.
    // A nondefault stream must belong to device_id().
    void upload(
        const T* source,
        std::size_t count,
        cudaStream_t stream = nullptr,
        std::size_t offset = 0)
    {
        check_range(count, offset);

        if (count == 0) {
            return;
        }
        if (source == nullptr) {
            throw std::invalid_argument(
                "DeviceBuffer upload: null host source");
        }

        scgm_device_detail::ScopedDevice device(device_id_);

        scgm_device_detail::check_cuda(
            cudaMemcpyAsync(
                pointer_ + offset,
                source,
                count * sizeof(T),
                cudaMemcpyHostToDevice,
                stream),
            "DeviceBuffer upload");

        scgm_device_detail::check_cuda(
            cudaStreamSynchronize(stream),
            "DeviceBuffer upload synchronization");
    }

    // Blocking device-to-host copy on the supplied stream.
    // Use the producer stream, or establish an explicit dependency first.
    void download(
        T* destination,
        std::size_t count,
        cudaStream_t stream = nullptr,
        std::size_t offset = 0) const
    {
        check_range(count, offset);

        if (count == 0) {
            return;
        }
        if (destination == nullptr) {
            throw std::invalid_argument(
                "DeviceBuffer download: null host destination");
        }

        scgm_device_detail::ScopedDevice device(device_id_);

        scgm_device_detail::check_cuda(
            cudaMemcpyAsync(
                destination,
                pointer_ + offset,
                count * sizeof(T),
                cudaMemcpyDeviceToHost,
                stream),
            "DeviceBuffer download");

        scgm_device_detail::check_cuda(
            cudaStreamSynchronize(stream),
            "DeviceBuffer download synchronization");
    }

    // Release ownership. Cleanup errors are reported and never thrown.
    // Complete operations using this buffer before calling reset().
    void reset() noexcept {
        if (pointer_ != nullptr) {
            int previous = -1;
            cudaError_t status = cudaGetDevice(&previous);

            scgm_device_detail::report_cleanup_error(
                status, "cudaGetDevice");

            if (status == cudaSuccess) {
                const bool changed = previous != device_id_;

                if (changed) {
                    status = cudaSetDevice(device_id_);
                    scgm_device_detail::report_cleanup_error(
                        status, "select owning device");
                }

                if (status == cudaSuccess) {
                    scgm_device_detail::report_cleanup_error(
                        cudaFree(pointer_), "cudaFree");
                }

                if (changed) {
                    scgm_device_detail::report_cleanup_error(
                        cudaSetDevice(previous), "restore CUDA device");
                }
            }
        }

        pointer_ = nullptr;
        count_ = 0;
        device_id_ = -1;
    }

private:
    void check_range(std::size_t count, std::size_t offset) const {
        // Subtraction avoids overflow from offset + count.
        if (offset > count_ || count > count_ - offset) {
            throw std::out_of_range(
                "DeviceBuffer: copy exceeds allocation");
        }
    }

    T* pointer_ = nullptr;
    std::size_t count_ = 0;
    int device_id_ = -1;
};

#endif // SCGM_DEVICE_BUFFERS_HPP