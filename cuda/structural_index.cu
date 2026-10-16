#include "gpu_structural_index.hpp"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

using namespace std;

namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw runtime_error(
            string(operation) + ": " + cudaGetErrorString(status));
    }
}

// Restore the caller's current device when leaving this scope.
class DeviceScope {
public:
    explicit DeviceScope(int device) {
        check_cuda(cudaGetDevice(&previous_), "cudaGetDevice");
        check_cuda(cudaSetDevice(device), "cudaSetDevice");
    }

    ~DeviceScope() {
        cudaSetDevice(previous_);
    }

    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;

private:
    int previous_ = 0;
};

template<class T>
void allocate_device(T*& pointer, size_t count) {
    if (count == 0) {
        pointer = nullptr;
        return;
    }

    if (count > numeric_limits<size_t>::max() / sizeof(T)) {
        throw overflow_error("device allocation size overflow");
    }

    check_cuda(
        cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)),
        "cudaMalloc");
}

struct TargetRecord {
    int64_t degree_bin;
    int64_t neighbor_bin;
    int target_id;
};

struct RecordLess {
    __host__ __device__
    bool operator()(const TargetRecord& a, const TargetRecord& b) const {
        if (a.degree_bin != b.degree_bin)
            return a.degree_bin < b.degree_bin;

        if (a.neighbor_bin != b.neighbor_bin)
            return a.neighbor_bin < b.neighbor_bin;

        return a.target_id < b.target_id;
    }
};

__device__ bool same_key(
    const TargetRecord& a,
    const TargetRecord& b
) {
    return a.degree_bin == b.degree_bin
        && a.neighbor_bin == b.neighbor_bin;
}

__global__ void quantize_targets(
    const NodeFeatures* features,
    int count,
    double degree_width,
    double neighbor_width,
    TargetRecord* records,
    int* error_flags
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= static_cast<size_t>(count)) return;

    const double degree = static_cast<double>(features[index].degree);
    const double mean =
        static_cast<double>(features[index].avg_neighbor_degree);

    // Bit 0: invalid raw feature.
    // Bit 1: quantized coordinate overflow.
    if (!isfinite(degree) || !isfinite(mean) ||
        degree < 0.0 || mean < 0.0) {
        atomicOr(error_flags, 1);
        return;
    }

    const double x = floor(log1p(degree) / degree_width);
    const double y = floor(log1p(mean) / neighbor_width);

    // int64 values must be strictly below 2^63.
    const double upper_bound = 9223372036854775808.0;

    if (!isfinite(x) || !isfinite(y) ||
        x < 0.0 || y < 0.0 ||
        x >= upper_bound || y >= upper_bound) {
        atomicOr(error_flags, 2);
        return;
    }

    records[index] = {
        static_cast<int64_t>(x),
        static_cast<int64_t>(y),
        static_cast<int>(index)
    };
}

__global__ void mark_boundaries(
    const TargetRecord* records,
    int count,
    int* boundaries
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= static_cast<size_t>(count)) return;

    boundaries[index] =
        index == 0 || !same_key(records[index], records[index - 1]);
}

__global__ void write_index(
    const TargetRecord* records,
    const int* boundaries,
    const int* bucket_numbers,
    int count,
    StructuralBucketKey* keys,
    size_t* offsets,
    int* target_ids
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= static_cast<size_t>(count)) return;

    target_ids[index] = records[index].target_id;

    if (boundaries[index]) {
        const int bucket = bucket_numbers[index] - 1;

        keys[bucket].degree_bin = records[index].degree_bin;
        keys[bucket].neighbor_bin = records[index].neighbor_bin;
        offsets[bucket] = index;
    }
}

__device__ int64_t bucket_distance(
    int64_t source_x,
    int64_t source_y,
    int64_t target_x,
    int64_t target_y
) {
    // Quantized coordinates are nonnegative.
    // Subtract smaller from larger to avoid signed overflow.
    const int64_t dx = source_x >= target_x
        ? source_x - target_x
        : target_x - source_x;

    const int64_t dy = source_y >= target_y
        ? source_y - target_y
        : target_y - source_y;

    return dx > dy ? dx : dy;
}

__global__ void plan_structural_queries(
    const TargetRecord* source_records,
    size_t source_count,
    const StructuralBucketKey* bucket_keys,
    const size_t* bucket_offsets,
    size_t bucket_count,
    size_t target_count,
    size_t desired_pool_size,
    int minimum_radius,
    int maximum_radius,
    GPUStructuralQuery* queries
) {
    const size_t source =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (source >= source_count) return;

    const TargetRecord record = source_records[source];

    GPUStructuralQuery query{};
    query.source_key.degree_bin = record.degree_bin;
    query.source_key.neighbor_bin = record.neighbor_bin;
    query.desired_pool_size = desired_pool_size;

    size_t pool_size = 0;

    // Correctness-first reference:
    // scan occupied buckets for each complete shell.
    for (int64_t radius = 0;
         radius <= static_cast<int64_t>(maximum_radius);
         ++radius) {
        for (size_t bucket = 0; bucket < bucket_count; ++bucket) {
            const auto key = bucket_keys[bucket];

            const int64_t distance = bucket_distance(
                record.degree_bin,
                record.neighbor_bin,
                key.degree_bin,
                key.neighbor_bin);

            if (distance == radius) {
                pool_size +=
                    bucket_offsets[bucket + 1] - bucket_offsets[bucket];
            }
        }

        query.final_radius = static_cast<int>(radius);

        // Decide only after counting the complete shell.
        if (radius >= minimum_radius &&
            pool_size >= desired_pool_size) {
            break;
        }
    }

    query.bucket_pool_size = pool_size;
    query.used_exhaustive_fallback =
        pool_size < desired_pool_size;

    query.effective_pool_size =
        query.used_exhaustive_fallback
            ? target_count
            : pool_size;

    queries[source] = query;
}


} // namespace

void GPUStructuralIndex::release() noexcept {
    if (device_id >= 0) {
        int previous = -1;
        const bool restore =
            cudaGetDevice(&previous) == cudaSuccess;

        if (cudaSetDevice(device_id) == cudaSuccess) {
            if (d_bucket_keys) cudaFree(d_bucket_keys);
            if (d_bucket_offsets) cudaFree(d_bucket_offsets);
            if (d_target_ids) cudaFree(d_target_ids);
        }

        if (restore) cudaSetDevice(previous);
    }

    d_bucket_keys = nullptr;
    d_bucket_offsets = nullptr;
    d_target_ids = nullptr;
    target_count = 0;
    bucket_count = 0;
    device_id = -1;
}

GPUStructuralIndex::~GPUStructuralIndex() {
    release();
}

GPUStructuralIndex::GPUStructuralIndex(
    GPUStructuralIndex&& other
) noexcept {
    *this = std::move(other);
}

GPUStructuralIndex& GPUStructuralIndex::operator=(
    GPUStructuralIndex&& other
) noexcept {
    if (this != &other) {
        release();

        config = other.config;
        target_count = exchange(other.target_count, 0);
        bucket_count = exchange(other.bucket_count, 0);
        device_id = exchange(other.device_id, -1);

        d_bucket_keys = exchange(other.d_bucket_keys, nullptr);
        d_bucket_offsets = exchange(other.d_bucket_offsets, nullptr);
        d_target_ids = exchange(other.d_target_ids, nullptr);
    }

    return *this;
}

GPUStructuralIndex build_gpu_structural_index(
    const NodeFeatures* d_raw_targets,
    size_t target_count,
    const StructuralIndexConfig& config,
    const GPUContext& context
) {
    validate_structural_index_config(config);

    if (target_count >
        static_cast<size_t>(numeric_limits<int>::max())) {
        throw overflow_error("target count exceeds int range");
    }

    if (target_count > 0 && d_raw_targets == nullptr) {
        throw invalid_argument("null target feature pointer");
    }

    DeviceScope device_scope(context.device_id);

    GPUStructuralIndex result;
    result.config = config;
    result.target_count = target_count;
    result.device_id = context.device_id;

    if (target_count == 0) {
        allocate_device(result.d_bucket_offsets, 1);

        check_cuda(
            cudaMemsetAsync(
                result.d_bucket_offsets, 0,
                sizeof(size_t), context.stream),
            "initialize empty offsets");

        check_cuda(
            cudaStreamSynchronize(context.stream),
            "finish empty index");

        return result;
    }

    // Ordinary device memory on this device is the input contract.
    cudaPointerAttributes attributes{};
    check_cuda(
        cudaPointerGetAttributes(&attributes, d_raw_targets),
        "inspect target feature pointer");

    if (attributes.type != cudaMemoryTypeDevice ||
        attributes.device != context.device_id) {
        throw invalid_argument(
            "target features must be device memory on the selected GPU");
    }

    const int count = static_cast<int>(target_count);
    constexpr unsigned threads = 256;
    const unsigned blocks =
        static_cast<unsigned>((target_count + threads - 1) / threads);

    thrust::device_vector<TargetRecord> records(target_count);
    thrust::device_vector<int> boundaries(target_count);
    thrust::device_vector<int> bucket_numbers(target_count);
    thrust::device_vector<int> error_flags(1, 0);

    auto* d_records = thrust::raw_pointer_cast(records.data());
    auto* d_boundaries = thrust::raw_pointer_cast(boundaries.data());
    auto* d_numbers = thrust::raw_pointer_cast(bucket_numbers.data());
    auto* d_errors = thrust::raw_pointer_cast(error_flags.data());

    quantize_targets<<<blocks, threads, 0, context.stream>>>(
        d_raw_targets, count,
        config.degree_bin_width,
        config.neighbor_bin_width,
        d_records, d_errors);

    check_cuda(cudaGetLastError(), "launch quantization");

    int errors = 0;
    check_cuda(
        cudaMemcpyAsync(
            &errors, d_errors, sizeof(int),
            cudaMemcpyDeviceToHost, context.stream),
        "read quantization flags");

    check_cuda(
        cudaStreamSynchronize(context.stream),
        "finish quantization");

    if (errors & 1)
        throw invalid_argument("negative or nonfinite indexed feature");

    if (errors & 2)
        throw overflow_error("quantized coordinate exceeds int64 range");

    auto policy = thrust::cuda::par.on(context.stream);

    thrust::sort(
        policy, records.begin(), records.end(), RecordLess{});

    mark_boundaries<<<blocks, threads, 0, context.stream>>>(
        d_records, count, d_boundaries);

    check_cuda(cudaGetLastError(), "launch boundary detection");

    thrust::inclusive_scan(
        policy,
        boundaries.begin(),
        boundaries.end(),
        bucket_numbers.begin());

    int bucket_count = 0;
    check_cuda(
        cudaMemcpyAsync(
            &bucket_count, d_numbers + count - 1, sizeof(int),
            cudaMemcpyDeviceToHost, context.stream),
        "read bucket count");

    check_cuda(
        cudaStreamSynchronize(context.stream),
        "finish bucket counting");

    result.bucket_count = static_cast<size_t>(bucket_count);

    allocate_device(result.d_bucket_keys, result.bucket_count);
    allocate_device(result.d_bucket_offsets, result.bucket_count + 1);
    allocate_device(result.d_target_ids, target_count);

    write_index<<<blocks, threads, 0, context.stream>>>(
        d_records, d_boundaries, d_numbers, count,
        result.d_bucket_keys,
        result.d_bucket_offsets,
        result.d_target_ids);

    check_cuda(cudaGetLastError(), "launch index output");

    check_cuda(
        cudaMemcpyAsync(
            result.d_bucket_offsets + result.bucket_count,
            &target_count, sizeof(size_t),
            cudaMemcpyHostToDevice, context.stream),
        "write final offset");

    check_cuda(
        cudaStreamSynchronize(context.stream),
        "finish index construction");

    return result;
}

StructuralIndex download_structural_index(
    const GPUStructuralIndex& index
) {
    if (index.device_id < 0 || index.d_bucket_offsets == nullptr) {
        throw invalid_argument("index is unbuilt or moved-from");
    }

    DeviceScope device_scope(index.device_id);

    StructuralIndex result;
    result.config = index.config;
    result.target_count = index.target_count;
    result.bucket_keys.resize(index.bucket_count);
    result.bucket_offsets.resize(index.bucket_count + 1);
    result.target_ids.resize(index.target_count);

    if (index.bucket_count > 0) {
        check_cuda(
            cudaMemcpy(
                result.bucket_keys.data(),
                index.d_bucket_keys,
                index.bucket_count * sizeof(StructuralBucketKey),
                cudaMemcpyDeviceToHost),
            "download bucket keys");
    }

    check_cuda(
        cudaMemcpy(
            result.bucket_offsets.data(),
            index.d_bucket_offsets,
            (index.bucket_count + 1) * sizeof(size_t),
            cudaMemcpyDeviceToHost),
        "download offsets");

    if (index.target_count > 0) {
        check_cuda(
            cudaMemcpy(
                result.target_ids.data(),
                index.d_target_ids,
                index.target_count * sizeof(int),
                cudaMemcpyDeviceToHost),
            "download target IDs");
    }

    return result;
}

void GPUStructuralQueries::release() noexcept {
    if (device_id >= 0 && d_queries != nullptr) {
        int previous = -1;
        const bool restore =
            cudaGetDevice(&previous) == cudaSuccess;

        if (cudaSetDevice(device_id) == cudaSuccess) {
            cudaFree(d_queries);
        }

        if (restore) cudaSetDevice(previous);
    }

    d_queries = nullptr;
    source_count = 0;
    device_id = -1;
}

GPUStructuralQueries::~GPUStructuralQueries() {
    release();
}

GPUStructuralQueries::GPUStructuralQueries(
    GPUStructuralQueries&& other
) noexcept {
    *this = std::move(other);
}

GPUStructuralQueries& GPUStructuralQueries::operator=(
    GPUStructuralQueries&& other
) noexcept {
    if (this != &other) {
        release();

        source_count = exchange(other.source_count, 0);
        device_id = exchange(other.device_id, -1);
        d_queries = exchange(other.d_queries, nullptr);
    }

    return *this;
}

GPUStructuralQueries retrieve_gpu_structural_candidates(
    const GPUStructuralIndex& index,
    const NodeFeatures* d_raw_sources,
    size_t source_count,
    int k,
    const GPUContext& context
) {
    validate_structural_index_config(index.config);

    if (index.device_id < 0 ||
        index.device_id != context.device_id ||
        index.d_bucket_offsets == nullptr) {
        throw invalid_argument(
            "retrieval requires a built index on the selected GPU");
    }

    if (source_count >
        static_cast<size_t>(numeric_limits<int>::max())) {
        throw overflow_error("source count exceeds supported int range");
    }

    if (k <= 0 ||
        (source_count > 0 &&
         static_cast<size_t>(k) > index.target_count)) {
        throw invalid_argument(
            "retrieval requires positive k and enough targets");
    }

    if (source_count > 0 && d_raw_sources == nullptr) {
        throw invalid_argument("null raw source feature pointer");
    }

    DeviceScope device_scope(context.device_id);

    GPUStructuralQueries result;
    result.source_count = source_count;
    result.device_id = context.device_id;

    if (source_count == 0) {
        return result;
    }

    cudaPointerAttributes attributes{};
    check_cuda(
        cudaPointerGetAttributes(&attributes, d_raw_sources),
        "inspect source feature pointer");

    if (attributes.type != cudaMemoryTypeDevice ||
        attributes.device != context.device_id) {
        throw invalid_argument(
            "source features must be device memory on the selected GPU");
    }

    // Saturating multiplication: min(M, multiplier * k).
    const size_t multiplier =
        static_cast<size_t>(index.config.pool_multiplier);

    const size_t count = static_cast<size_t>(k);

    const size_t desired =
        count > index.target_count / multiplier
            ? index.target_count
            : count * multiplier;

    thrust::device_vector<TargetRecord> source_records(source_count);
    thrust::device_vector<int> error_flags(1, 0);

    auto* d_records =
        thrust::raw_pointer_cast(source_records.data());

    auto* d_errors =
        thrust::raw_pointer_cast(error_flags.data());

    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>(
        (source_count + threads - 1) / threads);

    // Reuse the exact same GPU quantization code as target construction.
    quantize_targets<<<blocks, threads, 0, context.stream>>>(
        d_raw_sources,
        static_cast<int>(source_count),
        index.config.degree_bin_width,
        index.config.neighbor_bin_width,
        d_records,
        d_errors);

    check_cuda(cudaGetLastError(), "launch source quantization");

    int errors = 0;

    check_cuda(
        cudaMemcpyAsync(
            &errors,
            d_errors,
            sizeof(int),
            cudaMemcpyDeviceToHost,
            context.stream),
        "read source quantization errors");

    check_cuda(
        cudaStreamSynchronize(context.stream),
        "finish source quantization");

    if (errors & 1) {
        throw invalid_argument(
            "negative or nonfinite indexed source feature");
    }

    if (errors & 2) {
        throw overflow_error(
            "source bucket coordinate exceeds int64 range");
    }

    allocate_device(result.d_queries, source_count);

    plan_structural_queries<<<blocks, threads, 0, context.stream>>>(
        d_records,
        source_count,
        index.d_bucket_keys,
        index.d_bucket_offsets,
        index.bucket_count,
        index.target_count,
        desired,
        index.config.minimum_radius,
        index.config.maximum_radius,
        result.d_queries);

    check_cuda(cudaGetLastError(), "launch structural retrieval");

    check_cuda(
        cudaStreamSynchronize(context.stream),
        "finish structural retrieval");

    return result;
}

vector<GPUStructuralQuery> download_structural_queries(
    const GPUStructuralQueries& queries
) {
    if (queries.device_id < 0) {
        throw invalid_argument("queries are unbuilt or moved-from");
    }

    DeviceScope device_scope(queries.device_id);

    vector<GPUStructuralQuery> result(queries.source_count);

    if (!result.empty()) {
        check_cuda(
            cudaMemcpy(
                result.data(),
                queries.d_queries,
                result.size() * sizeof(GPUStructuralQuery),
                cudaMemcpyDeviceToHost),
            "download structural queries");
    }

    return result;
}