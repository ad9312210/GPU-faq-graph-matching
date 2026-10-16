//device index ownershop and build/download interfaces

/* - Defining device ownership
class GPU structural index:
    configuration
    target count
    occupied bucket count
    owning device ID

    device bucket keys
    device bucket offsets
    device grouped target IDs

    constructor:
        initialize pointers to null

    destructor:
        release owned device memory on correct device

    disable copying
    support moving without double-free



*/

#ifndef GPU_STRUCTURAL_INDEX_HPP
#define GPU_STRUCTURAL_INDEX_HPP

#include "structural_index.hpp"

class GPUStructuralIndex {
public:
    StructuralIndexConfig config;
    std::size_t target_count = 0;
    std::size_t bucket_count = 0;
    int device_id = -1;

    //device pointers
    StructuralBucketKey* d_bucket_keys = nullptr;
    std::size_t* d_bucket_offsets = nullptr;
    int* d_target_ids = nullptr;

    GPUStructuralIndex() = default;
    ~GPUStructuralIndex();

    //disable copying
    GPUStructuralIndex(const GPUStructuralIndex&) = delete;
    GPUStructuralIndex& operator=(const GPUStructuralIndex&) = delete;
    //support moving without double-free
    // GPUStructuralIndex(GPUStructuralIndex&&) = default;
    // GPUStructuralIndex& operator=(GPUStructuralIndex&&) = default;
    GPUStructuralIndex(GPUStructuralIndex&& other) noexcept;
GPUStructuralIndex& operator=(GPUStructuralIndex&& other) noexcept;

private:
    void release() noexcept;
};

// Input must be device-accessible raw NodeFeatures belonging to context.device_id.
// Caller retains ownership of input and keeps it valid until this call returns.
//
// Construction synchronizes context.stream before returning.
// Output arrays remain on the owning GPU.
// Throws on invalid inputs or CUDA failures.
GPUStructuralIndex build_gpu_structural_index(
    const NodeFeatures* d_raw_targets,
    std::size_t target_count,
    const StructuralIndexConfig& config,
    const GPUContext& context
);

// Explicit download for testing/diagnostics.
// Does not release or modify the GPU index.
StructuralIndex download_structural_index(
    const GPUStructuralIndex& index
);

//One source's retrieval decision
//Actual target IDs remain in the GPU target index

struct GPUStructuralQuery{
    StructuralBucketKey source_key;

    std::size_t desired_pool_size = 0;
    std::size_t bucket_pool_size = 0;
    std::size_t effective_pool_size = 0;

    int final_radius = -1;
    bool used_exhaustive_fallback = false;
};

//Owns one device query record per source
//The target index must remain alive separately

class GPUStructuralQueries {
public:
    std::size_t source_count = 0;
    int device_id = -1;
    GPUStructuralQuery* d_queries = nullptr;

    GPUStructuralQueries() = default;
    ~GPUStructuralQueries();

    GPUStructuralQueries(const GPUStructuralQueries&) = delete;
    GPUStructuralQueries& operator=(const GPUStructuralQueries&) = delete;

    GPUStructuralQueries(GPUStructuralQueries&& other) noexcept;

    GPUStructuralQueries& operator=(
        GPUStructuralQueries&& other
    ) noexcept;

private:
    void release() noexcept;
};

// Computes retrieval decisions on GPU.
// Input must be raw device features on context.device_id.
// Synchronizes the supplied stream before returning.
//
// Queries describe bucket ranges rather than duplicating every target ID.
// Index must be an unmodified, completed GPU index.
GPUStructuralQueries retrieve_gpu_structural_candidates(
    const GPUStructuralIndex& index,
    const NodeFeatures* d_raw_sources,
    std::size_t source_count,
    int k,
    const GPUContext& context
);

// Downloads decisions only, for diagnostics/tests.
std::vector<GPUStructuralQuery> download_structural_queries(
    const GPUStructuralQueries& queries
);
#endif