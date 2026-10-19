#ifndef SCGM_GPU_TOPK_HPP
#define SCGM_GPU_TOPK_HPP

#include "common.hpp"
#include <cstddef>
#include <memory>
#include <vector>

enum class GPUSelector { SerialHeap, Parallel };
enum class GPUFallbackReason {
    None, UnsupportedK, DeviceAllocationFailure, ScratchBudgetTooSmall
};

struct GPUTopKOptions {
    // Global device allocations for scoring batches and selection only.
    // Resident features/index, host input/output, and runtime overhead are excluded.
    std::size_t scratch_budget_bytes = 64u * 1024u * 1024u;
    std::size_t batch_candidates = 65536;
    // Explicit opt-in for the feature API; default SerialHeap retains its legacy path.
    bool force_streaming_serial = false;
};

struct GPUSelectionInfo {
    GPUSelector requested = GPUSelector::SerialHeap;
    GPUSelector used = GPUSelector::SerialHeap;
    GPUFallbackReason fallback = GPUFallbackReason::None;
    std::size_t scratch_peak_bytes = 0;
    std::size_t batch_capacity = 0;
    std::size_t batches = 0;
    std::size_t shared_bytes_per_block = 0;
    double selection_ms = 0.0;
    double transfer_ms = 0.0;
    // Feature API wall time includes allocations/transfers/cleanup; indexed
    // graph candidates also include retrieval, but not index setup or features.
    // Score-only microbenchmarks measure their full call externally.
    double candidate_stage_wall_ms = 0.0;
};

// Offsets start at zero, are nondecreasing, and end at both array lengths.
// Scores are finite and target IDs are in range. IDs are unique within each row:
// producers establish this invariant and tests verify it; selection does not deduplicate.
struct GPUScoredPools {
    std::vector<std::size_t> offsets{0};
    std::vector<int> target_ids;
    std::vector<float> scores;
};
struct GPUSelectedPools {
    std::vector<std::size_t> offsets{0};
    std::vector<int> target_ids;
    std::vector<float> scores;
};
inline constexpr int GPU_TOPK_CHUNK_SIZE = 256;
inline constexpr int GPU_TOPK_PARALLEL_MAX_K = 256;
struct GPUChunkTopK {
    std::vector<std::size_t> source_chunk_offsets{0};
    GPUSelectedPools chunks;
};

// Shared staging format. target == -1 is internal padding, never a public pool ID.
struct GPUScoredCandidate { float score; int target; };

// A reusable, bounded device workspace. Process one source pool at a time.
// begin_pool -> fill input()/push_host -> consume (repeat) -> finish_pool.
// Producers write directly into input(), avoiding a second scoring-batch allocation.
// consume and finish_pool complete on the supplied stream before returning.
// Parallel K is 1..256; unsupported K and insufficient parallel workspace use
// bounded serial selection. If even serial cannot fit, fail explicitly.
class GPUStreamingTopK {
public:
    GPUStreamingTopK(int num_targets, int k, const GPUContext& context,
                    GPUSelector selector = GPUSelector::Parallel,
                    const GPUTopKOptions& options = {});
    ~GPUStreamingTopK();
    GPUStreamingTopK(const GPUStreamingTopK&) = delete;
    GPUStreamingTopK& operator=(const GPUStreamingTopK&) = delete;
    void begin_pool();
    GPUScoredCandidate* input();
    std::size_t capacity() const;
    void consume(std::size_t count);
    void push_host(const GPUScoredCandidate* candidates, std::size_t count);
    std::vector<GPUScoredCandidate> finish_pool();
    // Finish into caller-owned device arrays on this selector's GPU.
    //
    // output_capacity must be at least the configured K.
    // Writes K slots; unused slots contain target=-1 and score=0.
    // *device_count receives the number of valid entries, excluding padding.
    //
    // All three output pointers must reference separate writable allocations
    // on the selector's device and must not alias its workspace.
    //
    // Completes the selector stream before returning.
    // No candidate data or count is downloaded to the host.
    void finish_pool_to_device(
        int* device_target_ids,
        float* device_scores,
        std::size_t output_capacity,
        std::size_t* device_count);
    
    const GPUSelectionInfo& info() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Diagnostic chunk results still include all partial lists on the host.
GPUChunkTopK gpu_select_chunk_topk(const GPUScoredPools& pools, int num_targets,
                                 int k, const GPUContext& context);

// Higher score first; exact ties use lower target ID. Return min(k,row size).
// Empty rows stay empty. Host input is streamed; no full device pool is allocated.
GPUSelectedPools gpu_select_topk(const GPUScoredPools& pools, int num_targets,
    int k, const GPUContext& context,
    GPUSelector selector = GPUSelector::SerialHeap,
    GPUSelectionInfo* info = nullptr, const GPUTopKOptions& options = {});
#endif
