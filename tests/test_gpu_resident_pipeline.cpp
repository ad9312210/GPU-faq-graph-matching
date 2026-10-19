#include "gpu_matching.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool close(float a, float b) {
    return std::isfinite(a) && std::isfinite(b) &&
        std::abs(a - b) <=
            2e-5f + 2e-4f * std::max(std::abs(a), std::abs(b));
}

void compare_features(
    const std::vector<NodeFeatures>& actual,
    const std::vector<NodeFeatures>& expected)
{
    require(actual.size() == expected.size(), "Feature count mismatch");

    for (std::size_t i = 0; i < actual.size(); ++i) {
        for (int f = 0; f < NUM_FEATURES; ++f) {
            require(
                close(actual[i].as_array(f), expected[i].as_array(f)),
                "Feature parity mismatch");
        }
    }
}

std::vector<NodeFeatures> download_raw(
    const DeviceFeatureSet& features,
    const GPUContext& context)
{
    std::vector<NodeFeatures> result(features.raw.size());
    features.raw.download(result.data(), result.size(), context.stream);
    return result;
}

void require_identical_raw(
    const std::vector<NodeFeatures>& first,
    const std::vector<NodeFeatures>& second)
{
    require(first.size() == second.size(), "Raw feature count changed");

    for (std::size_t i = 0; i < first.size(); ++i) {
        for (int f = 0; f < NUM_FEATURES; ++f) {
            require(
                first[i].as_array(f) == second[i].as_array(f),
                "Normalization modified raw features");
        }
    }
}

void compare_candidates(
    const std::vector<std::vector<Candidate>>& first,
    const std::vector<std::vector<Candidate>>& second)
{
    require(first.size() == second.size(), "Candidate row count mismatch");

    for (std::size_t row = 0; row < first.size(); ++row) {
        require(
            first[row].size() == second[row].size(),
            "Candidate row length mismatch");

        for (std::size_t i = 0; i < first[row].size(); ++i) {
            require(
                first[row][i].target == second[row][i].target,
                "Serial/parallel target mismatch");

            require(
                first[row][i].similarity == second[row][i].similarity,
                "Serial/parallel GPU score mismatch");

            if (i != 0) {
                const auto& previous = first[row][i - 1];
                const auto& current = first[row][i];

                require(
                    previous.similarity > current.similarity ||
                    (previous.similarity == current.similarity &&
                     previous.target < current.target),
                    "Incorrect candidate ordering");
            }
        }
    }
}

void check_buffers(const GPUContext& context) {
    DeviceBuffer<int> empty;
    require(empty.empty() && empty.data() == nullptr,
            "Default buffer is not empty");

    empty.upload(nullptr, 0, context.stream);
    empty.download(nullptr, 0, context.stream);

    DeviceBuffer<int> zero(0, context.device_id);
    require(zero.empty() && zero.data() == nullptr,
            "Zero-size buffer allocated storage");

    const int input[] = {7, 11, 19};
    int output[3] = {};

    DeviceBuffer<int> first(3, context.device_id);
    first.upload(input, 3, context.stream);

    int* original = first.data();
    DeviceBuffer<int> second(std::move(first));

    require(first.data() == nullptr && first.size() == 0,
            "Move construction retained source ownership");
    require(second.data() == original,
            "Move construction changed allocation");

    DeviceBuffer<int> third(8, context.device_id);
    third = std::move(second);

    require(second.data() == nullptr && second.size() == 0,
            "Move assignment retained source ownership");

    third.download(output, 3, context.stream);

    require(output[0] == 7 && output[1] == 11 && output[2] == 19,
            "Moved buffer data changed");

    // A later member fails after an earlier allocation succeeded.
    // Compute Sanitizer's leak check verifies cleanup during unwinding.
    struct PartialOwner {
        DeviceBuffer<int> first;
        DeviceBuffer<int> second;

        explicit PartialOwner(int device)
            : first(32, device),
              second(std::numeric_limits<std::size_t>::max(), device) {}
    };

    bool rejected = false;

    try {
        PartialOwner owner(context.device_id);
    } catch (const std::overflow_error&) {
        rejected = true;
    }

    require(rejected, "Allocation arithmetic overflow was accepted");
}

void check_case(
    const Graph& source,
    const Graph& target,
    int k,
    bool joint,
    bool indexed,
    const GPUContext& context)
{
    GPUTopKOptions options;
    options.batch_candidates = 256;
    options.force_streaming_serial = true;

    StructuralIndexConfig config;

    auto parallel = gpu_compute_device_candidates(
        source, target, k, context, joint,
        GPUSelector::Parallel, options,
        indexed ? &config : nullptr);

    auto serial = gpu_compute_device_candidates(
        source, target, k, context, joint,
        GPUSelector::SerialHeap, options,
        indexed ? &config : nullptr);

    require(
        parallel.residency.scoring_selection_scratch_peak_bytes <=
            options.scratch_budget_bytes,
        "Scoring/selection exceeded scratch budget");

    require(
        parallel.residency.known_h2d_bytes ==
            parallel.residency.adjacency_bytes,
        "Unexpected known feature/candidate upload");

    require(
        parallel.residency.transfer_tracking_complete == !indexed,
        "Incorrect transfer tracking completeness");

    if (k > GPU_TOPK_PARALLEL_MAX_K) {
        require(
            parallel.selection.used == GPUSelector::SerialHeap &&
            parallel.selection.fallback == GPUFallbackReason::UnsupportedK,
            "K fallback was not recorded");
    }

    const auto source_raw_before =
        download_raw(parallel.source_features, context);
    const auto target_raw_before =
        download_raw(parallel.target_features, context);

    compare_features(source_raw_before, compute_features(source));
    compare_features(target_raw_before, compute_features(target));

    // Reapply normalization and verify raw storage remains unchanged.
    gpu_normalize_device_features(
        parallel.source_features,
        parallel.target_features,
        joint,
        context);

    require_identical_raw(
        source_raw_before, download_raw(parallel.source_features, context));
    require_identical_raw(
        target_raw_before, download_raw(parallel.target_features, context));

    std::vector<NodeFeatures> source_normalized;
    std::vector<NodeFeatures> target_normalized;

    gpu_download_normalized_features(
        parallel, source_normalized, target_normalized, context);

    auto expected_source = compute_features(source);
    auto expected_target = compute_features(target);

    if (joint) {
        normalize_features_joint_zscore(expected_source, expected_target);
    } else {
        normalize_features_zscore(expected_source);
        normalize_features_zscore(expected_target);
    }

    compare_features(source_normalized, expected_source);
    compare_features(target_normalized, expected_target);

    const auto selected = gpu_download_candidates(parallel, context);
    const auto serial_selected = gpu_download_candidates(serial, context);

    // Identical GPU scoring and normalization isolate selector correctness.
    compare_candidates(selected, serial_selected);

    const CSRMatrix costs = gpu_download_costs(parallel, context);

    const CSRMatrix expected_costs = build_sparse_cost_matrix(
        selected, source.num_vertices, target.num_vertices);

    require(costs.row_ptr == expected_costs.row_ptr,
            "GPU cost offsets differ from CPU reference");
    require(costs.col_idx == expected_costs.col_idx,
            "GPU cost target order differs from CPU reference");
    require(costs.values == expected_costs.values,
            "GPU costs differ from 1 - similarity");

    // Compatibility wrapper must produce the same pipeline results.
    if (indexed) {
        auto host = gpu_compute_indexed_candidates(
            source, target, k, config, context, joint,
            GPUSelector::Parallel, options);

        require(host.matching.device_resident,
                "Indexed compatibility wrapper bypassed resident pipeline");
        compare_candidates(selected, host.matching.candidates);
        compare_features(source_normalized, host.matching.feats_source);
    } else {
        auto host = gpu_compute_candidates(
            source, target, k, context, joint,
            GPUSelector::Parallel, options);

        require(host.device_resident,
                "Compatibility wrapper bypassed resident pipeline");
        compare_candidates(selected, host.candidates);
        compare_features(source_normalized, host.feats_source);
    }

    // The previous host-input GPU scoring path is an additional reference.
    const auto reference = indexed
        ? gpu_score_indexed_pools(
            source_normalized, target_normalized,
            parallel.structural_index, parallel.structural_queries,
            k, context, GPUSelector::SerialHeap, options)
        : gpu_score_exhaustive_pools(
            source_normalized, target_normalized,
            k, context, GPUSelector::SerialHeap, options);

    /*
     * Different scoring kernel layouts may round differently.
     * Compare scores for shared targets and allow cutoff ambiguity.
     * Exact serial/parallel selection was checked above on shared GPU scoring.
     */
    for (std::size_t row = 0; row < selected.size(); ++row) {
        require(selected[row].size() == static_cast<std::size_t>(k),
                "Matching row does not contain K candidates");
        require(reference.candidates[row].size() == selected[row].size(),
                "Reference candidate count mismatch");

        for (const auto& candidate : selected[row]) {
            bool found = false;

            for (const auto& other : reference.candidates[row]) {
                if (candidate.target == other.target) {
                    require(close(candidate.similarity, other.similarity),
                            "Reference scoring mismatch");
                    found = true;
                    break;
                }
            }

            if (!found) {
                require(
                    candidate.similarity <=
                        reference.candidates[row].back().similarity + 1e-4f &&
                    candidate.similarity >=
                        selected[row].back().similarity - 1e-4f,
                    "Selection differs beyond the scoring cutoff tolerance");
            }
        }

        for (const auto& other : reference.candidates[row]) {
            bool found = false;

            for (const auto& candidate : selected[row]) {
                if (candidate.target == other.target) {
                    found = true;
                    break;
                }
            }

            if (!found) {
                require(
                    other.similarity <= selected[row].back().similarity + 1e-4f,
                    "A clearly better reference candidate was omitted");
            }
        }
    }

    // Reject nonfinite raw input before producing normalized output.
    if (!source_raw_before.empty()) {
        auto invalid_raw = source_raw_before;
        invalid_raw[0].degree =
            std::numeric_limits<float>::quiet_NaN();

        parallel.source_features.raw.upload(
            invalid_raw.data(), invalid_raw.size(), context.stream);

        bool rejected = false;

        try {
            gpu_normalize_device_features(
                parallel.source_features,
                parallel.target_features,
                joint,
                context);
        } catch (const std::runtime_error&) {
            rejected = true;
        }

        require(rejected, "Nonfinite normalization input was accepted");
    }
}

void check_empty_device_stages(const GPUContext& context) {
    // Graph construction requires positive vertex counts. Exercise empty
    // populations through the device-stage interfaces instead.
    for (int target_count : {0, 3}) {
        for (bool joint : {false, true}) {
            DeviceMatchingResult result;
            result.source_features.raw =
                DeviceBuffer<NodeFeatures>(0, context.device_id);
            result.target_features.num_vertices = target_count;
            result.target_features.raw = DeviceBuffer<NodeFeatures>(
                static_cast<std::size_t>(target_count), context.device_id);

            const std::vector<NodeFeatures> raw_targets(
                static_cast<std::size_t>(target_count), NodeFeatures{});
            result.target_features.raw.upload(
                raw_targets.data(), raw_targets.size(), context.stream);

            gpu_normalize_device_features(
                result.source_features, result.target_features, joint, context);

            require(result.source_features.normalized.empty(),
                    "Empty normalization produced source features");
            require_identical_raw(
                raw_targets, download_raw(result.target_features, context));

            std::vector<NodeFeatures> normalized_targets(raw_targets.size());
            result.target_features.normalized.download(
                normalized_targets.data(), normalized_targets.size(),
                context.stream);
            compare_features(normalized_targets, raw_targets);

            for (GPUSelector selector :
                 {GPUSelector::SerialHeap, GPUSelector::Parallel}) {
                auto selected = gpu_score_exhaustive_device_pools(
                    result.source_features, result.target_features,
                    1, context, selector);
                result.candidates = std::move(selected.candidates);
                result.costs = gpu_build_device_costs(result.candidates, context);

                require(result.candidates.entry_count == 0,
                        "Empty source population produced candidates");
                require(gpu_download_candidates(result, context).empty(),
                        "Empty source population produced host candidate rows");

                const auto costs = gpu_download_costs(result, context);
                require(costs.rows == 0 && costs.cols == target_count,
                        "Empty cost dimensions changed");
                require(costs.row_ptr == std::vector<int>{0},
                        "Empty costs require a single zero offset");
                require(costs.col_idx.empty() && costs.values.empty(),
                        "Empty costs contain entries");
            }
        }
    }
}

void run_tests(const GPUContext& context) {
    check_buffers(context);

    const Graph source = build_graph(
        3, {{0, 1}, {1, 2}});

    const Graph target = build_graph(
        5, {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {0, 4}});

    for (bool joint : {false, true}) {
        check_case(source, target, 2, joint, false, context);
        check_case(source, target, 2, joint, true, context);

        // Constant features and isolated vertices.
        check_case(
            build_graph(2, {}),
            build_graph(5, {}),
            3, joint, false, context);
    }

    check_empty_device_stages(context);

    // Preserve the K=512 / 513-target regression.
    check_case(
        build_graph(1, {}), build_graph(513, {}),
        512, true, false, context);
}

} // namespace

int main() {
    int count = 0;
    const cudaError_t available = cudaGetDeviceCount(&count);

    if (available == cudaErrorNoDevice ||
        available == cudaErrorInsufficientDriver ||
        (available == cudaSuccess && count == 0)) {
        std::cout << "SKIP: CUDA GPU unavailable\n";
        return 77;
    }

    cudaStream_t stream = nullptr;

    try {
        CUDA_CHECK(available);
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaStreamCreate(&stream));

        GPUContext context;
        context.device_id = 0;
        context.stream = stream;

        run_tests(context);

        CUDA_CHECK(cudaStreamDestroy(stream));
        stream = nullptr;

        std::cout << "GPU resident pipeline tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        if (stream != nullptr) {
            (void)cudaStreamSynchronize(stream);
            (void)cudaStreamDestroy(stream);
        }

        std::cerr << error.what() << '\n';
        return 1;
    }
}
