#include "gpu_matching.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void run_tests(const GPUContext& context) {
    DeviceCandidates candidates;
    candidates.num_sources = 3;
    candidates.num_targets = 5;
    candidates.requested_k = 3;
    candidates.entry_count = 5;

    // Candidate rows remain in similarity order.
    // The middle row is empty.
    const std::vector<std::size_t> offsets{0, 3, 3, 5};
    const std::vector<int> targets{4, 1, 3, 2, 0};
    const std::vector<float> scores{0.75f, 0.5f, -0.25f, 1.0f, 0.0f};

    candidates.row_offsets =
        DeviceBuffer<std::size_t>(offsets.size(), context.device_id);
    candidates.target_ids =
        DeviceBuffer<int>(targets.size(), context.device_id);
    candidates.similarities =
        DeviceBuffer<float>(scores.size(), context.device_id);

    candidates.row_offsets.upload(
        offsets.data(), offsets.size(), context.stream);
    candidates.target_ids.upload(
        targets.data(), targets.size(), context.stream);
    candidates.similarities.upload(
        scores.data(), scores.size(), context.stream);

    auto costs = gpu_build_device_costs(candidates, context);

    std::vector<int> actual_offsets(4);
    std::vector<int> actual_targets(5);
    std::vector<float> actual_costs(5);

    costs.row_ptr.download(
        actual_offsets.data(), actual_offsets.size(), context.stream);
    costs.col_idx.download(
        actual_targets.data(), actual_targets.size(), context.stream);
    costs.values.download(
        actual_costs.data(), actual_costs.size(), context.stream);

    require(
        actual_offsets == std::vector<int>({0, 3, 3, 5}),
        "Incorrect cost offsets");

    require(
        actual_targets == std::vector<int>({1, 3, 4, 0, 2}),
        "Cost targets are not sorted");

    require(
        actual_costs == std::vector<float>({0.5f, 1.25f, 0.25f, 1.0f, 0.0f}),
        "Costs are not paired with the correct targets");

    // The reverse index references existing CSR entries without copying costs.
    gpu_validate_device_csr(costs, context);
    gpu_build_device_reverse_index(costs, context);
    gpu_validate_device_reverse_index(costs, context);

    require(costs.has_reverse_index,
            "Reverse index was not marked available");

    std::vector<int> column_offsets(6);
    std::vector<int> source_rows(5);
    std::vector<int> csr_positions(5);

    costs.column_ptr.download(
        column_offsets.data(), column_offsets.size(), context.stream);
    costs.source_rows.download(
        source_rows.data(), source_rows.size(), context.stream);
    costs.csr_positions.download(
        csr_positions.data(), csr_positions.size(), context.stream);

    require(
        column_offsets == std::vector<int>({0, 1, 2, 3, 4, 5}),
        "Incorrect reverse column offsets");
    require(
        source_rows == std::vector<int>({2, 0, 2, 0, 0}),
        "Incorrect reverse source rows");
    require(
        csr_positions == std::vector<int>({3, 0, 4, 1, 2}),
        "Reverse entries reference incorrect CSR positions");

    std::vector<int> unchanged_targets(5);
    std::vector<float> unchanged_scores(5);

    candidates.target_ids.download(
        unchanged_targets.data(), unchanged_targets.size(), context.stream);
    candidates.similarities.download(
        unchanged_scores.data(), unchanged_scores.size(), context.stream);

    require(unchanged_targets == targets, "Candidate target order changed");
    require(unchanged_scores == scores, "Candidate scores changed");

    // Duplicate IDs must be rejected.
    auto duplicate_targets = targets;
    duplicate_targets[1] = duplicate_targets[0];

    candidates.target_ids.upload(
        duplicate_targets.data(), duplicate_targets.size(), context.stream);

    bool rejected = false;
    try {
        auto unused = gpu_build_device_costs(candidates, context);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "Duplicate targets were accepted");

    candidates.target_ids.upload(
        targets.data(), targets.size(), context.stream);

    // Nonfinite scores must be rejected.
    auto invalid_scores = scores;
    invalid_scores[0] = std::numeric_limits<float>::quiet_NaN();

    candidates.similarities.upload(
        invalid_scores.data(), invalid_scores.size(), context.stream);

    rejected = false;
    try {
        auto unused = gpu_build_device_costs(candidates, context);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "Nonfinite score was accepted");
}

} // namespace

int main() {
    int count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&count);

    if (availability == cudaErrorNoDevice ||
        availability == cudaErrorInsufficientDriver ||
        (availability == cudaSuccess && count == 0)) {
        std::cout << "SKIP: CUDA GPU unavailable\n";
        return 77;
    }

    cudaStream_t stream = nullptr;

    try {
        CUDA_CHECK(availability);
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaStreamCreate(&stream));

        GPUContext context;
        context.device_id = 0;
        context.stream = stream;

        run_tests(context);

        CUDA_CHECK(cudaStreamDestroy(stream));
        stream = nullptr;

        std::cout << "GPU device cost tests passed\n";
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
