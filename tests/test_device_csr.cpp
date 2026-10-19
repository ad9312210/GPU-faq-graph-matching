#include "common.hpp"
#include "csr_matrix.hpp"
#include "device_csr.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Exception, class F>
void reject(const char* name, F operation) {
    try {
        operation();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(std::string(name) + " was accepted");
}

template<class T>
std::vector<T> read(const DeviceBuffer<T>& buffer, const GPUContext& ctx) {
    std::vector<T> host(buffer.size());
    buffer.download(host.data(), host.size(), ctx.stream);
    return host;
}

template<class T>
void unchanged(const std::vector<T>& actual, const std::vector<T>& expected) {
    require(actual.size() == expected.size(), "Candidate storage size changed");
    // Byte comparison also covers signed zero and unused NaN padding.
    require(actual.empty() || std::memcmp(actual.data(), expected.data(),
            actual.size() * sizeof(T)) == 0, "Candidate storage was modified");
}

struct Input {
    int cols = 0;
    bool variable = false;
    std::vector<std::size_t> offsets{0};
    std::vector<int> counts;
    std::vector<int> ids;
    std::vector<float> scores;

    int rows() const { return static_cast<int>(offsets.size() - 1); }

    std::vector<std::vector<Candidate>> candidates() const {
        std::vector<std::vector<Candidate>> result(rows());
        for (int row = 0; row < rows(); ++row) {
            const std::size_t n = variable
                ? static_cast<std::size_t>(counts[row])
                : offsets[row + 1] - offsets[row];
            for (std::size_t i = 0; i < n; ++i) {
                const auto p = offsets[row] + i;
                result[row].push_back({ids[p], scores[p]});
            }
        }
        return result;
    }
};

struct Uploaded {
    DeviceBuffer<std::size_t> offsets;
    DeviceBuffer<int> counts, ids;
    DeviceBuffer<float> scores;
    DeviceCandidateRowsView view;

    Uploaded(const Input& host, const GPUContext& ctx)
        : offsets(host.offsets.size(), ctx.device_id),
          counts(host.variable ? host.counts.size() : 0, ctx.device_id),
          ids(host.ids.size(), ctx.device_id),
          scores(host.scores.size(), ctx.device_id) {
        offsets.upload(host.offsets.data(), host.offsets.size(), ctx.stream);
        if (host.variable)
            counts.upload(host.counts.data(), host.counts.size(), ctx.stream);
        ids.upload(host.ids.data(), host.ids.size(), ctx.stream);
        scores.upload(host.scores.data(), host.scores.size(), ctx.stream);
        view.num_rows = host.rows();
        view.num_cols = host.cols;
        view.device_id = ctx.device_id;
        view.storage_entries = host.ids.size();
        view.storage_offsets = offsets.data();
        view.row_counts = host.variable ? counts.data() : nullptr;
        view.target_ids = ids.data();
        view.similarities = scores.data();
    }

    void check_unchanged(const Input& host, const GPUContext& ctx) const {
        unchanged(read(offsets, ctx), host.offsets);
        if (host.variable) unchanged(read(counts, ctx), host.counts);
        unchanged(read(ids, ctx), host.ids);
        unchanged(read(scores, ctx), host.scores);
    }
};

Input example(bool variable) {
    Input input;
    input.cols = 7;
    input.variable = variable;
    if (!variable) {
        input.offsets = {0, 3, 3, 4, 6};
        input.ids = {6, 1, 4, 1, 5, 1};
        input.scores = {1.0f, 0.75f, -0.25f, -0.0f, 0.5f, -1.0f};
    } else {
        input.offsets = {0, 5, 7, 11, 14};
        input.counts = {3, 0, 1, 2};
        input.ids.assign(14, -999);
        input.scores.assign(14, std::numeric_limits<float>::quiet_NaN());
        const auto active = example(false).candidates();
        for (int row = 0; row < input.rows(); ++row) {
            for (std::size_t i = 0; i < active[row].size(); ++i) {
                const auto p = input.offsets[row] + i;
                input.ids[p] = active[row][i].target;
                input.scores[p] = active[row][i].similarity;
            }
        }
    }
    return input;
}

struct Reverse {
    std::vector<int> offsets, rows, positions;
    bool operator==(const Reverse& other) const {
        return offsets == other.offsets && rows == other.rows &&
               positions == other.positions;
    }
};

Reverse read_reverse(const DeviceCSR& csr, const GPUContext& ctx) {
    return {read(csr.column_ptr, ctx), read(csr.source_rows, ctx),
            read(csr.csr_positions, ctx)};
}

Reverse expected_reverse(const CSRMatrix& csr) {
    Reverse result;
    result.offsets.push_back(0);
    // Independent CPU column traversal provides deterministic expected indices.
    for (int column = 0; column < csr.cols; ++column) {
        for (int row = 0; row < csr.rows; ++row) {
            for (int p = csr.row_ptr[row]; p < csr.row_ptr[row + 1]; ++p) {
                if (csr.col_idx[p] == column) {
                    result.rows.push_back(row);
                    result.positions.push_back(p);
                }
            }
        }
        result.offsets.push_back(static_cast<int>(result.rows.size()));
    }
    return result;
}

void check_case(const Input& host, const GPUContext& ctx) {
    Uploaded input(host, ctx);
    auto csr = gpu_build_candidate_csr(input.view, ctx);
    gpu_validate_device_csr(csr, ctx);
    const auto expected = build_sparse_cost_matrix(
        host.candidates(), host.rows(), host.cols);
    require(csr.num_rows == expected.rows && csr.num_cols == expected.cols,
            "CSR dimensions changed");
    require(csr.entry_count == expected.col_idx.size(), "Incorrect entry count");
    require(read(csr.row_ptr, ctx) == expected.row_ptr, "CPU/GPU offsets differ");
    require(read(csr.col_idx, ctx) == expected.col_idx, "CPU/GPU target order differs");
    require(read(csr.values, ctx) == expected.values, "CPU/GPU costs differ");
    require(!csr.has_reverse_index && csr.column_ptr.empty() &&
            csr.source_rows.empty() && csr.csr_positions.empty(),
            "Reverse index was built without being requested");

    for (int repeat = 0; repeat < 3; ++repeat) {
        gpu_build_device_reverse_index(csr, ctx);
        gpu_validate_device_reverse_index(csr, ctx);
        const auto actual = read_reverse(csr, ctx);
        require(actual == expected_reverse(expected),
                "Reverse index differs from deterministic CPU reference");
        std::vector<int> seen(csr.entry_count, 0);
        for (int column = 0; column < csr.num_cols; ++column) {
            int previous = -1;
            for (int p = actual.offsets[column]; p < actual.offsets[column + 1]; ++p) {
                const int row = actual.rows[p], entry = actual.positions[p];
                require(row >= 0 && row < csr.num_rows && row > previous,
                        "Reverse source rows are invalid or unordered");
                require(entry >= 0 && static_cast<std::size_t>(entry) < seen.size(),
                        "Invalid reverse CSR position");
                require(expected.col_idx[entry] == column &&
                        entry >= expected.row_ptr[row] && entry < expected.row_ptr[row + 1],
                        "Reverse entry maps to the wrong row/column");
                ++seen[entry];
                previous = row;
            }
        }
        require(std::all_of(seen.begin(), seen.end(), [](int n) { return n == 1; }),
                "Reverse index does not visit every CSR entry exactly once");
    }
    require(read(csr.row_ptr, ctx) == expected.row_ptr &&
            read(csr.col_idx, ctx) == expected.col_idx &&
            read(csr.values, ctx) == expected.values,
            "Reverse construction modified CSR storage");
    input.check_unchanged(host, ctx);
    csr.clear_reverse_index();
    require(!csr.has_reverse_index && csr.column_ptr.empty() &&
            csr.source_rows.empty() && csr.csr_positions.empty(),
            "Reverse cleanup retained storage");
    reject<std::invalid_argument>("Missing reverse index", [&] {
        gpu_validate_device_reverse_index(csr, ctx);
    });
}

void check_limits() {
    const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
    // Exercise the supported boundary without allocating enormous matrices.
    DeviceCSR::validate_dimensions(std::numeric_limits<int>::max(),
                                   std::numeric_limits<int>::max(), limit);
    require(DeviceCSR::checked_int_offset(limit) == std::numeric_limits<int>::max(),
            "INT_MAX offset was rejected");
    reject<std::overflow_error>("Offset above INT_MAX", [&] {
        (void)DeviceCSR::checked_int_offset(limit + 1);
    });
    reject<std::overflow_error>("Entry count above INT_MAX", [&] {
        DeviceCSR::validate_dimensions(1, 1, limit + 1);
    });
    reject<std::invalid_argument>("Negative dimensions", [] {
        DeviceCSR::validate_dimensions(-1, 2, 0);
    });
    reject<std::invalid_argument>("Nonempty zero-dimensional CSR", [] {
        DeviceCSR::validate_dimensions(0, 2, 1);
    });
}

void check_invalid_inputs(const GPUContext& ctx) {
    const auto base = example(true);
    auto reject_input = [&](const Input& host, const char* name) {
        Uploaded input(host, ctx);
        reject<std::invalid_argument>(name, [&] {
            auto unused = gpu_build_candidate_csr(input.view, ctx);
        });
    };
    auto bad = base;
    bad.counts[0] = -1;
    reject_input(bad, "Negative count");
    bad = base;
    bad.counts[0] = 6;
    reject_input(bad, "Count above row capacity");
    bad = base;
    bad.offsets[0] = 1;
    reject_input(bad, "Nonzero initial offset");
    bad = base;
    --bad.offsets.back();
    reject_input(bad, "Incorrect terminal offset");
    bad = base;
    bad.offsets[2] = 4;
    reject_input(bad, "Decreasing offsets");
    bad = base;
    bad.offsets[1] = std::numeric_limits<std::size_t>::max();
    reject_input(bad, "Offset outside storage");
    bad = base;
    bad.ids[1] = bad.ids[0];
    reject_input(bad, "Duplicate active target");
    for (int target : {-1, base.cols}) {
        bad = base;
        bad.ids[0] = target;
        reject_input(bad, "Out-of-range active target");
    }
    for (float score : {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity()}) {
        bad = base;
        bad.scores[0] = score;
        reject_input(bad, "Nonfinite active score");
    }

    Uploaded input(base, ctx);
    auto view = input.view;
    view.storage_offsets = nullptr;
    reject<std::invalid_argument>("Missing offsets", [&] {
        auto unused = gpu_build_candidate_csr(view, ctx);
    });
    view = input.view;
    view.target_ids = nullptr;
    reject<std::invalid_argument>("Missing targets", [&] {
        auto unused = gpu_build_candidate_csr(view, ctx);
    });
    view = input.view;
    view.device_id = -1;
    reject<std::invalid_argument>("Wrong owning device", [&] {
        auto unused = gpu_build_candidate_csr(view, ctx);
    });
    view = input.view;
    view.storage_entries = std::numeric_limits<std::size_t>::max();
    reject<std::overflow_error>("Storage byte-count overflow", [&] {
        auto unused = gpu_build_candidate_csr(view, ctx);
    });
}

void check_corruption(const GPUContext& ctx) {
    const auto host = example(false);
    Uploaded input(host, ctx);
    auto corrupt_csr = [&](auto corrupt) {
        auto csr = gpu_build_candidate_csr(input.view, ctx);
        corrupt(csr);
        reject<std::invalid_argument>("Corrupted CSR", [&] {
            gpu_validate_device_csr(csr, ctx);
        });
        reject<std::invalid_argument>("Reverse construction from invalid CSR", [&] {
            gpu_build_device_reverse_index(csr, ctx);
        });
    };
    corrupt_csr([&](DeviceCSR& csr) {
        const int value = 1;
        csr.row_ptr.upload(&value, 1, ctx.stream);
    });
    corrupt_csr([&](DeviceCSR& csr) {
        const int value = 5;
        csr.row_ptr.upload(&value, 1, ctx.stream, 4);
    });
    corrupt_csr([&](DeviceCSR& csr) {
        const int value = 7;
        csr.row_ptr.upload(&value, 1, ctx.stream, 1);
    });
    // Negative, out-of-range, and unsorted-but-unique columns.
    for (int value : {-1, 7, 5}) {
        corrupt_csr([&](DeviceCSR& csr) {
            csr.col_idx.upload(&value, 1, ctx.stream);
        });
    }
    corrupt_csr([&](DeviceCSR& csr) {
        const int value = 1;
        csr.col_idx.upload(&value, 1, ctx.stream, 1);
    });
    for (float value : {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity()}) {
        corrupt_csr([&](DeviceCSR& csr) {
            csr.values.upload(&value, 1, ctx.stream);
        });
    }

    auto corrupt_reverse = [&](auto corrupt) {
        auto csr = gpu_build_candidate_csr(input.view, ctx);
        gpu_build_device_reverse_index(csr, ctx);
        corrupt(csr);
        reject<std::invalid_argument>("Corrupted reverse index", [&] {
            gpu_validate_device_reverse_index(csr, ctx);
        });
    };
    corrupt_reverse([&](DeviceCSR& csr) {
        const int value = 1;
        csr.column_ptr.upload(&value, 1, ctx.stream);
    });
    corrupt_reverse([&](DeviceCSR& csr) {
        const int value = 5;
        csr.column_ptr.upload(&value, 1, ctx.stream, 7);
    });
    corrupt_reverse([&](DeviceCSR& csr) {
        const int value = -1;
        csr.column_ptr.upload(&value, 1, ctx.stream, 2);
    });
    for (int value : {-1, 4}) {
        corrupt_reverse([&](DeviceCSR& csr) {
            csr.source_rows.upload(&value, 1, ctx.stream);
        });
    }
    corrupt_reverse([&](DeviceCSR& csr) {
        const int value = 0;
        csr.source_rows.upload(&value, 1, ctx.stream, 1);
    });
    for (int value : {-1, 6, 1}) {
        corrupt_reverse([&](DeviceCSR& csr) {
            csr.csr_positions.upload(&value, 1, ctx.stream);
        });
    }
    corrupt_reverse([&](DeviceCSR& csr) {
        const int value = 0;
        csr.csr_positions.upload(&value, 1, ctx.stream, 1);
    });
    corrupt_reverse([](DeviceCSR& csr) { csr.source_rows.reset(); });

    auto csr = gpu_build_candidate_csr(input.view, ctx);
    gpu_build_device_reverse_index(csr, ctx);
    const auto before = read_reverse(csr, ctx);
    const auto costs = read(csr.values, ctx);
    const float invalid = std::numeric_limits<float>::quiet_NaN();
    csr.values.upload(&invalid, 1, ctx.stream);
    reject<std::invalid_argument>("Invalid reverse rebuild", [&] {
        gpu_build_device_reverse_index(csr, ctx);
    });
    require(csr.has_reverse_index && read_reverse(csr, ctx) == before,
            "Failed rebuild destroyed an existing reverse index");
    csr.values.upload(costs.data(), costs.size(), ctx.stream);
    gpu_validate_device_reverse_index(csr, ctx);
}

void run_gpu_tests(const GPUContext& ctx) {
    check_case(example(false), ctx);
    check_case(example(true), ctx);

    Input empty;
    empty.cols = 5;
    empty.offsets = {0, 0, 0, 0};
    check_case(empty, ctx);
    empty.variable = true;
    empty.offsets = {0, 2, 4, 7};
    empty.counts = {0, 0, 0};
    empty.ids.assign(7, -999);
    empty.scores.assign(7, std::numeric_limits<float>::quiet_NaN());
    check_case(empty, ctx);
    for (int columns : {0, 5}) {
        Input zero_rows;
        zero_rows.cols = columns;
        check_case(zero_rows, ctx);
    }
    Input zero_columns;
    zero_columns.offsets = {0, 0, 0};
    check_case(zero_columns, ctx);

    std::mt19937 random(20261019u);
    for (int trial = 0; trial < 16; ++trial) {
        Input host;
        host.cols = 17;
        host.variable = trial % 2 != 0;
        for (int row = 0; row < 23; ++row) {
            std::vector<int> ids(host.cols);
            std::iota(ids.begin(), ids.end(), 0);
            std::shuffle(ids.begin(), ids.end(), random);
            const std::size_t count = random() % (ids.size() + 1);
            if (host.variable) host.counts.push_back(static_cast<int>(count));
            for (std::size_t i = 0; i < count; ++i) {
                host.ids.push_back(ids[i]);
                host.scores.push_back(
                    (static_cast<int>(random() % 17) - 8) / 8.0f);
            }
            const std::size_t padding = host.variable ? 1 + random() % 3 : 0;
            host.ids.insert(host.ids.end(), padding, -999);
            host.scores.insert(host.scores.end(), padding,
                               std::numeric_limits<float>::quiet_NaN());
            host.offsets.push_back(host.ids.size());
        }
        check_case(host, ctx);
    }
    check_invalid_inputs(ctx);
    check_corruption(ctx);
}

} // namespace

int main() {
    cudaStream_t stream = nullptr;
    try {
        check_limits();
        int count = 0;
        const cudaError_t available = cudaGetDeviceCount(&count);
        if (available == cudaErrorNoDevice ||
            available == cudaErrorInsufficientDriver ||
            (available == cudaSuccess && count == 0)) {
            std::cout << "SKIP: CUDA GPU unavailable; metadata checks passed\n";
            return 77;
        }
        CUDA_CHECK(available);
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaStreamCreate(&stream));
        GPUContext context;
        context.device_id = 0;
        context.stream = stream;
        run_gpu_tests(context);
        CUDA_CHECK(cudaStreamDestroy(stream));
        stream = nullptr;
        std::cout << "Device CSR correctness tests passed\n";
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
