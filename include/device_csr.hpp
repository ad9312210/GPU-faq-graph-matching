#ifndef SCGM_DEVICE_CSR_HPP
#define SCGM_DEVICE_CSR_HPP

#include "device_buffers.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>

struct DeviceCSR {
    int num_rows = 0;
    int num_cols = 0;
    std::size_t entry_count = 0;

    /*
     * Owning CSR storage:
     *
     * row_ptr.size() == num_rows + 1
     * col_idx.size() == values.size() == entry_count
     *
     * row_ptr[0] == 0
     * row_ptr[num_rows] == entry_count
     *
     * Offsets are nondecreasing. Equal consecutive offsets represent
     * an empty row.
     *
     * Within each row:
     *   - target IDs are in [0, num_cols);
     *   - target IDs are strictly increasing and therefore unique;
     *   - costs are finite;
     *   - cost = 1.0f - similarity.
     *
     * Missing entries represent forbidden assignments.
     *
     * Construction must validate these invariants before returning.
     */
    DeviceBuffer<int> row_ptr;
    DeviceBuffer<int> col_idx;
    DeviceBuffer<float> values;

    /*
     * Optional reverse column index:
     *
     * column_ptr.size() == num_cols + 1
     * source_rows.size() == csr_positions.size() == entry_count
     *
     * column_ptr[0] == 0
     * column_ptr[num_cols] == entry_count
     *
     * For each reverse-index position p in column c:
     *   source_rows[p] identifies its source row.
     *   csr_positions[p] identifies the corresponding CSR entry.
     *   col_idx[csr_positions[p]] == c.
     *
     * Source rows are strictly increasing within each column.
     * Every CSR entry appears exactly once.
     *
     * Costs are accessed through values[csr_positions[p]], so the
     * reverse index does not duplicate cost storage.
     *
     * Set has_reverse_index only after construction and validation
     * have completed successfully.
     */
    bool has_reverse_index = false;
    DeviceBuffer<int> column_ptr;
    DeviceBuffer<int> source_rows;
    DeviceBuffer<int> csr_positions;

    // Owns all allocations. Moving transfers ownership.
    // Borrowed pointers must not outlive this object.
    // Complete outstanding GPU operations before destroying or replacing it.
    DeviceCSR() noexcept = default;
    ~DeviceCSR() = default;

    DeviceCSR(const DeviceCSR&) = delete;
    DeviceCSR& operator=(const DeviceCSR&) = delete;

    DeviceCSR(DeviceCSR&&) noexcept = default;
    DeviceCSR& operator=(DeviceCSR&&) noexcept = default;

    // Check dimensions and compatibility with the existing int-based CSR.
    // This validates metadata only; device contents require GPU validation.
    static void validate_dimensions(
        int rows,
        int cols,
        std::size_t entries)
    {
        if (rows < 0 || cols < 0) {
            throw std::invalid_argument(
                "DeviceCSR: dimensions must be nonnegative");
        }

        if (entries >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error(
                "DeviceCSR: entry count exceeds int capacity");
        }

        if ((rows == 0 || cols == 0) && entries != 0) {
            throw std::invalid_argument(
                "DeviceCSR: zero-dimensional matrix cannot contain entries");
        }
    }

    static int checked_int_offset(std::size_t offset) {
        if (offset >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error(
                "DeviceCSR: offset exceeds int capacity");
        }

        return static_cast<int>(offset);
    }

    // Remove the optional reverse index without changing CSR storage.
    // Complete operations using these arrays before calling this method.
    void clear_reverse_index() noexcept {
        column_ptr.reset();
        source_rows.reset();
        csr_positions.reset();
        has_reverse_index = false;
    }
};

// Compatibility name for the existing GPU cost pipeline.
// This is the same owning type, not a second representation.
using DeviceCosts = DeviceCSR;


struct GPUContext;

// Borrowed, completed device inputs on device_id.
//
// storage_offsets has num_rows + 1 elements:
//   storage_offsets[0] == 0
//   storage_offsets[num_rows] == storage_entries
//
// target_ids and similarities each have storage_entries elements.
//
// With row_counts != nullptr:
//   row_counts has num_rows elements.
//   Each count is nonnegative and <= its row's storage capacity.
//   Valid candidates occupy the first row_counts[row] slots of that row.
//
// With row_counts == nullptr:
//   Every slot is valid, and storage_offsets already describes compact rows.
//
// Target IDs must be unique within each active row.
// Unused storage slots are ignored.
//
// Caller retains ownership and keeps inputs alive until construction returns.
struct DeviceCandidateRowsView {
    int num_rows = 0;
    int num_cols = 0;
    int device_id = -1;

    std::size_t storage_entries = 0;

    const std::size_t* storage_offsets = nullptr;
    const int* row_counts = nullptr;
    const int* target_ids = nullptr;
    const float* similarities = nullptr;
};

// Returns completed, separately owned, target-sorted CSR costs.
// Does not modify candidate storage or construct the optional reverse index.
//
// Only validation flags and the compact entry count are downloaded.
// CSR construction workspace is separate from scoring/selection scratch.
DeviceCSR gpu_build_candidate_csr(
    const DeviceCandidateRowsView& candidates,
    const GPUContext& context);

// Validate completed CSR contents on GPU.
// Checks offsets, target bounds, strictly increasing target IDs,
// and finite costs. Downloads only a validation flag.
void gpu_validate_device_csr(
    const DeviceCSR& matrix,
    const GPUContext& context);

// Construct an optional, deterministic column-oriented reverse index.
//
// Columns contain source rows in ascending order.
// csr_positions identifies the corresponding entry in matrix.values.
//
// Preserves an existing reverse index if construction fails.
// All GPU operations complete before this function returns.
void gpu_build_device_reverse_index(
    DeviceCSR& matrix,
    const GPUContext& context);

// Requires has_reverse_index == true.
// Validates CSR first, then validates reverse-index consistency.
void gpu_validate_device_reverse_index(
    const DeviceCSR& matrix,
    const GPUContext& context);

#endif // SCGM_DEVICE_CSR_HPP