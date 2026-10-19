#include "candidate_repair.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace {
void validate_row(const std::vector<Candidate>& row, int targets, bool ranked) {
    std::unordered_set<int> seen;
    for (std::size_t i = 0; i < row.size(); ++i) {
        const auto c = row[i];
        if (c.target < 0 || c.target >= targets || !std::isfinite(c.similarity) ||
            !seen.insert(c.target).second)
            throw std::invalid_argument("Invalid repair candidate ID, score, or duplicate");
        if (ranked && i) {
            const auto p = row[i - 1];
            if (c.similarity > p.similarity ||
                (c.similarity == p.similarity && c.target < p.target))
                throw std::invalid_argument("Repair prefix is not ranked");
        }
    }
}
std::size_t add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::overflow_error("Repair entry count overflow");
    return a + b;
}
}

const char* candidate_repair_stop_name(CandidateRepairStop stop) noexcept {
    switch (stop) {
    case CandidateRepairStop::Feasible: return "feasible";
    case CandidateRepairStop::PrefixLimit: return "prefix_limit";
    case CandidateRepairStop::RoundLimit: return "round_limit";
    case CandidateRepairStop::CandidateLimit: return "candidate_limit";
    case CandidateRepairStop::PayloadLimit: return "payload_limit";
    }
    return "unknown";
}

CandidateConflictRegion candidate_conflict_region(
    const std::vector<std::vector<Candidate>>& candidates,
    const CandidateSupportStats& matching) {
    const auto n = candidates.size();
    const auto m = matching.target_to_source.size();
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        m > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        matching.source_to_target.size() != n)
        throw std::invalid_argument("Repair matching dimensions disagree");
    CandidateConflictRegion result;
    std::vector<unsigned char> seen_sources(n, 0), seen_targets(m, 0);
    std::vector<int> queue;
    for (std::size_t s = 0; s < n; ++s) {
        validate_row(candidates[s], static_cast<int>(m), false);
        const int t = matching.source_to_target[s];
        if (t < -1 || (t >= 0 && (static_cast<std::size_t>(t) >= m ||
            matching.target_to_source[t] != static_cast<int>(s) ||
            std::none_of(candidates[s].begin(), candidates[s].end(),
                [t](Candidate c) { return c.target == t; }))))
            throw std::invalid_argument("Repair matching is inconsistent");
        if (t == -1) {
            result.unmatched_sources.push_back(static_cast<int>(s));
            seen_sources[s] = 1;
            queue.push_back(static_cast<int>(s));
        }
    }
    for (std::size_t t = 0; t < m; ++t) {
        const int s = matching.target_to_source[t];
        if (s < -1 || (s >= 0 && (static_cast<std::size_t>(s) >= n ||
            matching.source_to_target[s] != static_cast<int>(t))))
            throw std::invalid_argument("Repair inverse matching is inconsistent");
    }
    for (std::size_t head = 0; head < queue.size(); ++head) {
        const int s = queue[head];
        for (const auto c : candidates[s]) {
            if (matching.source_to_target[s] == c.target || seen_targets[c.target]) continue;
            seen_targets[c.target] = 1;
            const int next = matching.target_to_source[c.target];
            if (next >= 0 && !seen_sources[next]) {
                seen_sources[next] = 1;
                queue.push_back(next);
            }
        }
    }
    for (std::size_t s = 0; s < n; ++s)
        if (seen_sources[s]) result.sources.push_back(static_cast<int>(s));
    return result;
}

CandidateRepairResult repair_candidate_support(
    std::vector<std::vector<Candidate>>& candidates, int targets,
    const CandidatePrefixProvider& prefix, const CandidateRepairOptions& o) {
    if (targets < 0 || candidates.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        o.initial_k <= 0 || o.k_limit < o.initial_k || o.k_limit > targets ||
        o.max_rounds < 0 || !prefix)
        throw std::invalid_argument("Invalid repair configuration");
    std::size_t edges = 0;
    for (const auto& row : candidates) {
        validate_row(row, targets, true);
        if (row.size() != static_cast<std::size_t>(o.initial_k))
            throw std::invalid_argument("Initial repair rows must contain initial_k entries");
        edges = add(edges, row.size());
    }
    if (edges > o.max_candidate_edges || edges > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        edges > o.max_payload_bytes / sizeof(Candidate))
        throw std::invalid_argument("Initial support exceeds repair limits");
    CandidateRepairResult r;
    r.initial_candidate_edges = edges;
    r.final_candidate_edges = edges;
    int k = o.initial_k;
    for (;;) {
        r.support = analyze_candidate_support(candidates, targets);
        r.history.push_back({k, r.support.maximum_matching_size, edges, 0, 0});
        if (r.support.maximum_matching_size == static_cast<int>(candidates.size())) {
            r.stop = CandidateRepairStop::Feasible; return r;
        }
        if (k == o.k_limit) { r.stop = CandidateRepairStop::PrefixLimit; return r; }
        if (r.rounds == o.max_rounds) { r.stop = CandidateRepairStop::RoundLimit; return r; }
        const auto conflict = candidate_conflict_region(candidates, r.support);
        r.history.back().conflict_sources = conflict.sources.size();
        const int next_k = k > o.k_limit / 2 ? o.k_limit : k * 2;
        std::vector<std::vector<Candidate>> staged(conflict.sources.size());
        std::size_t added = 0;
        for (std::size_t i = 0; i < conflict.sources.size(); ++i) {
            // Reserve the full requested prefix against the logical payload
            // limit before invoking its producer.
            const auto live = add(add(edges, added), static_cast<std::size_t>(next_k));
            if (live > o.max_payload_bytes / sizeof(Candidate)) {
                r.stop = CandidateRepairStop::PayloadLimit; return r;
            }
            const int s = conflict.sources[i];
            auto proposed = prefix(s, next_k);
            if (proposed.size() != static_cast<std::size_t>(next_k))
                throw std::invalid_argument("Repair producer returned an incomplete prefix");
            validate_row(proposed, targets, true);
            std::unordered_set<int> accepted;
            for (const auto c : candidates[s]) accepted.insert(c.target);
            // Reuse this allocation as the additions buffer; no second copy.
            proposed.erase(std::remove_if(proposed.begin(), proposed.end(),
                [&](Candidate c) { return accepted.count(c.target) != 0; }), proposed.end());
            added = add(added, proposed.size());
            if (add(edges, added) > o.max_candidate_edges ||
                add(edges, added) > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                r.stop = CandidateRepairStop::CandidateLimit; return r;
            }
            staged[i] = std::move(proposed);
        }
        // All potentially throwing allocations precede changes to row values.
        for (std::size_t i = 0; i < conflict.sources.size(); ++i) {
            auto& row = candidates[conflict.sources[i]];
            row.reserve(add(row.size(), staged[i].size()));
        }
        for (std::size_t i = 0; i < conflict.sources.size(); ++i) {
            auto& row = candidates[conflict.sources[i]];
            row.insert(row.end(), staged[i].begin(), staged[i].end());
        }
        edges = add(edges, added);
        r.final_candidate_edges = edges;
        r.edges_added = add(r.edges_added, added);
        r.history.back().edges_added = added;
        ++r.rounds;
        k = next_k;
    }
}
