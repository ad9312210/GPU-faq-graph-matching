#include "candidate_repair.hpp"
#include <iostream>
#include <stdexcept>
#include <limits>
#include <string>

namespace {
void require(bool value) {
    if (!value) throw std::runtime_error("candidate repair check failed");
}
using Rows = std::vector<std::vector<Candidate>>;
Rows initial() { return {{{0, 1.0f}}, {{0, 1.0f}}, {{2, 1.0f}}}; }
std::vector<Candidate> prefix(int s, int k) {
    require(k == 2);
    if (s == 0) return {{0, 1.0f}, {1, 0.5f}};
    if (s == 1) return {{0, 1.0f}, {1, 0.4f}};
    throw std::runtime_error("Unaffected source was expanded");
}
}
int main() {
    try {
        CandidateRepairOptions options;
        options.initial_k = 1;
        options.k_limit = 2;
        options.max_rounds = 1;
        auto rows = initial();
        auto region = candidate_conflict_region(rows, analyze_candidate_support(rows, 3));
        require(region.sources == std::vector<int>({0, 1}));
        require(region.unmatched_sources.size() == 1);
        auto repaired = repair_candidate_support(rows, 3, prefix, options);
        require(repaired.stop == CandidateRepairStop::Feasible);
        require(repaired.rounds == 1 && repaired.edges_added == 2);
        require(repaired.initial_candidate_edges == 3 && repaired.final_candidate_edges == 5);
        require(repaired.support.maximum_matching_size == 3);
        require(rows[2].size() == 1);
        require(rows[0][0].target == 0 && rows[0][0].similarity == 1.0f);
        require(rows[1][0].target == 0 && rows[1][0].similarity == 1.0f);

        // Identical inputs must produce identical rows and diagnostics.
        auto repeated_rows = initial();
        const auto repeated = repair_candidate_support(repeated_rows, 3, prefix, options);
        require(repeated.stop == repaired.stop && repeated.rounds == repaired.rounds);
        for (std::size_t s = 0; s < rows.size(); ++s) {
            require(rows[s].size() == repeated_rows[s].size());
            for (std::size_t i = 0; i < rows[s].size(); ++i)
                require(rows[s][i].target == repeated_rows[s][i].target &&
                        rows[s][i].similarity == repeated_rows[s][i].similarity);
        }

        // All prefixes remain inside two targets: three sources cannot match.
        Rows deficient{{{0, 1.0f}}, {{0, 1.0f}}, {{0, 1.0f}}};
        const auto irreparable = repair_candidate_support(deficient, 3,
            [](int, int) { return std::vector<Candidate>{{0, 1.0f}, {1, 0.5f}}; }, options);
        require(irreparable.stop == CandidateRepairStop::PrefixLimit);
        require(irreparable.support.maximum_matching_size == 2);
        require(irreparable.rounds == 1 && irreparable.edges_added == 3);
        require(std::string(candidate_repair_stop_name(irreparable.stop)) == "prefix_limit");

        rows = initial();
        options.max_rounds = 0;
        require(repair_candidate_support(rows, 3, prefix, options).stop == CandidateRepairStop::RoundLimit);
        options.max_rounds = 1;
        options.max_candidate_edges = 4;
        require(repair_candidate_support(rows, 3, prefix, options).stop == CandidateRepairStop::CandidateLimit);
        require(rows[0].size() == 1 && rows[1].size() == 1);
        options.max_candidate_edges = 100;
        options.max_payload_bytes = 3 * sizeof(Candidate);
        require(repair_candidate_support(rows, 3, prefix, options).stop == CandidateRepairStop::PayloadLimit);
        options.max_payload_bytes = 1024;
        options.k_limit = 1;
        require(repair_candidate_support(rows, 3, prefix, options).stop == CandidateRepairStop::PrefixLimit);
        options.k_limit = 2;

        bool rejected = false;
        try {
            repair_candidate_support(rows, 3, [](int, int) {
                return std::vector<Candidate>{{0, 1.0f}, {0, 0.5f}};
            }, options);
        } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected && rows[0].size() == 1 && rows[1].size() == 1);

        // Invalid later callback output must not commit earlier staged rows.
        rejected = false;
        try {
            repair_candidate_support(rows, 3, [](int s, int) {
                if (s == 0) return std::vector<Candidate>{{0, 1.0f}, {1, 0.5f}};
                return std::vector<Candidate>{{0, 1.0f},
                    {1, std::numeric_limits<float>::quiet_NaN()}};
            }, options);
        } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected && rows[0].size() == 1 && rows[1].size() == 1);

        // Tied scores must follow target-ID order in provider prefixes.
        rejected = false;
        try {
            repair_candidate_support(rows, 3, [](int, int) {
                return std::vector<Candidate>{{1, 1.0f}, {0, 1.0f}};
            }, options);
        } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected);

        // A rectangular support can be repaired without assuming N == M.
        Rows rectangle{{{0, 1.0f}}, {{0, 1.0f}}};
        require(repair_candidate_support(rectangle, 3, prefix, options).stop == CandidateRepairStop::Feasible);
        Rows feasible{{{0, 1.0f}}, {{1, 1.0f}}};
        require(repair_candidate_support(feasible, 3, prefix, options).rounds == 0);
        std::cout << "Candidate repair tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
