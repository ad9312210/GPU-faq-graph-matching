#include "gpu_topk.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace {
std::string quote(const std::string& text) {
    std::ostringstream out; out << '"';
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else out << c;
    }
    return out.str() + '"';
}
const char* selector_name(GPUSelector selector) { return selector == GPUSelector::Parallel ? "parallel" : "serial_heap"; }
const char* fallback_name(GPUFallbackReason reason) {
    switch (reason) {
    case GPUFallbackReason::None: return "none";
    case GPUFallbackReason::UnsupportedK: return "unsupported_k";
    case GPUFallbackReason::ScratchBudgetTooSmall: return "scratch_budget_too_small";
    case GPUFallbackReason::DeviceAllocationFailure: return "device_allocation_failure";
    }
    throw std::runtime_error("Unknown fallback reason");
}
std::vector<std::size_t> numbers(const std::string& value, bool allow_zero = false) {
    std::vector<std::size_t> result; std::stringstream stream(value); std::string token;
    while (std::getline(stream, token, ',')) {
        std::size_t consumed = 0;
        if (token.empty() || token.front() == '-') throw std::invalid_argument("Invalid numeric list");
        const auto parsed = std::stoull(token, &consumed);
        if (consumed != token.size() || parsed > static_cast<unsigned long long>(std::numeric_limits<int>::max()) ||
            (!allow_zero && parsed == 0)) throw std::invalid_argument("Numeric list outside supported range");
        result.push_back(static_cast<std::size_t>(parsed));
    }
    if (result.empty()) throw std::invalid_argument("Empty numeric list");
    return result;
}
std::uint32_t bits(float value) { std::uint32_t b; std::memcpy(&b,&value,sizeof(b)); return b; }
std::string fingerprint(const GPUScoredPools& pools) {
    std::uint64_t value = 14695981039346656037ULL;
    for (std::size_t i = 0; i < pools.scores.size(); ++i)
        for (std::uint32_t word : {static_cast<std::uint32_t>(pools.target_ids[i]),bits(pools.scores[i])})
            for (int shift = 0; shift < 32; shift += 8) { value ^= (word >> shift) & 255; value *= 1099511628211ULL; }
    std::ostringstream out; out << std::hex << std::setw(16) << std::setfill('0') << value; return out.str();
}
struct Trial { double wall; GPUSelectionInfo info; };
double median(std::vector<double> values) { std::sort(values.begin(),values.end()); return values[values.size()/2]; }
} // namespace
int main(int argc, char** argv) {
    GPUContext context;
    bool json = false;
    try {
        std::vector<std::size_t> sizes{65537}, ks{64}, batches{256,4096,65536};
        int warmup = 3, trials = 11; unsigned seed = 173;
        std::size_t budget = 64u*1024u*1024u;
        std::vector<std::string> distributions{"random"};
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--json") { json = true; continue; }
            if (arg == "--help") {
                std::cout << "--json --sizes N,... --ks K,... --batches B,... --budget-bytes N --warmup N --trials N --seed N --distributions random,ties,negative,zeros\n";
                return 0;
            }
            if (i+1 >= argc) throw std::invalid_argument("Missing benchmark option value");
            const std::string value = argv[++i];
            if (arg == "--sizes") sizes = numbers(value,true);
            else if (arg == "--ks") ks = numbers(value);
            else if (arg == "--batches") batches = numbers(value);
            else if (arg == "--warmup") warmup = static_cast<int>(numbers(value,true).at(0));
            else if (arg == "--trials") trials = static_cast<int>(numbers(value).at(0));
            else if (arg == "--seed") seed = static_cast<unsigned>(numbers(value,true).at(0));
            else if (arg == "--budget-bytes") {
                std::size_t consumed = 0;
                if (value.empty() || value.front() == '-') throw std::invalid_argument("Invalid byte budget");
                const auto parsed = std::stoull(value,&consumed);
                if (consumed != value.size() || parsed == 0 || parsed > std::numeric_limits<std::size_t>::max())
                    throw std::invalid_argument("Invalid byte budget");
                budget = static_cast<std::size_t>(parsed);
            }
            else if (arg == "--distributions") {
                distributions.clear(); std::stringstream stream(value); std::string item;
                while (std::getline(stream,item,',')) {
                    if (item != "random" && item != "ties" && item != "negative" && item != "zeros")
                        throw std::invalid_argument("Unknown score distribution");
                    distributions.push_back(item);
                }
                if (distributions.empty()) throw std::invalid_argument("Empty score distributions");
            } else throw std::invalid_argument("Unknown benchmark option: " + arg);
        }
        int devices = 0; const auto availability = cudaGetDeviceCount(&devices);
        if (availability == cudaErrorNoDevice || availability == cudaErrorInsufficientDriver ||
            (availability == cudaSuccess && devices == 0)) {
            std::cout << (json ? "{\"kind\":\"skipped\",\"reason\":\"no usable GPU\"}\n" : "SKIPPED: no usable GPU\n");
            return 77;
        }
        CUDA_CHECK(availability); CUDA_CHECK(cudaSetDevice(0)); CUDA_CHECK(cudaStreamCreate(&context.stream));
        cudaDeviceProp properties{}; CUDA_CHECK(cudaGetDeviceProperties(&properties,0));
        int driver = 0, runtime = 0; CUDA_CHECK(cudaDriverGetVersion(&driver)); CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
        if (json) std::cout << "{\"kind\":\"environment\",\"gpu\":" << quote(properties.name)
            << ",\"compute_capability\":" << quote(std::to_string(properties.major)+"."+std::to_string(properties.minor))
            << ",\"driver_version\":" << driver << ",\"runtime_version\":" << runtime << "}\n";
        else std::cout << "GPU: " << properties.name << "\nselector,pool_size,k,distribution,batch_capacity,median_wall_ms,median_selection_ms,fallback\n";
        std::cout << std::setprecision(17);
        for (auto size : sizes) for (const auto& distribution : distributions) {
            GPUScoredPools pools; pools.offsets = {0,size}; pools.target_ids.resize(size); pools.scores.resize(size);
            std::iota(pools.target_ids.begin(),pools.target_ids.end(),0);
            std::mt19937 rng(seed); std::shuffle(pools.target_ids.begin(),pools.target_ids.end(),rng);
            for (std::size_t i = 0; i < size; ++i) {
                const int sample = static_cast<int>(rng()%2049)-1024;
                pools.scores[i] = distribution == "ties" ? -0.5f : distribution == "zeros" ?
                    (i%2 ? -0.0f : 0.0f) : distribution == "negative" ?
                    -static_cast<float>(1+rng()%1024)/1024.0f : static_cast<float>(sample)/1024.0f;
            }
            const auto input_hash = fingerprint(pools);
            std::vector<std::size_t> order(size); std::iota(order.begin(),order.end(),0);
            std::sort(order.begin(),order.end(),[&](auto a,auto b) {
                return pools.scores[a] > pools.scores[b] ||
                    (pools.scores[a] == pools.scores[b] && pools.target_ids[a] < pools.target_ids[b]);
            });
            for (auto k : ks) for (auto batch : batches) {
                GPUTopKOptions options; options.scratch_budget_bytes = budget; options.batch_candidates = batch;
                std::vector<Trial> records[2];
                // Interleave both selectors and alternate order to reduce fixed-order bias.
                for (int run = -warmup; run < trials; ++run) for (int position = 0; position < 2; ++position) {
                    const int method = (position + (run+warmup)%2)%2;
                    const auto selector = method ? GPUSelector::Parallel : GPUSelector::SerialHeap;
                    const auto begin = std::chrono::steady_clock::now(); GPUSelectionInfo info;
                    const auto result = gpu_select_topk(pools,static_cast<int>(std::max<std::size_t>(size,1)),
                        static_cast<int>(k),context,selector,&info,options);
                    const double wall = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
                    const auto keep = std::min(size,k);
                    if (result.offsets != std::vector<std::size_t>{0,keep} || result.target_ids.size() != keep || result.scores.size() != keep)
                        throw std::runtime_error("Microbenchmark oracle output shape mismatch");
                    for (std::size_t rank = 0; rank < keep; ++rank)
                        if (result.target_ids[rank] != pools.target_ids[order[rank]] ||
                            bits(result.scores[rank]) != bits(pools.scores[order[rank]]))
                            throw std::runtime_error("Microbenchmark exact full-sort oracle failed");
                    if (info.scratch_peak_bytes > budget) throw std::runtime_error("Microbenchmark scratch budget exceeded");
                    if (run >= 0) records[method].push_back(Trial{wall,info});
                }
                for (int method = 0; method < 2; ++method) {
                    const auto requested = method ? GPUSelector::Parallel : GPUSelector::SerialHeap;
                    if (!json) {
                        std::vector<double> wall, selection;
                        for (const auto& trial : records[method]) { wall.push_back(trial.wall); selection.push_back(trial.info.selection_ms); }
                        const auto& info = records[method].back().info;
                        std::cout << selector_name(requested) << ',' << size << ',' << k << ',' << distribution << ',' << info.batch_capacity
                            << ',' << median(wall) << ',' << median(selection) << ',' << fallback_name(info.fallback) << '\n';
                        continue;
                    }
                    std::cout << "{\"kind\":\"case\",\"pool_size\":" << size << ",\"k\":" << k
                        << ",\"distribution\":" << quote(distribution) << ",\"seed\":" << seed
                        << ",\"input_fingerprint_fnv64\":" << quote(input_hash) << ",\"batch_requested\":" << batch
                        << ",\"scratch_budget_bytes\":" << budget << ",\"chunk_size\":256,\"warmup\":" << warmup
                        << ",\"requested\":" << quote(selector_name(requested)) << ",\"oracle_exact\":true,\"trials\":[";
                    bool first = true;
                    for (const auto& trial : records[method]) {
                        if (!first) std::cout << ','; first = false;
                        const auto& info = trial.info;
                        std::cout << "{\"wall_ms\":" << trial.wall << ",\"selection_ms\":" << info.selection_ms
                            << ",\"transfer_ms\":" << info.transfer_ms << ",\"used\":" << quote(selector_name(info.used))
                            << ",\"fallback\":" << quote(fallback_name(info.fallback))
                            << ",\"implementation\":" << quote(info.used == GPUSelector::Parallel ? "streaming_parallel_merge" : "streaming_serial_heap")
                            << ",\"batch_capacity\":" << info.batch_capacity << ",\"batches\":" << info.batches
                            << ",\"scratch_peak_bytes\":" << info.scratch_peak_bytes << '}';
                    }
                    std::cout << "]}\n";
                }
            }
        }
        CUDA_CHECK(cudaStreamDestroy(context.stream)); context.stream = nullptr; return 0;
    } catch (const std::exception& e) {
        if (context.stream) cudaStreamDestroy(context.stream);
        if (json) std::cout << "{\"kind\":\"error\",\"reason\":" << quote(e.what()) << "}\n";
        std::cerr << e.what() << '\n'; return 1;
    }
}
