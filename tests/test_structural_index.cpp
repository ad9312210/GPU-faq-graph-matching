#include "structural_index.hpp"
#include "candidate_generation.hpp"
#include "cosine_similarity.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
using namespace std;

static void require(bool condition, const char* message) {
    if (!condition) throw runtime_error(message);
}

template<class Exception, class Function>


static void rejects(Function function) {
    bool rejected = false;
    try { function(); } catch (const Exception&) { rejected = true; }
    require(rejected, "expected exception not thrown");
}


static NodeFeatures feature(float degree, float mean) {
    NodeFeatures f{}; f.degree = degree; f.avg_neighbor_degree = mean; return f;
}

// Independent retrieval oracle: scan occupied keys, classify by distance,
// sort by (radius,key,target), then apply whole-shell stopping.
static StructuralCandidatePool oracle(const StructuralIndex& index, const NodeFeatures& source, int k) {
    const auto center = quantize_structural_features(source, index.config);

    StructuralCandidatePool p;
    auto desired = min(index.target_count, static_cast<size_t>(k) * index.config.pool_multiplier);

    for (int r = 0; r <= index.config.maximum_radius; ++r) {
        for (size_t b = 0; b < index.bucket_keys.size(); ++b) {
            auto key = index.bucket_keys[b];
            auto dx = abs(key.degree_bin-center.degree_bin);
            auto dy = abs(key.neighbor_bin-center.neighbor_bin);
            if (max(dx,dy) == r) {
                for (auto j=index.bucket_offsets[b]; j<index.bucket_offsets[b+1]; ++j)
                    p.target_ids.push_back(index.target_ids[j]);
            }
        }
        p.stats.final_radius=r;
        if (r>=index.config.minimum_radius && p.target_ids.size()>=desired) break;
    }

    p.stats.bucket_pool_size=p.target_ids.size();
    if(p.target_ids.size()<desired) {
        p.stats.used_exhaustive_fallback=true;
        p.target_ids.clear();
        for(size_t j=0;j<index.target_count;++j) p.target_ids.push_back(static_cast<int>(j));
    }
    return p;
}


int main() {
    try {
        StructuralIndexConfig config;
        rejects<invalid_argument>([&]{auto c=config;c.degree_bin_width=0;validate_structural_index_config(c);});
        rejects<invalid_argument>([&]{auto c=config;c.minimum_radius=4;validate_structural_index_config(c);});
        rejects<invalid_argument>([&]{auto c=config;c.pool_multiplier=0;validate_structural_index_config(c);});
        rejects<invalid_argument>([&]{quantize_structural_features(feature(-1,0),config);});
        rejects<invalid_argument>([&]{quantize_structural_features(feature(INFINITY,0),config);});
        rejects<overflow_error>([&]{auto c=config;c.degree_bin_width=numeric_limits<double>::denorm_min();quantize_structural_features(feature(1,0),c);});
        require(quantize_structural_features(feature(0,0),config)==StructuralBucketKey{0,0},"zero key");

        auto boundary=config;boundary.degree_bin_width=log1p(1.0);
        require(quantize_structural_features(feature(1,0),boundary).degree_bin==1,"boundary key");
        require(quantize_structural_features(feature(nextafter(1.0f,0.0f),0),boundary).degree_bin==0,"below boundary");
        require(quantize_structural_features(feature(nextafter(1.0f,2.0f),0),boundary).degree_bin==1,"above boundary");


        auto empty=build_structural_index({},config);
        require(empty.bucket_offsets==vector<size_t>{0},"empty offsets");
        rejects<invalid_argument>([&]{retrieve_structural_candidates(empty,feature(0,0),1);});
        auto huge=build_structural_index(vector<NodeFeatures>(100,feature(0,0)),config);
        auto all=retrieve_structural_candidates(huge,feature(0,0),1);
        require(all.target_ids.size()==100 && !all.stats.used_exhaustive_fallback,"oversized bucket truncated");
        require(all.stats.final_radius==1,"minimum radius ignored");
        auto fallback=retrieve_structural_candidates(huge,feature(100000,100000),1);
        require(fallback.stats.used_exhaustive_fallback && fallback.stats.bucket_pool_size==0,"fallback missing");
        rejects<invalid_argument>([&]{retrieve_structural_candidates(huge,feature(0,0),0);});
        rejects<invalid_argument>([&]{retrieve_structural_candidates(huge,feature(0,0),101);});
        mt19937 rng(20261016);


        for(int trial=0;trial<100;++trial) {
            config.minimum_radius=trial%2;config.maximum_radius=3;config.pool_multiplier=1+trial%4;
            vector<NodeFeatures> targets;
            for(int j=0;j<23;++j) targets.push_back(feature(rng()%100,rng()%100));
            auto index=build_structural_index(targets,config);
            require(index.bucket_offsets.size()==index.bucket_keys.size()+1,"offset count");
            require(index.bucket_offsets.back()==targets.size(),"final offset");
            auto ids=index.target_ids;sort(ids.begin(),ids.end());
            for(int j=0;j<23;++j) require(ids[j]==j,"target uniqueness");
            for(size_t b=0;b<index.bucket_keys.size();++b) {
                if(b) require(index.bucket_keys[b-1]<index.bucket_keys[b],"key order");
                int previous=-1;
                for(auto j=index.bucket_offsets[b];j<index.bucket_offsets[b+1];++j) {
                    int id=index.target_ids[j];require(id>previous,"bucket ID order");previous=id;
                    require(quantize_structural_features(targets[id],config)==index.bucket_keys[b],"bucket membership");
                }
            }


            auto source=feature(rng()%100,rng()%100);
            int k=1+trial%5;
            auto actual=retrieve_structural_candidates(index,source,k);
            auto expected=oracle(index,source,k);
            require(actual.target_ids==expected.target_ids,"retrieval differs from independent oracle");
            require(actual.stats.final_radius==expected.stats.final_radius,"stopping radius");
            require(actual.stats.bucket_pool_size==expected.stats.bucket_pool_size,"pool count");
            require(actual.stats.used_exhaustive_fallback==expected.stats.used_exhaustive_fallback,"fallback decision");
            require(set<int>(actual.target_ids.begin(),actual.target_ids.end()).size()==actual.target_ids.size(),"duplicate retrieval");
            vector<StructuralRetrievalStats> stats;
            auto selected=generate_indexed_top_k_candidates({source},{source},targets,index,k,&stats);
            vector<Candidate> ranked;
            for(int id:expected.target_ids) ranked.push_back({id,cosine_similarity(source,targets[id])});
            sort(ranked.begin(),ranked.end(),[](auto a,auto b){return a.similarity>b.similarity || (a.similarity==b.similarity && a.target<b.target);});
            require(selected[0].size()==static_cast<size_t>(k),"candidate count");
            for(int j=0;j<k;++j) require(selected[0][j].target==ranked[j].target && selected[0][j].similarity==ranked[j].similarity,"indexed ranking");
            if(actual.stats.used_exhaustive_fallback) {
                auto exhaustive=generate_top_k_candidates({source},targets,k);
                for(int j=0;j<k;++j) require(selected[0][j].target==exhaustive[0][j].target,"fallback exhaustive parity");
            }
        }
        // Equal zero-vector scores: target ID determines ranking.
        auto tied=generate_indexed_top_k_candidates({feature(0,0)},{feature(0,0)},vector<NodeFeatures>(100,feature(0,0)),huge,3);
        require(tied[0][0].target==0 && tied[0][2].target==2,"ties");
        cout << "test_structural_index: explicit checks and 100 retrieval/ranking oracle cases PASSED\n";
        return 0;
    } catch(const exception& error) {
        cerr << "test_structural_index: FAIL: " << error.what() << '\n';
        return 1;
    }
}
