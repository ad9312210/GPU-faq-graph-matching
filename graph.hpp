#ifndef SCGM_GRAPH_HPP
#define SCGM_GRAPH_HPP

#include "common.hpp"

struct Graph {
    int num_vertices;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
};

Graph build_graph(int num_vertices, const std::vector<std::pair<int, int>>& edges);

Graph load_graph(const std::string& filename);

bool has_edge(const Graph& g, int u, int v);

int degree(const Graph& g, int v);

void validate_graph(const Graph& g);

std::vector<int> neighbors(const Graph& g, int v);

void save_graph(const Graph& g, const std::string& filename);

Graph permute_graph(const Graph& g, const std::vector<int>& perm);

#endif // SCGM_GRAPH_HPP