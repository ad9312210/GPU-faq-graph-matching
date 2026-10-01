#include "graph.hpp"

Graph build_graph(int num_vertices, const std::vector<std::pair<int, int>>& edges) {
    if (num_vertices <= 0) {
        throw std::invalid_argument("build_graph: num_vertices must be positive");
    }

    // Validate edges and check for duplicates
    std::set<std::pair<int,int>> edge_set;
    for (auto& e : edges) {
        int u = e.first, v = e.second;
        if (u < 0 || u >= num_vertices || v < 0 || v >= num_vertices) {
            throw std::invalid_argument(
                "build_graph: edge (" + std::to_string(u) + "," +
                std::to_string(v) + ") out of range [0," +
                std::to_string(num_vertices) + ")");
        }
        if (u == v) {
            throw std::invalid_argument(
                "build_graph: self-loop (" + std::to_string(u) + "," +
                std::to_string(v) + ") not allowed");
        }
        int a = std::min(u, v), b = std::max(u, v);
        if (edge_set.count({a, b})) {
            throw std::invalid_argument(
                "build_graph: duplicate edge (" + std::to_string(u) + "," +
                std::to_string(v) + ")");
        }
        edge_set.insert({a, b});
    }

    // Build adjacency lists
    std::vector<std::vector<int>> adj(num_vertices);
    for (auto& e : edges) {
        adj[e.first].push_back(e.second);
        adj[e.second].push_back(e.first);
    }

    // Sort adjacency lists
    for (auto& a : adj) {
        std::sort(a.begin(), a.end());
    }

    // Build CSR
    Graph g;
    g.num_vertices = num_vertices;
    g.row_ptr.resize(num_vertices + 1);
    g.row_ptr[0] = 0;
    for (int i = 0; i < num_vertices; i++) {
        g.row_ptr[i + 1] = g.row_ptr[i] + static_cast<int>(adj[i].size());
    }
    g.col_idx.resize(g.row_ptr[num_vertices]);
    for (int i = 0; i < num_vertices; i++) {
        for (int j = 0; j < (int)adj[i].size(); j++) {
            g.col_idx[g.row_ptr[i] + j] = adj[i][j];
        }
    }

    return g;
}

Graph load_graph(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("load_graph: cannot open file: " + filename);
    }

    int num_vertices = 0;
    std::vector<std::pair<int, int>> edges;

    std::string line;
    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#' || line[0] == '%') continue;

        std::istringstream iss(line);
        // First non-comment line: check if it's "num_vertices num_edges" header
        if (num_vertices == 0) {
            int nv, ne;
            if (iss >> nv >> ne) {
                num_vertices = nv;
                continue;
            } else {
                throw std::runtime_error("load_graph: first line must be 'num_vertices num_edges'");
            }
        }

        int u, v;
        if (iss >> u >> v) {
            edges.push_back({u, v});
        }
    }

    if (num_vertices == 0) {
        throw std::runtime_error("load_graph: empty or invalid file");
    }

    return build_graph(num_vertices, edges);
}

bool has_edge(const Graph& g, int u, int v) {
    if (u < 0 || u >= g.num_vertices || v < 0 || v >= g.num_vertices) {
        return false;
    }
    int start = g.row_ptr[u];
    int end = g.row_ptr[u + 1];
    return std::binary_search(g.col_idx.begin() + start, g.col_idx.begin() + end, v);
}

int degree(const Graph& g, int v) {
    if (v < 0 || v >= g.num_vertices) {
        throw std::out_of_range("degree: vertex " + std::to_string(v) + " out of range");
    }
    return g.row_ptr[v + 1] - g.row_ptr[v];
}

void validate_graph(const Graph& g) {
    if (g.num_vertices <= 0) {
        throw std::runtime_error("validate_graph: num_vertices must be positive");
    }
    if ((int)g.row_ptr.size() != g.num_vertices + 1) {
        throw std::runtime_error("validate_graph: row_ptr size mismatch");
    }
    if (g.row_ptr[0] != 0) {
        throw std::runtime_error("validate_graph: row_ptr[0] != 0");
    }
    for (int i = 0; i < g.num_vertices; i++) {
        if (g.row_ptr[i + 1] < g.row_ptr[i]) {
            throw std::runtime_error("validate_graph: row_ptr not monotonically increasing");
        }
    }
    int nnz = g.row_ptr[g.num_vertices];
    if ((int)g.col_idx.size() != nnz) {
        throw std::runtime_error("validate_graph: col_idx size mismatch");
    }
    for (int i = 0; i < nnz; i++) {
        if (g.col_idx[i] < 0 || g.col_idx[i] >= g.num_vertices) {
            throw std::runtime_error("validate_graph: col_idx out of range");
        }
    }
}

std::vector<int> neighbors(const Graph& g, int v) {
    if (v < 0 || v >= g.num_vertices) {
        throw std::out_of_range("neighbors: vertex out of range");
    }
    int start = g.row_ptr[v];
    int end = g.row_ptr[v + 1];
    return std::vector<int>(g.col_idx.begin() + start, g.col_idx.begin() + end);
}

void save_graph(const Graph& g, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("save_graph: cannot open file: " + filename);
    }
    int num_edges = g.row_ptr[g.num_vertices] / 2;
    file << g.num_vertices << " " << num_edges << "\n";
    for (int u = 0; u < g.num_vertices; u++) {
        for (int j = g.row_ptr[u]; j < g.row_ptr[u + 1]; j++) {
            int v = g.col_idx[j];
            if (u < v) {
                file << u << " " << v << "\n";
            }
        }
    }
}

Graph permute_graph(const Graph& g, const std::vector<int>& perm) {
    if ((int)perm.size() != g.num_vertices) {
        throw std::invalid_argument("permute_graph: permutation size mismatch");
    }
    std::vector<std::pair<int, int>> new_edges;
    for (int u = 0; u < g.num_vertices; u++) {
        for (int j = g.row_ptr[u]; j < g.row_ptr[u + 1]; j++) {
            int v = g.col_idx[j];
            if (u < v) {
                new_edges.push_back({perm[u], perm[v]});
            }
        }
    }
    // Find max vertex ID in permutation
    int max_id = *std::max_element(perm.begin(), perm.end());
    return build_graph(max_id + 1, new_edges);
}