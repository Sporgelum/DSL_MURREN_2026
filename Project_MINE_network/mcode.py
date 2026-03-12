"""
MCODE module detection (Bader & Hogue 2003, BMC Bioinformatics 4:2).

Used by Li et al. (Nat Immunol 2014) for de-novo module search in blood
transcription module (BTM) networks.

Algorithm:
  Stage 1 — Vertex weighting: w(v) = k-core(v) × local_density(v)
  Stage 2 — Seed-and-extend: BFS from highest-weight seeds
  Stage 3 — Filter by min_size and min_density

Adapted from generate_net_python_pval.py with minimal changes.
"""

import numpy as np
import igraph as ig
import time


def _k_core_levels(adj_sym):
    """K-core decomposition using igraph (fast, O(V + E))."""
    g = ig.Graph.Adjacency((adj_sym > 0).tolist(), mode="undirected")
    return np.array(g.coreness(), dtype=int)


def _local_density(v, neighbours_v, adj_sym):
    """Edge density within the closed neighbourhood of v."""
    members = [v] + list(neighbours_v)
    n = len(members)
    if n < 2:
        return 0.0
    sub = adj_sym[np.ix_(members, members)]
    present = int(np.triu(sub, k=1).sum())
    possible = n * (n - 1) // 2
    return present / possible


def mcode(adj_matrix, gene_names,
          score_threshold=0.2, min_size=3, min_density=0.3):
    """
    MCODE module detection.

    Parameters
    ----------
    adj_matrix      : (n, n) binary symmetric adjacency
    gene_names      : list of str, length n
    score_threshold : fraction of max node weight to include neighbours
    min_size        : discard modules smaller than this
    min_density     : discard modules below this edge density

    Returns
    -------
    modules    : dict {module_id: [gene_name, ...]}
    membership : dict {gene_name: module_id}  (largest module if ambiguous)
    """
    print("[INFO] Running MCODE module detection...")
    t0 = time.time()

    adj_sym = np.maximum(adj_matrix, adj_matrix.T).astype(np.uint8)
    n = adj_sym.shape[0]

    # Stage 1: vertex weighting
    print("[INFO] MCODE stage 1: vertex weighting...")
    core = _k_core_levels(adj_sym)
    neighbours = [list(np.where(adj_sym[v] > 0)[0]) for v in range(n)]

    weights = np.zeros(n)
    for v in range(n):
        weights[v] = core[v] * _local_density(v, neighbours[v], adj_sym)

    max_weight = weights.max()
    if max_weight == 0:
        print("[WARN] All node weights zero — graph may be empty.")
        return {}, {}

    print(f"[INFO] Node weight range: [{weights.min():.4f}, {max_weight:.4f}]")

    # Stage 2: seed-and-extend
    print("[INFO] MCODE stage 2: seed-and-extend...")
    wt_threshold = score_threshold * max_weight
    seed_order = np.argsort(-weights)
    visited = np.zeros(n, dtype=bool)
    raw_modules = []

    for seed in seed_order:
        if visited[seed]:
            continue
        module_nodes = set()
        queue = [seed]
        while queue:
            v = queue.pop()
            if v in module_nodes:
                continue
            if weights[v] >= wt_threshold:
                module_nodes.add(v)
                for u in neighbours[v]:
                    if u not in module_nodes and weights[u] >= wt_threshold:
                        queue.append(u)
        if module_nodes:
            raw_modules.append(sorted(module_nodes))
            for v in module_nodes:
                visited[v] = True

    print(f"[INFO] MCODE stage 2: {len(raw_modules)} raw complexes")

    # Stage 3: post-processing
    print(f"[INFO] MCODE stage 3: filtering (min_size={min_size}, "
          f"min_density={min_density})...")
    modules = {}
    mid = 0
    for node_list in raw_modules:
        if len(node_list) < min_size:
            continue
        sub = adj_sym[np.ix_(node_list, node_list)]
        m_edges = int(np.triu(sub, k=1).sum())
        m_nodes = len(node_list)
        m_possible = m_nodes * (m_nodes - 1) // 2
        density = m_edges / m_possible if m_possible > 0 else 0.0
        if density < min_density:
            continue
        modules[mid] = [gene_names[i] for i in node_list]
        mid += 1

    elapsed = time.time() - t0
    print(f"[INFO] MCODE: {len(modules)} modules after post-processing "
          f"({elapsed:.1f}s)")
    if modules:
        sizes = sorted([len(v) for v in modules.values()], reverse=True)
        print(f"[INFO] Module sizes (top 10): {sizes[:10]}")

    # Build membership: gene → largest module
    gene_to_modules = {}
    for mid_id, genes in modules.items():
        for g in genes:
            gene_to_modules.setdefault(g, []).append((len(genes), mid_id))

    membership = {
        g: max(entries, key=lambda x: x[0])[1]
        for g, entries in gene_to_modules.items()
    }

    return modules, membership
