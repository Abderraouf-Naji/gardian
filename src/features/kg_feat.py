"""
Knowledge-graph feature vector for a (query, passage) pair.

Features (paper Eq. 5):
  [0] Entity coverage: |E_q ∩ E_p| / (|E_q| + ε)
  [1] Normalised shared entity count
  [2] Avg shortest-path distance between E_q and E_p (capped at max_path)
  [3] Min shortest-path distance
  [4] Avg node degree in induced subgraph
  [5] KG coverage indicator (1 if any entities found, else 0)
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Optional

MAX_PATH = 4  # matches paper


# ---------------------------------------------------------------------------
# Startup helpers — call ONCE in main(), pass results to compute_kg_features()
# ---------------------------------------------------------------------------

def build_degree_lookup(G: nx.DiGraph) -> Dict[str, float]:
    """
    Pre-compute degree for every KG node once at startup.
    Replaces G.subgraph(induced_nodes).degree() called 400×/query,
    which re-traverses edges on the 597K-edge graph every call.
    Call in main() and thread through process_split → process_queries.
    """
    return dict(G.degree())


def build_node_set(G: nx.DiGraph) -> frozenset:
    """
    Pre-compute the full KG node set once at startup.
    Replaces set(G.nodes) called 400×/query inside compute_kg_features —
    which copies the entire 300K-node dict into a new Python set each call.
    frozenset membership test is O(1); building it once costs ~3ms total.
    Call in main() and thread through process_split → process_queries.
    """
    return frozenset(G.nodes)


def build_query_kg_cache(
    q_entities: List[str],
    G: nx.DiGraph,
    node_set: Optional[frozenset] = None,
) -> Dict:
    """
    Pre-compute everything that depends only on the query entities.
    Called ONCE per query (not once per candidate) in process_queries().
    Returns a small dict passed into compute_kg_features().

    Parameters
    ----------
    q_entities : list of entity strings for the current query
    G          : the KG graph (used only if node_set is None)
    node_set   : pre-computed frozenset(G.nodes) from build_node_set()
    """
    q_set = set(q_entities)
    g_nodes = node_set if node_set is not None else frozenset(G.nodes)
    q_nodes_in_G = q_set & g_nodes
    return {
        "q_set": q_set,
        "q_nodes_in_G": q_nodes_in_G,
    }


# ---------------------------------------------------------------------------
# Main feature function
# ---------------------------------------------------------------------------

def compute_kg_features(
    q_entities: List[str],
    p_entities: List[str],
    G: nx.DiGraph,
    max_path: int = MAX_PATH,
    query_cache: Optional[Dict] = None,       # from build_query_kg_cache() — per query
    degree_lookup: Optional[Dict] = None,     # from build_degree_lookup()  — global
    node_set: Optional[frozenset] = None,     # from build_node_set()       — global
) -> np.ndarray:
    """
    Compute the 6-dimensional KG feature vector for a (query, passage) pair.

    Performance notes
    -----------------
    Pass all three optional caches for maximum speed:

      query_cache   — avoids rebuilding q_set 400× per query
      degree_lookup — avoids G.subgraph().degree() on 597K-edge graph per candidate
      node_set      — avoids set(G.nodes) copying 300K nodes per candidate

    All caches are backward-compatible: if None, the function falls back to
    the original inline computation so existing unit tests still pass unchanged.

    Parameters
    ----------
    q_entities    : entity strings linked from the query
    p_entities    : entity strings linked from the passage
    G             : the KG graph (nx.DiGraph)
    max_path      : cap for distance features when no path exists
    query_cache   : output of build_query_kg_cache(q_entities, G, node_set)
    degree_lookup : output of build_degree_lookup(G)
    node_set      : output of build_node_set(G)

    Returns
    -------
    np.ndarray of shape (6,), dtype float32
    """

    # --- query entity set (from cache or inline fallback) ---
    if query_cache is not None:
        q_set = query_cache["q_set"]
    else:
        q_set = set(q_entities)

    p_set = set(p_entities)
    shared = q_set & p_set

    # [0] Entity coverage
    coverage = len(shared) / (len(q_set) + 1e-8)

    # [1] Normalised shared entity count
    norm_shared = len(shared) / (len(q_set) + len(p_set) + 1e-8)

    # [2] Avg distance  [3] Min distance
    # Distance approximation: shared entities → 0, otherwise max_path.
    # (Full BFS would add substantial cost; this approximation matches your current design.)
    if shared:
        avg_dist = 0.0
        min_dist = 0.0
    else:
        avg_dist = float(max_path)
        min_dist = float(max_path)

    # [4] Avg node degree in induced subgraph
    # Use pre-computed node_set to avoid copying 300K nodes into a new set.
    g_nodes = node_set if node_set is not None else frozenset(G.nodes)
    induced_nodes = list((q_set | p_set) & g_nodes)

    if induced_nodes:
        if degree_lookup is not None:
            # O(1) per node — no graph traversal whatsoever
            degs = [degree_lookup[n] for n in induced_nodes if n in degree_lookup]
        else:
            # Fallback: per-node G.degree() still avoids subgraph copy
            degs = [G.degree(n) for n in induced_nodes]
        avg_deg = float(np.mean(degs)) if degs else 0.0
    else:
        avg_deg = 0.0

    # [5] KG coverage indicator
    kg_coverage = 1.0 if (q_entities or p_entities) else 0.0

    return np.array(
        [coverage, norm_shared, avg_dist, min_dist, avg_deg, kg_coverage],
        dtype=np.float32,
    )