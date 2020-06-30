import numpy as np
from igraph import Graph


def falling_factorial(n: int, k: int) -> float:
    """Computes the falling factorial, (n)_k = n * (n-1) * ... * (n-k+1)."""
    # Testing input
    if not k <= n:
        raise ValueError(f"Input {k} must be smaller than {n}")

    return np.prod(np.arange(n-k+1, n+1))


def count_in_complete(graph: Graph, n: int) -> float:
    """Counts copies of g in a complete graph over n vertices"""
    return falling_factorial(n, len(graph.vs))/graph.count_subisomorphisms_vf2(graph)


def count_copies(graph: Graph, subgraph: Graph) -> float:
    """Counts copies of f in g"""
    # Counting copies and automorphisms
    nb_sub_iso = graph.count_subisomorphisms_vf2(subgraph)
    nb_self_iso = subgraph.count_subisomorphisms_vf2(subgraph)

    return nb_sub_iso/nb_self_iso
