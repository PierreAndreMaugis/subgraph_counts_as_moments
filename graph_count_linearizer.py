import numpy as np
from igraph import Graph
from typing import List
from itertools import product, permutations, combinations
from graph_classes import Term, graph_from_edgelist, add_graph_to_graph_list


def linearizer_pair(graph1: Graph, graph2: Graph) -> List[Graph]:
    """For g1 and g2 two iGraph graphs, produces all the graph necessary to linearize the product of the counts of g1
    and g2 in any larger graph."""
    # Vertex sets
    v1 = np.array(graph1.vs.indices)
    v2 = np.array(graph2.vs.indices) + np.max(v1) + 1  # Relabelling

    # Edge-lists
    edge_list1 = np.array(graph1.get_edgelist())
    edge_list2 = np.array(graph2.get_edgelist()) + np.max(v1) + 1  # Relabelling
    edge_list = np.vstack((edge_list1, edge_list2))  # Edgelist of the disjoint union

    # Initializing
    gs = [graph_from_edgelist(edge_list)]

    # We need to consider all ordered tuples of vertices of each graph and then match these.
    for k in np.arange(min(len(v1), len(v2)))+1:  # For a given tuple size
        # For all k-subset of vertices in g1 and g2
        for vg1, vg2 in product(*[combinations(v1, k), combinations(v2, k)]):
            for p_vg1 in permutations(vg1):  # Over all permutations of vg1
                # We now have to creat a graph where p_vg1[i] is merged with vg2[i], for i in range(k)
                new_edge_list = edge_list.copy()
                for i in range(k):
                    new_edge_list[edge_list == p_vg1[i]] = vg2[i]
                new_g = graph_from_edgelist(new_edge_list)
                new_g = new_g.permute_vertices(new_g.canonical_permutation())  # Might save time

                # We discard this graph if we have already seen it; the most similar graph is likely to be
                # at the end of the list, so we start from there, and stop once we find any match
                gs = add_graph_to_graph_list(gs, new_g)

    return gs


def linearizer_base(graphs: List[Graph]) -> List[Graph]:
    """Generalizes linearizer_base to product of any list of graphs"""
    # Cases
    if len(graphs) == 1:  # For length 1, there is nothing to do
        return graphs
    elif len(graphs) == 2:  # For length 2 we can revert to the _base function
        return linearizer_pair(graphs[0], graphs[1])
    else:
        # Pick up one graph, the first in the list
        g0 = graphs[0]
        # Get all the components form the rest fo the list by a recursive call
        base_components = linearizer_base(graphs[1:])
        # We now build on base_components by, for each base_component, create all the components
        # created by base_component and g0
        components = []
        for base_component in base_components:
            for created_component in linearizer_pair(base_component, g0):
                components = add_graph_to_graph_list(components, created_component)

        return components


def c_h(graphs: List[Graph], h: Graph) -> int:
    """Computes parameter c_h from Appendix A Definition 5 in https://arxiv.org/abs/1701.00505"""
    # Extracting parameters
    n = h.ecount()
    e = np.array(h.get_edgelist())

    # We build all the collections of subsets of the row indices of e. Each collection contains a
    # set for each g in gs, and each set contains the same number of row indices as the number of
    # edges in the matching graph. (Creating the the iterator takes no time and memory)
    collection_of_subsets = product(*[combinations(range(n), g.ecount()) for g in graphs])

    # For all collections of subsets of the rows of e, we keep those collections such that: each
    # of the set of edges thus index induce a copy of a different element of gs; all the rows of
    # e are present. (Creating the the iterator takes no time and memory)
    def filter_fun(i):
        if not len(set.union(*[set(list(ii)) for ii in i])) == h.ecount():
            # If the union of rows do not cover e
            return False
        else:
            for j, g in zip(i, graphs):
                if not graph_from_edgelist(e[j, :]).isomorphic(g):
                    # If the indexed rows do not induce g
                    return False
            # If we get here, we have met all criterion
            return True
    filtered_collection_of_subsets = filter(filter_fun, collection_of_subsets)

    # We count how many cases remain
    count = sum(1 for _ in filtered_collection_of_subsets)

    return count


def linearizer(graphs: List[Graph]) -> List[Term]:
    """Computes the factor necessary to linearize the product of counts of the elements of g."""
    return [Term(factor=c_h(graphs, h), graph=h) for h in linearizer_base(graphs)]
