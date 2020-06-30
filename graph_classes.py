import numpy as np
from igraph import Graph
from typing import List, Union, Tuple
from dataclasses import dataclass
from itertools import permutations


@dataclass
class Term:
    """Convenient class to code linear forms and polynomials built on subgraph counts"""
    factor: Union[int, np.array]
    graph: Union[Graph, Tuple[Graph]]


def add_graph_to_term_list(terms: List[Term], new_graphs: Union[List[Graph], Graph]) -> List[Term]:
    """Given a list of terms, adds a new one, or a list of them. The goal is no pair of terms should be isomorphic"""
    if isinstance(new_graphs, Graph):
        new_graph_is_new = True
        i = len(terms)
        while new_graph_is_new and i > 0:
            i -= 1
            new_graph_is_new = not new_graphs.isomorphic(terms[i].graph)

        if new_graph_is_new:
            terms += [Term(factor=1, graph=new_graphs)]
        else:
            terms[i].factor += 1

        return terms

    if isinstance(new_graphs, List) and [isinstance(new_graph, Graph) for new_graph in new_graphs]:
        for new_graph in new_graphs:
            terms = add_graph_to_term_list(terms, new_graph)

        return terms


def add_term_to_term_list(terms: List[Term], new_term: Term) -> List[Term]:
    """Given a list of terms, adds a new one, or a list of them. The goal is no pair of terms should be isomorphic"""
    new_term_is_new = True
    i = len(terms)
    while new_term_is_new and i > 0:
        i -= 1
        new_term_is_new = not new_term.graph.isomorphic(terms[i].graph)

    if new_term_is_new:  # new_term_is_new is new, so we add a term
        terms.append(new_term)
    else:  # we update the matching term
        terms[i].factor += new_term.factor

    return terms


def add_graph_to_graph_list(graphs: List[Graph], new_graphs: Union[List[Graph], Graph]) -> List[Graph]:
    """Given a list of graphs, adds a new one, or a list of them. The goal is no pair of graphs should be isomorphic"""
    if isinstance(new_graphs, Graph):
        new_graph_is_new = True
        i = len(graphs)
        while new_graph_is_new and i > 0:
            i -= 1
            new_graph_is_new = not new_graphs.isomorphic(graphs[i])

        if new_graph_is_new:
            graphs += [new_graphs]

        return graphs

    if isinstance(new_graphs, List) and [isinstance(new_graph, Graph) for new_graph in new_graphs]:
        for new_graph in new_graphs:
            graphs = add_graph_to_graph_list(graphs, new_graph)

        return graphs


def graph_from_edgelist(edge_list: Union[List, np.ndarray]) -> Graph:
    """Takes a numpy edgelist (2 columns, each row is an edge) as input and returns an iGraph graph from it"""
    # Testing input
    if not isinstance(edge_list, np.ndarray):
        edge_list = np.array(edge_list)

    # Create a graph with enough vertices
    graph = Graph(np.max(edge_list) + 1)

    # Add in edges
    graph.add_edges([tuple(x) for x in edge_list])

    # Clean up
    graph.simplify()
    graph.delete_vertices([v for v in graph.vs if graph.degree(v) == 0])

    return graph


def get_connected_components(graph: Graph) -> List[Term]:
    """Returns a list of the non isomorphic connected components in g. For each graph it also
    returns the count of time a copy of the graph appears as a component."""

    connected_component_membership = np.array(graph.components().membership)
    connected_components = [graph.subgraph(np.where(connected_component_membership == c)[0])
                            for c in set(connected_component_membership)]

    ts = add_graph_to_term_list([], connected_components)

    return ts


def eq_graph_tuple(tuple1: Union[List[Graph], Tuple[Graph]], tuple2: Union[List[Graph], Tuple[Graph]]) -> bool:
    """Checks whether two graph tuples/list/iterator are equal, up to permutation"""
    if len(tuple1) != len(tuple2):
        return False
    elif {len(graph.vs) for graph in tuple1} != {len(g.vs) for g in tuple2}:
        return False
    elif {len(graph.es) for graph in tuple1} != {len(g.es) for g in tuple2}:
        return False
    else:
        for permuted_graph_tuple in permutations(tuple1):
            state = True
            for i in range(len(tuple1)):
                state *= permuted_graph_tuple[i].isomorphic(tuple2[i])
            if state:
                return True

        return False
