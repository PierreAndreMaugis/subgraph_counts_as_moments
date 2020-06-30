import numpy as np
from igraph import Graph
from typing import List
from itertools import product
from graph_counting import count_copies, count_in_complete
from graph_classes import Term, get_connected_components
from graph_count_linearizer import linearizer


def estimate_moment(graph: Graph, subgraph: Graph) -> float:
    """Estimate the density of f in g; i.e., the number of copies of f in g divided by the number of copies of f in the
    complete graph over the same vertices as g."""
    return count_copies(graph, subgraph)/count_in_complete(subgraph, graph.vcount())


def moment_base(graph: Graph, b: np.ndarray, pi: np.ndarray) -> float:
    """Computes the expected density of copies of the graph g in a SBM where the probability of being in each block is
    listed in pi and the block matrix is b. Implements Formula 1.3 in arXiv:1509.07754. """
    # Parameters extraction
    edges = graph.get_edgelist()
    vertices = graph.vs.indices
    blocks = np.arange(len(pi))

    # We have to go through all assignments of vertices in blocks
    assignments = product(*[blocks for _ in vertices])

    # For each assignment is assignments we can compute the probability of a that copy
    def prob_of_copy(assignment):
        prob_of_assignment = np.prod(pi[np.array(assignment)])
        prob_of_edges = np.prod([b[assignment[i], assignment[j]] for i, j in edges])

        return prob_of_assignment*prob_of_edges

    return sum(map(prob_of_copy, assignments))


def moment(graph: Graph, b: np.ndarray, pi: np.ndarray) -> float:
    """ Computes the expected density of copies of the graph g in a SBM where the probability of being in each block is
    listed in pi and the block matrix is b. Uses that the density of a disconnected graph is the product of the density
    of each connected components. """
    return np.prod([moment_base(graph=component.graph, b=b, pi=pi) ** component.factor
                    for component in get_connected_components(graph=graph)])


def moments_cov_base(graph1: Graph, graph2: Graph, b: np.ndarray, pi: np.ndarray, n: int,
                     terms: List[Term] = None, x_h: List[float] = None,
                     mu1: List[float] = None, mu2: List[float] = None,
                     x1: List[float] = None, x2: List[float] = None) -> float:
    """Covariance of the densities of sub-graphs in a order n SBM where the probability of being in each block is listed
    in pi and the block matrix is b. From Proof of Proposition 1 of https://arxiv.org/abs/1701.00505"""
    # Getting the parameters (if not provided)
    terms = build_terms([graph1, graph2])[1][1] if terms is None else terms
    x_h = [count_in_complete(term.graph, n) for term in terms] if x_h is None else x_h
    mu1 = moment(graph1, b, pi) if mu1 is None else mu1
    mu2 = moment(graph2, b, pi) if mu2 is None else mu2
    x1 = count_in_complete(graph1, n) if x1 is None else x1
    x2 = count_in_complete(graph2, n) if x2 is None else x2

    return sum(term.factor * x_h[i] * (moment(graph=term.graph, b=b, pi=pi) - mu1 * mu2) / (x1 * x2)
               for i, term in enumerate(terms))


def build_terms(graphs: List[Graph]) -> List[List[List[Term]]]:
    """Computes linear terms and weight for every pair of elements in gs"""
    terms = [[[] for _ in graphs] for _ in graphs]
    for i, g1 in enumerate(graphs):
        for j, g2 in enumerate(graphs):
            if i <= j:  # As the ordering is indifferent, we can only compute ordered pair
                # Getting terms
                terms[i][j] = linearizer(graphs=[g1, g2])

                # Removing the disjoint copy from the terms
                disjoint_copy = g1.disjoint_union(g2)
                disjoint_copy = disjoint_copy.permute_vertices(disjoint_copy.canonical_permutation())
                looking_for_disjoint_copy = True
                k = 0  # Usually the disjoint copy is the first element
                while looking_for_disjoint_copy and k < len(terms[i][j]):
                    if disjoint_copy.isomorphic(terms[i][j][k].graph):
                        del terms[i][j][k]
                        looking_for_disjoint_copy = False
                    else:
                        k += 1
            else:
                terms[i][j] = terms[j][i]
    return terms


def build_x_h(terms: List[List[List[Term]]], n: int) -> List[List[List[float]]]:
    """Computes count_in_complete for every element in components outputted by build_components"""
    return [[[count_in_complete(graph=term.graph, n=n) for term in term_ij] for term_ij in term_i] for term_i in terms]


def build_mu_g(graphs: List[Graph], b: np.ndarray, pi: np.ndarray) -> np.array:
    """Computes mu for every element in gs"""
    return np.array([moment(graph=graph, b=b, pi=pi) for graph in graphs])


def build_x_g(graphs: List[Graph], n: int) -> np.array:
    """Computes count_in_complete for every element in gs"""
    return np.array([count_in_complete(graph=graph, n=n) for graph in graphs])


def moments_cov(graphs: List[Graph], b: np.ndarray, pi: np.ndarray, n: int,
                terms: List[List[List[Term]]] = None, x_h: List[List[List[float]]] = None,
                mu_g: np.array = None, x_g: np.array = None) -> np.array:
    """Covariance of the densities of sub-graphs in a order n SBM where the probability of being in each block is listed
    in pi and the block matrix is b. From Proof of Proposition 1 of https://arxiv.org/abs/1701.00505"""
    # Getting parameters
    terms = build_terms(graphs=graphs) if terms is None else terms
    x_h = build_x_h(terms=terms, n=n) if x_h is None else x_h
    mu_g = build_mu_g(graphs=graphs, b=b, pi=pi) if mu_g is None else mu_g
    x_g = build_x_g(graphs=graphs, n=n) if x_g is None else x_g

    # Computing covariance matrix
    k = len(graphs)
    cov_mat = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i <= j:
                cov_mat[i, j] = moments_cov_base(graph1=graphs[i], graph2=graphs[j], b=b, pi=pi, n=n,
                                                 terms=terms[i][j], x_h=x_h[i][j],
                                                 mu1=mu_g[i], mu2=mu_g[j], x1=x_g[i], x2=x_g[j])
            else:
                cov_mat[i, j] = cov_mat[j, i]
    return cov_mat
