from typing import Optional, Tuple

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


def draw_graph(X: np.ndarray, Z: np.ndarray, tight: bool = False) -> None:
    """
    Draws a graph based on the adjacency matrix X and node assignments Z.

    Parameters
    ----------
    X : np.ndarray
        Adjacency matrix representing connections between nodes.
    Z : np.ndarray
        Node assignments matrix.
    tight : bool, optional
        Whether to use a tight layout for the graph visualization, by default False.
    """
    n = X.shape[0]
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            if X[i, j] == 1:
                G.add_edge(i, j)
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos, node_size=100 if tight else 300, node_color=np.argmax(Z, axis=1))
    plt.show()


def random_graph_from_parameters(
    n: int, Q: int, alpha: np.ndarray, pi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a random SBM graph based on input parameters.

    Parameters
    ----------
    n : int
        Number of nodes in the graph.
    Q : int
        Number of classes.
    alpha : np.ndarray
        Class membership probabilities.
    pi : np.ndarray
        Connection probabilities between classes.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the adjacency matrix X and the node assignments matrix Z.
    """
    X = np.zeros((n, n))
    Z = np.zeros((n, Q))
    for i in range(n):
        Z[i, :] = np.random.multinomial(1, alpha)

    for i in range(n):
        for j in range(i + 1, n):
            i_class = np.where(Z[i, :] == 1)[0][0]
            j_class = np.where(Z[j, :] == 1)[0][0]
            X[i, j] = np.random.binomial(1, pi[i_class, j_class])
            X[j, i] = X[i, j]

    return X, Z


def random_graph(
    n: Optional[int] = None, Q: Optional[int] = None, occupation: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates random SBM parameters and graph.

    Parameters
    ----------
    n : int, optional
        Number of nodes in the graph. If not provided, a random value between 2 and 100 will be used.
    Q : int, optional
        Number of classes. If not provided, a random value between 2 and n/10 will be used.
    occupation : float, optional
        Scaling factor for the connection probabilities. By default 0.5.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the adjacency matrix X, the node assignments matrix Z, class membership probabilities alpha, and connection probabilities matrix pi.
    """
    if n is None:
        n = np.random.randint(2, 100)
    if Q is None:
        Q = np.random.randint(2, n // 10)
    alpha = np.random.dirichlet([1.5] * Q)
    pi = np.random.rand(Q, Q)
    pi = pi @ np.transpose(pi)
    pi = pi / np.sum(pi, axis=0) * occupation

    X, Z = random_graph_from_parameters(n, Q, alpha, pi)

    return X, Z, alpha, pi
