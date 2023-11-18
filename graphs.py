import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


def draw_graph(X, Z):
    n = X.shape[0]
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            if X[i, j] == 1:
                G.add_edge(i, j)
    nx.draw(G, node_color=np.argmax(Z, axis=1))
    plt.show()


def random_graph_from_parameters(n, Q, alpha, pi):
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


def random_graph(n=None, Q=None, occupation=0.5):
    if n is None:
        n = np.random.randint(2, 100)
    if Q is None:
        Q = np.random.randint(2, n // 10)
    alpha = np.random.rand(Q)
    alpha = alpha / np.sum(alpha)
    pi = np.random.rand(Q, Q)
    pi = pi @ np.transpose(pi)
    pi = pi / np.sum(pi, axis=0) * occupation

    X, Z = random_graph_from_parameters(n, Q, alpha, pi)

    return X, Z, alpha, pi
