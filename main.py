# A mixture model for random graphs
# J.-J. Daudin · F. Picard · S. Robin

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# n noeuds
# Q classes
# X : n x n
# Z : n x Q
# tau : n x Q
# alpha : Q
# pi : Q x Q

MAX_FIXED_POINT_ITERATIONS = 1000
EPSILON = 1e-10
EM_ITERATIONS = 100


def m_step(X, tau):
    alpha = np.mean(tau, axis=0)
    Q = tau.shape[1]
    n = X.shape[0]
    pi = np.zeros((Q, Q))
    for q in range(Q):
        for l in range(Q):
            num = 0
            denum = 0
            for i in range(n):
                for j in range(n):
                    if i != j:
                        num += tau[i, q] * tau[j, l] * X[i, j]
                        denum += tau[i, q] * tau[j, l]
            # pi[q, l] = np.einsum("ij,i,j->", X, tau[:, q], tau[:, l]) / (
            #    np.einsum("i,j->", tau[:, q], tau[:, l]) - np.sum(tau[:, q] * tau[:, l])
            # )
            pi[q, l] = num / denum
    return alpha, pi


def b(x, pi):
    return pi**x * (1 - pi) ** (1 - x)


def e_step(X, alpha, pi):
    n = X.shape[0]
    Q = alpha.shape[0]
    tau = np.repeat(alpha.reshape(1, Q), n, axis=0)
    precomputed_b_s = np.zeros((n, Q, n, Q))
    for i in range(n):
        for q in range(Q):
            for j in range(n):
                for l in range(Q):
                    precomputed_b_s[i, q, j, l] = b(X[i, j], pi[q, l])
    for _ in range(MAX_FIXED_POINT_ITERATIONS):
        previous_tau = np.copy(tau)
        for i in range(n):
            for q in range(Q):
                tau[i, q] = alpha[q]
                for j in range(n):
                    if j == i:
                        continue
                    tau[i, q] *= np.prod(
                        np.float_power(precomputed_b_s[i, q, j, :], tau[j, :])
                    )
        # tau = tau / np.sum(tau, axis=1)[:, np.newaxis]
        if np.linalg.norm(previous_tau - tau, ord=1) < EPSILON:
            break
    return tau


def log_likelihood(X, alpha, pi, tau):
    n = X.shape[0]
    Q = alpha.shape[0]
    ll = 0
    for i in range(n):
        for q in range(Q):
            ll += tau[i, q] * np.log(alpha[q]) - tau[i, q] * np.log(tau[i, q])
            for j in range(n):
                if j == i:
                    continue
                else:
                    for l in range(Q):
                        ll += tau[i, q] * tau[j, l] * np.log(b(X[i, j], pi[q, l]))
    return ll


def em_algorithm(X, Q):
    # Initialisation
    alpha = np.random.rand(Q)
    alpha = alpha / np.sum(alpha)
    pi = np.ones((Q, Q)) / Q**2
    # EM algorithm
    for i in range(EM_ITERATIONS):
        tau = e_step(X, alpha, pi)
        alpha, pi = m_step(X, tau)
        print(
            f"After EM iteration {i}/{EM_ITERATIONS} : Log likelihood {log_likelihood(X, alpha, pi, tau)}...",
            end="\r",
        )
    print()
    return alpha, pi, tau


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
            if i == j:
                continue
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

    return random_graph_from_parameters(n, Q, alpha, pi)


if __name__ == "__main__":
    # Test the algorithm on a random graph
    X, Z = random_graph_from_parameters(
        20,
        2,
        np.array([0.5, 0.5]),
        np.array([[0.2, 0], [0, 0.2]]),
    )

    draw_graph(X, Z)

    alpha, pi, tau = em_algorithm(X, 3)
    print(alpha)
    print(pi)
    print(tau)
