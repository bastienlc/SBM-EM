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

MAX_FIXED_POINT_ITERATIONS = 100
EPSILON = 1e-8
PRECISION = 1e-5
EM_ITERATIONS = 1000


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
            pi[q, l] = num / denum
    return alpha, pi


def b(x, pi):
    return pi**x * (1 - pi) ** (1 - x)


def e_step(X, alpha, pi):
    n = X.shape[0]
    Q = alpha.shape[0]
    tau = np.repeat(alpha.reshape(1, Q), n, axis=0)
    for _ in range(MAX_FIXED_POINT_ITERATIONS):
        previous_tau = np.copy(tau)
        for i in range(n):
            for q in range(Q):
                tau[i, q] = alpha[q]
                for j in range(n):
                    if i != j:
                        for l in range(Q):
                            tau[i, q] *= b(X[i, j], pi[q, l]) ** previous_tau[j, l]
            tau[i, :] /= np.sum(tau[i, :])

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
                if i != j:
                    for l in range(Q):
                        ll += (
                            (1 / 2)
                            * tau[i, q]
                            * tau[j, l]
                            * np.log(b(X[i, j], pi[q, l]))
                        )
    return ll


def parameters_are_ok(alpha, pi, tau):
    if np.abs(np.sum(alpha) - 1) > PRECISION:
        return False
    if np.any(alpha < 0):
        return False
    if np.any(alpha > 1):
        return False
    if np.any(pi < 0):
        return False
    if np.any(pi > 1):
        return False
    if np.any(tau < 0):
        return False
    if np.any(tau > 1):
        return False
    if np.any((np.sum(tau, axis=1) - 1) > PRECISION):
        return False
    if np.any(pi - pi.transpose() > PRECISION):
        return False
    return True


def em_algorithm(X, Q):
    # Initialisation
    tau = np.random.rand(X.shape[0], Q)
    tau = tau / np.sum(tau, axis=1)[:, np.newaxis]
    previous_ll = -np.inf
    # EM algorithm
    for i in range(EM_ITERATIONS):
        alpha, pi = m_step(X, tau)
        tau = e_step(X, alpha, pi)
        ll = log_likelihood(X, alpha, pi, tau)
        print(
            f"After EM iteration {i}/{EM_ITERATIONS} : Log likelihood {ll}...",
            end="",
        )
        if not parameters_are_ok(alpha, pi, tau):
            print(" Values warning.", end="")
            raise ValueError("Parameters are not ok")
        if previous_ll - PRECISION > ll:
            print(" Log likelihood warning.", end="")
            raise ValueError("Log likelihood is decreasing")
        print("Alpha", alpha, end="")
        print("\r", end="", flush=True)
        if np.abs(ll - previous_ll) < PRECISION:
            break
        previous_ll = ll
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
        40,
        2,
        np.array([0.5, 0.5]),
        np.array([[0, 0.1], [0.1, 0.2]]),
    )

    draw_graph(X, Z)

    alpha, pi, tau = em_algorithm(X, 2)
    print(alpha)
    print(pi)
    print(tau)
