# A mixture model for random graphs
# J.-J. Daudin · F. Picard · S. Robin

import numpy as np

from constants import *
from faster import e_step, init_tau, init_X, log_likelihood, m_step, parameters_are_ok
from graphs import draw_graph, random_graph

# n noeuds
# Q classes
# X : n x n
# Z : n x Q
# tau : n x Q
# alpha : Q
# pi : Q x Q


def em_algorithm(X, Q):
    # Initialisation
    n = X.shape[0]
    X = init_X(X)
    tau = init_tau(n, Q)
    previous_ll = -1e100
    # EM algorithm
    for i in range(EM_ITERATIONS):
        alpha, pi = m_step(X, tau)
        tau = e_step(X, alpha, pi)
        ll = log_likelihood(X, alpha, pi, tau)
        print(
            f"After EM iteration {i}/{EM_ITERATIONS} : Log likelihood {ll:5f}...",
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
        if abs(ll - previous_ll) < PRECISION:
            break
        previous_ll = ll
    print()
    return alpha, pi, tau


def sort_parameters(alpha, pi):
    sort_indices = np.argsort(alpha)
    return alpha[sort_indices], pi[sort_indices, :][:, sort_indices]


if __name__ == "__main__":
    # Test the algorithm on a random graph
    Q = 2
    n = 200
    X, Z, alpha, pi = random_graph(n, Q)
    alpha, pi = sort_parameters(alpha, pi)

    draw_graph(X, Z)

    # with profile(
    #    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False
    # ) as prof:
    #    with record_function("em_algorithm"):
    estimated_alpha, estimated_pi, tau = em_algorithm(X, Q)
    estimated_alpha, estimated_pi = sort_parameters(estimated_alpha, estimated_pi)

    print("Estimated alpha", estimated_alpha)
    print("Alpha", alpha)
    print("Estimated pi", estimated_pi)
    print("Pi", pi)

    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
