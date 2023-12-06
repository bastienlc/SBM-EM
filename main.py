import numpy as np
import torch

from src.constants import *
from src.graphs import draw_graph, random_graph
from src.faster import e_step, init_tau, init_X, log_likelihood, m_step, parameters_are_ok


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

    if isinstance(alpha, torch.Tensor):
        alpha = alpha.cpu().detach().numpy()
        pi = pi.cpu().detach().numpy()
        tau = tau.cpu().detach().numpy()

    return alpha, pi, tau


def sort_parameters(alpha, pi):
    sort_indices = np.argsort(alpha)
    return alpha[sort_indices], pi[sort_indices, :][:, sort_indices]


if __name__ == "__main__":
    # Test the algorithm on a random graph
    Q = 2
    n = 10
    X, Z, alpha, pi = random_graph(n, Q)
    alpha, pi = sort_parameters(alpha, pi)

    draw_graph(X, Z)

    estimated_alpha, estimated_pi, tau = em_algorithm(X, Q)
    estimated_alpha, estimated_pi = sort_parameters(estimated_alpha, estimated_pi)

    print("Estimated alpha", estimated_alpha)
    print("Alpha", alpha)
    print("Estimated pi", estimated_pi)
    print("Pi", pi)
