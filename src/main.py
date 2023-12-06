import numpy as np
import torch

from constants import *
from critertions import draw_critertion
from graphs import draw_graph, random_graph
from opti import e_step, init_tau, init_X, log_likelihood, m_step, parameters_are_ok
from utils import drop_init, sort_parameters


def em_algorithm(X, Q, n_init=10):
    if n_init >= EM_ITERATIONS:
        raise ValueError(f"n_init should be smaller than {EM_ITERATIONS}")
    # Initialisation
    n = X.shape[0]
    X = init_X(X)
    tau = [init_tau(n, Q) for _ in range(n_init)]
    previous_ll = [-1e100 for _ in range(n_init)]
    # EM algorithm
    for i in range(EM_ITERATIONS):
        for k in range(n_init):
            alpha, pi = m_step(X, tau[k])
            alpha, pi = sort_parameters(alpha, pi)
            tau[k] = e_step(X, alpha, pi)
            ll = log_likelihood(X, alpha, pi, tau[k]).item()

            # Coherence checks
            if not parameters_are_ok(alpha, pi, tau[k]):
                raise ValueError("Parameters are not ok")
            if previous_ll[k] - PRECISION > ll:
                raise ValueError("Log likelihood is decreasing")

            previous_ll[k] = ll
        print(
            f"After EM iteration {i}/{EM_ITERATIONS} : Mean log likelihood ({n_init} paths) {np.mean(previous_ll):5f}...",
            end="",
        )
        print("\r", end="", flush=True)

        # Drop some inits after some time
        if i * n_init > EM_ITERATIONS // n_init:
            n_init, tau, previous_ll = drop_init(n_init, tau, previous_ll)
    print()

    best_init = np.argmax(previous_ll)
    best_tau = tau[best_init]
    best_alpha, best_pi = m_step(X, best_tau)

    if isinstance(best_tau, torch.Tensor):
        best_alpha = best_alpha.cpu().detach().numpy()
        best_pi = best_pi.cpu().detach().numpy()
        best_tau = best_tau.cpu().detach().numpy()

    return best_alpha, best_pi, best_tau


if __name__ == "__main__":
    # Test the algorithm on a random graph
    Q = 2
    n = 200
    X, Z, alpha, pi = random_graph(n, Q)
    alpha, pi = sort_parameters(alpha, pi)

    draw_graph(X, Z)
    draw_critertion(n)

    estimated_alpha, estimated_pi, tau = em_algorithm(X, Q, n_init=10)

    print(estimated_alpha)
    print(alpha)
