import os
import numpy as np
import torch
from matplotlib import pyplot as plt

from src.constants import EPSILON
from src.graphs import random_graph
from src.implementations import IMPLEMENTATIONS
from src.utils import sort_parameters
from src.data import graph_and_params_from_archive


IMAGE_PATH = "images/"


def random_init(n, Q, implementation=IMPLEMENTATIONS["pytorch"]):
    X, _, alpha, pi = random_graph(n, Q)
    X, alpha, pi = map(implementation.input, [X, alpha, pi])
    alpha, pi = sort_parameters(alpha, pi)
    tau = implementation.init_tau(n, Q)
    return (tau, X, alpha, pi)


def run_parallel_fixed_point_iterations(
    n_paths=50,
    n_iterations=100,
    X_fixed=None,
    alpha_fixed=None,
    pi_fixed=None,
    tau_fixed=None,
    implementation=IMPLEMENTATIONS["pytorch"],
):
    assert (X_fixed is None and alpha_fixed is None and pi_fixed is None) or (
        X_fixed is not None and alpha_fixed is not None and pi_fixed is not None
    )
    assert not (
        (X_fixed is None and tau_fixed is None)
        or (X_fixed is not None and tau_fixed is not None)
    )
    paths_tau_diff = []
    converged_paths = np.zeros(n_paths, dtype=bool)
    for path in range(n_paths):
        if path % 10 == 0:
            print(f"{path}/{n_paths}")
        if X_fixed is None:
            tau, X, alpha, pi = random_init(n, Q, implementation=implementation)
        else:
            X = X_fixed
            alpha = alpha_fixed
            pi = pi_fixed
        if tau_fixed is None:
            tau = implementation.init_tau(n, Q)
        else:
            tau = tau_fixed
        path_tau_diff = []
        for _ in range(n_iterations):
            previous_tau = tau.clone()
            tau = implementation.fixed_point_iteration(tau, X, alpha, pi)
            path_tau_diff.append(torch.linalg.norm(previous_tau - tau, ord=1).item())
            if path_tau_diff[-1] < EPSILON:
                converged_paths[path] = True
                break
        paths_tau_diff.append(path_tau_diff)

    for k in range(n_paths):
        converged = converged_paths[k]
        plt.plot(
            list(range(1, len(paths_tau_diff[k]) + 1)),
            paths_tau_diff[k],
            color="tab:green" if converged else "tab:red",
        )
    plt.xscale("log")
    plt.title(r"Norm change for $\tau$ between each fixed point iteration")
    plt.xlabel("Iteration")
    plt.ylabel(r"$\|\tau_n - \tau_{n-1}\|$")
    if X_fixed is None:
        plt.savefig(
            os.path.join(IMAGE_PATH, "fixed_point_convergence_fixed_tau.png"), dpi=600
        )
    if tau_fixed is None:
        plt.savefig(
            os.path.join(IMAGE_PATH, "fixed_point_convergence_fixed_X_alpha_pi.png"),
            dpi=600,
        )
    plt.show()
    plt.close()


if __name__ == "__main__":
    IMPLEMENTATION = IMPLEMENTATIONS["pytorch"]

    n = 100
    Q = 3

    X, alpha, pi = None, None, None
    # _, X, alpha, pi = random_init(n, Q, IMPLEMENTATION)
    # tau = None
    tau = IMPLEMENTATION.init_tau(n, Q)

    run_parallel_fixed_point_iterations(
        n_paths=50,
        X_fixed=X,
        alpha_fixed=alpha,
        pi_fixed=pi,
        tau_fixed=tau,
        implementation=IMPLEMENTATION,
    )
