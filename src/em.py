import numpy as np

from .constants import *
from .implementations import IMPLEMENTATIONS
from .utils import drop_init, sort_parameters


def em_algorithm(
    X, Q, n_init=10, iterations=100, implementation="pytorch_log", verbose=True
):
    implementation = IMPLEMENTATIONS[implementation]
    if n_init >= iterations:
        raise ValueError(f"n_init should be smaller than {iterations}")
    # Initialisation
    n = X.shape[0]
    X = implementation.input(X)
    tau = [implementation.init_tau(n, Q) for _ in range(n_init)]
    previous_ll = [-1e100 for _ in range(n_init)]
    # EM algorithm
    for i in range(iterations):
        for k in range(n_init):
            alpha, pi = implementation.m_step(X, tau[k])
            alpha, pi = sort_parameters(alpha, pi)
            tau[k] = implementation.e_step(X, alpha, pi)
            ll = implementation.output(
                implementation.log_likelihood(X, alpha, pi, tau[k])
            )

            # Coherence checks
            if not implementation.parameters_are_ok(alpha, pi, tau[k]):
                raise ValueError("Parameters are not ok")
            if previous_ll[k] - PRECISION > ll:
                if ENFORCE_INCREASING_LIKELIHOOD:
                    raise ValueError(
                        f"Log likelihood is decreasing of {previous_ll[k] - ll}"
                    )
                else:
                    print(f"\nLog likelihood is decreasing of {previous_ll[k] - ll}")

            previous_ll[k] = ll
        if verbose:
            print(
                f"After EM iteration {i+1}/{iterations} : Mean log likelihood ({n_init} paths) {np.mean(previous_ll):5f}...",
                end="",
            )
            print("\r", end="")

        # Drop some inits after some time
        if i * n_init > iterations // n_init:
            n_init, tau, previous_ll = drop_init(n_init, tau, previous_ll)
    if verbose:
        print()

    best_init = np.argmax(previous_ll)
    best_tau = tau[best_init]
    best_alpha, best_pi = implementation.m_step(X, best_tau)

    return (
        implementation.output(best_alpha),
        implementation.output(best_pi),
        implementation.output(best_tau),
    )
