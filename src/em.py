import numpy as np

from .constants import *
from .implementations import IMPLEMENTATIONS
from .utils import drop_init, sort_parameters


class DecreasingLogLikelihoodException(Exception):
    pass


def em_algorithm(
    X,
    Q,
    n_init=10,
    iterations=100,
    implementation="pytorch_log",
    verbose=True,
    diagnostic=False,
):
    implementation = IMPLEMENTATIONS[implementation]
    if n_init >= iterations:
        raise ValueError(f"n_init should be smaller than {iterations}")
    # Initialisation
    n = X.shape[0]
    X = implementation.input(X)
    tau = [implementation.init_tau(n, Q) for _ in range(n_init)]
    previous_ll = [-1e100 for _ in range(n_init)]
    if diagnostic:
        assert n_init == 1, "diagnostic mode only works with one initialization"
        ll_log = np.zeros(iterations)
    # EM algorithm
    for i in range(iterations):
        for k in range(n_init):
            alpha, pi = implementation.m_step(X, tau[k])
            alpha, pi = sort_parameters(alpha, pi)
            tau[k] = implementation.e_step(X, alpha, pi)
            ll = implementation.output(
                implementation.log_likelihood(X, alpha, pi, tau[k])
            )
            if diagnostic:
                ll_log[i] = ll

            # Coherence checks
            if not implementation.parameters_are_ok(alpha, pi, tau[k]):
                raise ValueError("Parameters are not ok")
            if previous_ll[k] - PRECISION > ll:
                raise DecreasingLogLikelihoodException("Log likelihood is decreasing")

            previous_ll[k] = ll
        if verbose:
            print(
                f"After EM iteration {i+1}/{iterations} : Mean log likelihood ({n_init} paths) {np.mean(previous_ll):5f}...",
                end="",
            )
            print("\r", end="", flush=True)

        # Drop some inits after some time
        if i * n_init > iterations // n_init:
            n_init, tau, previous_ll = drop_init(n_init, tau, previous_ll)
    if verbose:
        print()

    best_init = np.argmax(previous_ll)
    best_tau = tau[best_init]
    best_alpha, best_pi = implementation.m_step(X, best_tau)

    if diagnostic:
        return ll_log
    else:
        return (
            implementation.output(best_alpha),
            implementation.output(best_pi),
            implementation.output(best_tau),
        )
