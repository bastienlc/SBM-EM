from typing import Tuple, Union

import numpy as np
import torch

from .constants import *
from .implementations import GenericImplementation, get_implementation
from .utils import drop_inits, sort_parameters


def em_iteration(
    implementation: GenericImplementation,
    X: Union[np.ndarray, torch.Tensor],
    tau: Union[np.ndarray, torch.Tensor],
) -> Tuple[Union[np.ndarray, torch.Tensor], float]:
    """Performs one iteration of the EM algorithm
    Args:
        implementation: the implementation to use
        X: the adjacence matrix of the graph
        tau: the current tau
    Returns:
        tau: the updated tau
        ll: the log likelihood
    """
    alpha, pi = implementation.m_step(X, tau)
    alpha, pi = sort_parameters(alpha, pi)
    tau = implementation.e_step(X, alpha, pi)
    ll = implementation.output(implementation.log_likelihood(X, alpha, pi, tau))

    if not implementation.parameters_are_ok(alpha, pi, tau):
        raise ValueError("Parameters are not ok")

    return tau, ll


def em_algorithm(
    X: np.ndarray,
    Q: int,
    n_init: int = 10,
    iterations: int = 100,
    implementation: str = "pytorch_log",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Runs the EM algorithm
    Args:
        X: the adjacence matrix of the graph
        Q: the number of clusters
        n_init: the number of runs to perform in parallel
        iterations: the number of EM iterations
        implementation: the implementation to use
        verbose: whether to print the log likelihood at each iteration
    Returns:
        alpha: the estimated alpha
        pi: the estimated pi
        tau: the estimated tau
    """
    # Initialisation
    impl = get_implementation(implementation)
    n = X.shape[0]
    X = impl.input(X)
    taus = [impl.init_tau(n, Q) for _ in range(n_init)]
    log_likelihoods = np.full((n_init, iterations), -np.inf)
    inits = list(range(n_init))

    # EM algorithm
    for i in range(iterations):
        for init in inits:
            inits_to_drop = []

            taus[init], log_likelihoods[init, i] = em_iteration(impl, X, taus[init])

            if ENFORCE_INCREASING_LIKELIHOOD and (
                log_likelihoods[init, i - 1] - PRECISION > log_likelihoods[init, i]
            ):
                inits_to_drop.append(init)

        inits = drop_inits(
            inits,
            i,
            iterations,
            log_likelihoods,
            inits_to_drop=inits_to_drop,
        )

        if verbose:
            print(
                f"EM iteration {i+1}/{iterations} | Max LL ({len(inits)} path{'s' if n_init>1 else ''}) {np.max(log_likelihoods[inits, i]):5f}",
                end="",
            )
            print("\r", end="")

    if verbose:
        print("")

    tau = taus[inits[0]]
    alpha, pi = impl.m_step(X, tau)

    return (
        impl.output(alpha),
        impl.output(pi),
        impl.output(tau),
        log_likelihoods,
    )
