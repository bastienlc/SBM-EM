from typing import Optional, Tuple, Union

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
    """
    Performs one iteration of the EM algorithm.

    Parameters
    ----------
    implementation : GenericImplementation
        The implementation to use.
    X : Union[np.ndarray, torch.Tensor]
        The adjacency matrix of the graph.
    tau : Union[np.ndarray, torch.Tensor]
        The current tau.

    Returns
    -------
    Tuple[Union[np.ndarray, torch.Tensor], float]
        A tuple containing the updated tau and the log likelihood.
    """
    alpha, pi = implementation.m_step(X, tau)
    alpha, pi = sort_parameters(alpha, pi)
    tau = implementation.e_step(X, alpha, pi)
    ll = implementation.output(implementation.log_likelihood(X, alpha, pi, tau))

    if not implementation.check_parameters(alpha, pi, tau):
        raise ValueError("Parameters are not ok")

    return tau, ll


def em_algorithm(
    X: np.ndarray,
    Q: int,
    n_init: Optional[int] = 10,
    iterations: Optional[int] = 100,
    implementation: Optional[str] = "pytorch_log",
    verbose: Optional[bool] = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs the EM algorithm.

    Parameters
    ----------
    X : np.ndarray
        The adjacency matrix of the graph.
    Q : int
        The number of clusters.
    n_init : int, optional
        The number of runs to perform in parallel, by default 10.
    iterations : int, optional
        The number of EM iterations, by default 100.
    implementation : str, optional
        The implementation to use, by default "pytorch_log".
    verbose : bool, optional
        Whether to print the log likelihood at each iteration, by default True.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the estimated alpha, pi, tau, and log likelihoods.
    """
    try:
        # Initialization
        impl = get_implementation(implementation)
        n = X.shape[0]
        X = impl.input(X)
        taus = [impl.init_tau(n, Q) for _ in range(n_init)]
        log_likelihoods = np.full((n_init, iterations), -np.inf)
        inits = list(range(n_init))

        best_tau, best_ll = None, -np.inf

        # EM algorithm
        for i in range(iterations):  # Iterations of the EM algorithm
            for init in inits:  # Run each initialisation
                inits_to_drop = []

                taus[init], log_likelihoods[init, i] = em_iteration(impl, X, taus[init])

                if ENFORCE_INCREASING_LIKELIHOOD and (
                    log_likelihoods[init, i - 1] - PRECISION > log_likelihoods[init, i]
                ):
                    inits_to_drop.append(init)

                if (
                    log_likelihoods[init, i] > best_ll
                ):  # Keep track of the best likelihood so far
                    best_tau, best_ll = taus[init], log_likelihoods[init, i]

            inits = drop_inits(
                inits,
                i,
                iterations,
                log_likelihoods,
                inits_to_drop=inits_to_drop,
            )  # Drop the initialisations that are not performing well, as well as the ones in inits_to_drop

            if verbose:
                print(
                    f"EM iteration {i+1}/{iterations} | Max LL ({len(inits)} path{'s' if n_init>1 else ''}) {np.max(log_likelihoods[inits, i]):5f}",
                    end="",
                )
                print("\r", end="")

        if verbose:
            print("")

        # If the algorithm ran its course, return the last tau obtained and the associated alpha and pi
        tau = taus[inits[0]]
        alpha, pi = impl.m_step(X, tau)

        return (
            impl.output(alpha),
            impl.output(pi),
            impl.output(tau),
            log_likelihoods,
        )
    except (
        KeyboardInterrupt
    ):  # If the user interrupts the algorithm, return the best parameters so far
        if verbose:
            print("")
        alpha, pi = impl.m_step(X, best_tau)
        return (
            impl.output(alpha),
            impl.output(pi),
            impl.output(best_tau),
            log_likelihoods,
        )
