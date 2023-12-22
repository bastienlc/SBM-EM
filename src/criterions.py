from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .em import em_algorithm
from .implementations import get_implementation


def ICL_criterion(
    Q: int,
    X: np.ndarray,
    alpha: np.ndarray,
    pi: np.ndarray,
    tau: np.ndarray,
    implementation: str = "pytorch_log",
) -> float:
    """
    Computes the ICL (Integrated Complete-data Likelihood) criterion.

    Parameters
    ----------
    Q : int
        The number of classes.
    X : np.ndarray
        The adjacency matrix of the graph.
    alpha : np.ndarray
        Estimated alpha.
    pi : np.ndarray
        Estimated pi.
    tau : np.ndarray
        Estimated tau.
    implementation : str, optional
        The implementation to use, by default "pytorch".

    Returns
    -------
    float
        The ICL criterion value.
    """
    impl = get_implementation(implementation)
    X, alpha, pi, tau = map(impl.input, [X, alpha, pi, tau])
    max_ll = impl.output(impl.log_likelihood(X, alpha, pi, tau, elbo=False))
    n = X.shape[0]
    return (
        max_ll
        - 1 / 2 * Q * (Q + 1) / 2 * np.log(n * (n - 1) / 2)
        - (Q - 1) / 2 * np.log(n)
    )


def draw_criterion(
    X: np.ndarray,
    Q_to_test: np.array,
    em_n_init: int = 10,
    em_iterations: int = 100,
    implementation: str = "pytorch_log",
    verbose: Optional[bool] = True,
    save_as: Optional[str] = None,
) -> None:
    """
    Draws the ICL criterion plot for different numbers of classes.

    Parameters
    ----------
    X : np.ndarray
        The adjacency matrix of the graph.
    Q_to_test : np.array
        List of values of Q to test.
    em_n_init : int, optional
        The number of runs to perform in parallel during EM algorithm, by default 10.
    em_iterations : int, optional
        The number of EM iterations, by default 100.
    implementation : str, optional
        The implementation to use, by default "pytorch".
    verbose : bool, optional
        Whether to print progress information, by default True.
    save_as: str, optional
        The name under which to save the figure. If not provided, figure is not saved.

    Returns
    -------
    None
    """
    y = []
    for Q in Q_to_test:
        if verbose:
            print(f"Running EM algorithm for Q = {Q}...")
        alpha, pi, tau, _ = em_algorithm(
            X,
            Q,
            n_init=em_n_init,
            iterations=em_iterations,
            implementation=implementation,
            verbose=verbose,
        )
        y.append(ICL_criterion(Q, X, alpha, pi, tau, implementation=implementation))
        if verbose:
            print(f"ICL criterion for Q = {Q} : {y[-1]}\n")

    plt.plot(Q_to_test, y)
    index = np.argmax(y)
    plt.axvline(x=Q_to_test[index], color="red")
    plt.title("ICL criterion")
    plt.xlabel("Number of classes")
    plt.ylabel("ICL")
    if save_as is not None:
        plt.savefig(f"images/{save_as}")
    plt.show()
