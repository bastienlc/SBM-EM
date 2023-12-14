import matplotlib.pyplot as plt
import numpy as np

from .implementations import IMPLEMENTATIONS
from .em import em_algorithm


def ICL_criterion(Q, X, alpha, pi, tau, implementation="pytorch"):
    implementation = IMPLEMENTATIONS[implementation]
    X, alpha, pi, tau = map(implementation.input, [X, alpha, pi, tau])
    max_ll = implementation.output(
        implementation.log_likelihood(X, alpha, pi, tau, elbo=False)
    )
    n = X.shape[0]
    return (
        max_ll
        - 1 / 2 * Q * (Q + 1) / 2 * np.log(n * (n - 1) / 2)
        - (Q - 1) / 2 * np.log(n)
    )


def draw_criterion(
    X,
    min_Q=1,
    max_Q=20,
    em_n_init=10,
    em_iterations=100,
    implementation="pytorch",
    verbose=True,
):
    y = []
    Q_list = list(range(min_Q, max_Q + 1))
    for Q in Q_list:
        if verbose:
            print(f"Running EM algorithm for Q = {Q}...")
        alpha, pi, tau = em_algorithm(
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

    plt.plot(Q_list, y)
    index = np.argmax(y)
    plt.axvline(x=Q_list[index], color="red")
    plt.title("ICL criterion")
    plt.xlabel("Number of classes")
    plt.ylabel("Criterion")
    plt.show()
