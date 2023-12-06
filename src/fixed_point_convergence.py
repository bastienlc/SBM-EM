import torch
from matplotlib import pyplot as plt

from constants import EPSILON
from graphs import random_graph
from opti import DEVICE, fixed_point_iteration, init_tau
from utils import sort_parameters

n = 100
Q = 3


def random_init(n, Q):
    X, Z, alpha, pi = random_graph(n, Q)
    alpha, pi = sort_parameters(alpha, pi)
    tau = init_tau(n, Q)
    return (
        tau,
        torch.tensor(X, device=DEVICE, dtype=torch.float64),
        torch.tensor(alpha, device=DEVICE, dtype=torch.float64),
        torch.tensor(pi, device=DEVICE, dtype=torch.float64),
    )


n_paths = 100
n_iterations = 100

paths = []
for _ in range(n_paths):
    tau, X, alpha, pi = random_init(n, Q)
    path = []
    for _ in range(n_iterations):
        previous_tau = tau.clone()
        tau = fixed_point_iteration(tau, X, alpha, pi)
        path.append(torch.linalg.norm(previous_tau - tau, ord=1).item())
        if path[-1] < EPSILON:
            break
    if len(path) == n_iterations:
        values_to_study = (X, alpha, pi)
    paths.append(path)

for k in range(n_paths):
    plt.plot(list(range(1, len(paths[k]) + 1)), paths[k])

plt.xscale("log")
plt.title("Norm change between each fixed point iteration")
plt.savefig("fixed_point_convergence.png", dpi=600)
plt.close()

# Either we converge very fast or we don't converge at all
# -> Do multiple inits.

# Are we strugling to converge because of the initialization of tau or because of pi and alpha ?
X, alpha, pi = values_to_study
paths = []
for _ in range(n_paths):
    tau = init_tau(n, Q)
    path = []
    for _ in range(n_iterations):
        previous_tau = tau.clone()
        tau = fixed_point_iteration(tau, X, alpha, pi)
        path.append(torch.linalg.norm(previous_tau - tau, ord=1).item())
        if path[-1] < EPSILON:
            break
    paths.append(path)

for k in range(n_paths):
    plt.plot(list(range(1, len(paths[k]) + 1)), paths[k])

plt.xscale("log")
plt.title("Norm change between each fixed point iteration")
plt.savefig("fixed_point_convergence_defavorable.png", dpi=600)
