import torch
from matplotlib import pyplot as plt

from src.constants import EPSILON
from src.graphs import random_graph
from src.implementations import IMPLEMENTATIONS
from src.utils import sort_parameters
from src.data import graph_and_params_from_archive


IMPLEMENTATION = IMPLEMENTATIONS["pytorch"]


n = 300
Q = 3
# X, Z, alpha, pi = graph_and_params_from_archive(f"sample_graph.npz")
# X, Z, alpha, pi = map(IMPLEMENTATION.input, [X, Z, alpha, pi])
# # X is a graph with 300 nodes and 3 communities, with
# # alpha = [0.33333333 0.33333333 0.33333333]
# # and pi = [[0.9  0.02 0.02]
# #           [0.02 0.9  0.02]
# #           [0.02 0.02 0.9 ]]


def random_init(n, Q):
    X, Z, alpha, pi = random_graph(n, Q)
    X, Z, alpha, pi = map(IMPLEMENTATION.input, [X, Z, alpha, pi])
    alpha, pi = sort_parameters(alpha, pi)
    tau = IMPLEMENTATION.init_tau(n, Q)
    return (tau, X, alpha, pi)


n_paths = 50
n_iterations = 100


torch.random.manual_seed(1)
paths_tau_diff = []
paths_elbo = []
for path in range(n_paths):
    if path % 10 == 0:
        print(f"{path}/{n_paths}")
    tau, X, alpha, pi = random_init(n, Q)
    # tau = IMPLEMENTATION.init_tau(n, Q)
    path_tau_diff = []
    path_elbo = []
    for _ in range(n_iterations):
        previous_tau = tau.clone()
        tau = IMPLEMENTATION.fixed_point_iteration(tau, X, alpha, pi)
        path_tau_diff.append(torch.linalg.norm(previous_tau - tau, ord=1).item())
        path_elbo.append(IMPLEMENTATION.log_likelihood(X, alpha, pi, tau).item())
        if path_tau_diff[-1] < EPSILON:
            break
    if len(path_tau_diff) == n_iterations:
        values_to_study = (X, alpha, pi)
    paths_tau_diff.append(path_tau_diff)
    paths_elbo.append(path_elbo)

for k in range(n_paths):
    plt.plot(list(range(1, len(paths_tau_diff[k]) + 1)), paths_tau_diff[k])
plt.xscale("log")
plt.title("Norm change between each fixed point iteration")
# plt.savefig("fixed_point_convergence.png", dpi=600)
plt.savefig("test_tau.png", dpi=600)
plt.close()

for k in range(n_paths):
    plt.plot(list(range(1, len(paths_elbo[k]) + 1)), paths_elbo[k])
plt.xscale("log")
plt.title("Elbo")
# plt.savefig("fixed_point_convergence.png", dpi=600)
plt.savefig("test_elbo.png", dpi=600)


# Either we converge very fast or we don't converge at all
# -> Do multiple inits.

# # Are we strugling to converge because of the initialization of tau or because of pi and alpha ?
# X, alpha, pi = values_to_study
# paths = []
# for _ in range(n_paths):
#     tau = init_tau(n, Q)
#     path_tau_diff = []
#     for _ in range(n_iterations):
#         previous_tau = tau.clone()
#         tau = fixed_point_iteration(tau, X, alpha, pi)
#         path_tau_diff.append(torch.linalg.norm(previous_tau - tau, ord=1).item())
#         if path_tau_diff[-1] < EPSILON:
#             break
#     paths.append(path_tau_diff)

# for k in range(n_paths):
#     plt.plot(list(range(1, len(paths[k]) + 1)), paths[k])

# plt.xscale("log")
# plt.title("Norm change between each fixed point iteration")
# plt.savefig("fixed_point_convergence_defavorable.png", dpi=600)
