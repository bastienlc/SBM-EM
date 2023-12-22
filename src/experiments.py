import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import itertools
import networkx as nx
from graspologic.plot import adjplot
import pandas as pd
import pickle

from .data import graph_and_params_from_archive, DATA_PATH
from .em import em_algorithm
from .metrics import (
    normalized_mutual_information,
    rand_index,
    modularity,
    clustering_coefficient,
)
from .implementations import get_implementation


RESULTS_PATH = "results/"


def rearrange_tau(q, true_classes_clusters, Q):
    # Rearrange q such that the number of misclassified nodes is minimized
    if isinstance(q, torch.Tensor):
        pred_clusters = np.argmax(q.detach().numpy(), axis=1)
    else:
        pred_clusters = np.argmax(q, axis=1)
    # Store all possible permutations of the classes
    permutations = list()
    for permutation in itertools.permutations(range(Q)):
        permutations.append(permutation)

    # Compute the number of misclassified nodes for each permutation
    misclassified_nodes = list()
    for permutation in permutations:
        misclassified_nodes.append(
            np.sum(pred_clusters != [permutation[i] for i in true_classes_clusters])
        )
    # Find the permutation that minimizes the number of misclassified nodes
    best_permutation = permutations[np.argmin(misclassified_nodes)]

    # Rearrange q according to the best permutation
    new_q = np.zeros_like(q)
    for i in range(Q):
        new_q[:, i] = q[:, best_permutation[i]]

    return new_q


def param_distance(alpha_pred, pi_pred, alpha, pi, Q):
    best_distance = np.infty
    for permutation in itertools.permutations(range(Q)):
        alpha_perm = np.zeros(Q)
        for i in range(Q):
            alpha_perm[i] = alpha_pred[permutation[i]]
        pi_perm = np.zeros((Q, Q))
        for i in range(Q):
            for j in range(Q):
                pi_perm[i, j] = pi_pred[permutation[i], permutation[j]]
        dist_alpha = np.linalg.norm(alpha_perm - alpha) / Q
        dist_pi = np.linalg.norm(pi_perm - pi) / (Q**2)
        if dist_alpha + dist_pi < best_distance:
            best_distance = dist_alpha + dist_pi
    return best_distance


# Experiments on SBM dataset


def compare_elbos(
    X, alpha, pi, Z, alpha_pred, pi_pred, tau_pred, implementation="pytorch"
):
    implementation = get_implementation(implementation)
    X, alpha, pi, Z, alpha_pred, pi_pred, tau_pred = map(
        implementation.input, [X, alpha, pi, Z, alpha_pred, pi_pred, tau_pred]
    )
    true_elbo = implementation.log_likelihood(X, alpha, pi, Z, elbo=True)
    pred_elbo = implementation.log_likelihood(
        X, alpha_pred, pi_pred, tau_pred, elbo=True
    )
    return (true_elbo - pred_elbo) / true_elbo


def write_SBM_pred_results(alpha, pi, tau, path, graph):
    np.savez(
        os.path.join(RESULTS_PATH, path, f"graph_{graph}_results.npz"),
        alpha=alpha,
        pi=pi,
        tau=tau,
    )


def load_SBM_pred_results(path, graph):
    with np.load(
        os.path.join(RESULTS_PATH, path, f"graph_{graph}_results.npz")
    ) as data:
        alpha = data["alpha"]
        pi = data["pi"]
        tau = data["tau"]
    return alpha, pi, tau


def write_report(experiment, implementation="pytorch"):
    experiment_path = os.path.join("SBM", f"experiment_{experiment}")
    with np.load(os.path.join(DATA_PATH, experiment_path, "params.npz")) as params:
        n_graphs = params["n_graphs"]
        n = params["n"]
        Q = params["Q"]
    passed_graphs = np.load(
        os.path.join(RESULTS_PATH, experiment_path, "passed_graphs.npy")
    )

    elbo_diffs = []
    param_distances = []
    nmis = []
    rands = []
    true_modularities = []
    pred_modularities = []
    for graph_idx in range(n_graphs):
        if graph_idx in passed_graphs:
            X, Z, alpha, pi = graph_and_params_from_archive(
                os.path.join(experiment_path, f"graph_{graph_idx}.npz")
            )
            alpha_pred, pi_pred, tau_pred = load_SBM_pred_results(
                experiment_path, graph_idx
            )

            # Fitting quality metrics
            elbo_diffs.append(
                get_implementation(implementation).output(
                    compare_elbos(
                        X,
                        alpha,
                        pi,
                        Z,
                        alpha_pred,
                        pi_pred,
                        tau_pred,
                        implementation=implementation,
                    )
                )
            )
            param_distances.append(param_distance(alpha_pred, pi_pred, alpha, pi, Q))

            # Clustering quality metrics
            true_labels = np.argmax(Z, axis=1)
            pred_labels = np.argmax(tau_pred, axis=1)
            nmis.append(normalized_mutual_information(true_labels, pred_labels))
            rands.append(rand_index(true_labels, pred_labels))
            true_clustering = np.array([true_labels == q for q in range(Q)])
            pred_clustering = np.array([pred_labels == q for q in range(Q)])
            true_modularities.append(modularity(X, true_clustering))
            pred_modularities.append(modularity(X, pred_clustering))
    param_distances = np.array(param_distances)
    elbo_diffs = np.array(elbo_diffs)
    nmis = np.array(nmis)
    rands = np.array(rands)
    true_modularities = np.array(true_modularities)
    pred_modularities = np.array(pred_modularities)

    with open(os.path.join(RESULTS_PATH, experiment_path, "report.txt"), "w") as report:
        report.write(f"Experiment {experiment}\n")
        report.write(f"Number of graphs passed: {len(passed_graphs)}\n")
        report.write(
            f"Params distance: {np.mean(param_distances)} +/- {np.std(param_distances)}\n"
        )
        report.write(
            f"ELBO difference: {np.mean(elbo_diffs)} +/- {np.std(elbo_diffs)}\n"
        )
        report.write(f"NMI: {np.mean(nmis)} +/- {np.std(nmis)}\n")
        report.write(f"Rand index: {np.mean(rands)} +/- {np.std(rands)}\n")
        report.write(
            f"True modularity: {np.mean(true_modularities)} +/- {np.std(true_modularities)}\n"
        )
        report.write(
            f"Pred modularity: {np.mean(pred_modularities)} +/- {np.std(pred_modularities)}\n"
        )
        report.write("\n")

        report.write("Metrics with best graphs:\n")
        best_graph_param_distance = np.argmin(param_distances)
        best_graph_param_distance_idx = passed_graphs[best_graph_param_distance]
        report.write(
            f"Best graph for params distance: {best_graph_param_distance_idx}\n"
        )
        report.write(f"Params distance: {param_distances[best_graph_param_distance]}\n")
        best_graph_elbo = np.argmax(elbo_diffs)
        best_graph_elbo_idx = passed_graphs[best_graph_elbo]
        report.write(f"Best graph for elbo: {best_graph_elbo_idx}\n")
        report.write(f"ELBO difference: {elbo_diffs[best_graph_elbo]}\n")
        best_graph_nmi = np.argmax(nmis)
        best_graph_nmi_idx = passed_graphs[best_graph_nmi]
        report.write(f"Best graph for NMI: {best_graph_nmi_idx}\n")
        report.write(f"NMI: {nmis[best_graph_nmi]}\n")
        best_graph_rand = np.argmax(rands)
        best_graph_rand_idx = passed_graphs[best_graph_rand]
        report.write(f"Best graph for Rand index: {best_graph_rand_idx}\n")
        report.write(f"Rand index: {rands[best_graph_rand]}\n")
        best_graph_gt_modularity = np.argmax(true_modularities)
        best_graph_gt_modularity_idx = passed_graphs[best_graph_gt_modularity]
        report.write(f"Best graph for gt modularity: {best_graph_gt_modularity_idx}\n")
        report.write(
            f"True modularity: {true_modularities[best_graph_gt_modularity]}\n"
        )
        report.write(
            f"Pred modularity: {pred_modularities[best_graph_gt_modularity]}\n"
        )
        best_graph_pred_modularity = np.argmax(pred_modularities)
        best_graph_pred_modularity_idx = passed_graphs[best_graph_pred_modularity]
        report.write(
            f"Best graph for gt modularity: {best_graph_pred_modularity_idx}\n"
        )
        report.write(
            f"True modularity: {true_modularities[best_graph_pred_modularity]}\n"
        )
        report.write(
            f"Pred modularity: {pred_modularities[best_graph_pred_modularity]}\n"
        )


def launch_experiment(experiment=1, n_init=5, n_iter=100, implementation="pytorch"):
    experiment_path = os.path.join("SBM", f"experiment_{experiment}")
    with np.load(os.path.join(DATA_PATH, experiment_path, "params.npz")) as params:
        n_graphs = params["n_graphs"]
        n = params["n"]
        Q = params["Q"]

    passed_graphs = []

    for graph_idx in range(n_graphs):
        try:
            if (n_graphs >= 10) and graph_idx % (n_graphs // 10) == 0:
                print(f"{(100*graph_idx)//n_graphs}% complete...")
            X, Z, _, _ = graph_and_params_from_archive(
                os.path.join(experiment_path, f"graph_{graph_idx}.npz")
            )
            alpha_pred, pi_pred, tau_pred, _ = em_algorithm(
                X,
                Q=Q,
                n_init=n_init,
                iterations=n_iter,
                implementation=implementation,
                verbose=False,
            )
            passed_graphs.append(graph_idx)
            tau_pred = rearrange_tau(tau_pred, np.argmax(Z, axis=1), Q)
            write_SBM_pred_results(
                alpha=alpha_pred,
                pi=pi_pred,
                tau=tau_pred,
                path=experiment_path,
                graph=graph_idx,
            )
        except:
            print("!! Skipped one graph due to error in em algorithm !!")
            continue

    np.save(
        os.path.join(RESULTS_PATH, experiment_path, "passed_graphs.npy"), passed_graphs
    )


# Experiments on real datasets


def show_karate_gt_vs_prediction(G, tau, y):
    tau = rearrange_tau(tau, y, 2)

    color_list = ["g", "c", "r"]
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(
        "Karate club ; NMI: {:.3f}".format(
            normalized_mutual_information(y, np.argmax(tau, axis=1))
        ),
        fontsize=20,
    )
    fig.set_facecolor("w")
    fig.valign = "center"
    fig.halign = "center"
    pos = nx.spring_layout(G, seed=42)
    node_colors = [color_list[i] for i in y]
    nx.draw_networkx(G, node_color=node_colors, ax=ax[0], pos=pos)
    ax[0].set_title("Ground truth")
    node_colors = np.argmax(tau, axis=1)
    node_colors = [color_list[i] for i in node_colors]
    nx.draw_networkx(G, node_color=node_colors, ax=ax[1], pos=pos)
    ax[1].set_title("Predicted")
    node_colors = np.argmax(tau, axis=1)
    node_colors[node_colors != y] = 2
    node_colors = [color_list[i] for i in node_colors]
    nx.draw_networkx(G, node_color=node_colors, ax=ax[2], pos=pos)
    ax[2].set_title("Misclassified")
    plt.show()


def report_metrics(X, tau, y, Q, full_clustering_coeff=True):
    tau = rearrange_tau(tau, y, Q)
    pred_labels = np.argmax(tau, axis=1)
    print("NMI: {:.3f}".format(normalized_mutual_information(y, pred_labels)))
    print("Rand index: {:.3f}".format(rand_index(y, pred_labels)))
    gt_clustering = np.array([y == q for q in range(Q)])
    pred_clustering = np.array([pred_labels == q for q in range(Q)])
    print("Gt Modularity: {:.3f}".format(modularity(X, gt_clustering)))
    print("Pred Modularity: {:.3f}".format(modularity(X, pred_clustering)))
    if full_clustering_coeff:
        print("Graph clustering coefficient:", clustering_coefficient(X, None))
    print(
        "Per class gt clustering coefficients:",
        [clustering_coefficient(X, gt_clustering[q]) for q in range(Q)],
    )
    print(
        "Per class pred clustering coefficients:",
        [clustering_coefficient(X, pred_clustering[q]) for q in range(Q)],
    )


def draw_dot_plot(X, classification, ground_truth, save_as=None):
    meta = pd.DataFrame(
        data={
            "Class": classification,
            "True class": ground_truth,
        },
    )
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    adjplot(
        data=X, ax=ax, meta=meta, plot_type="scattermap", group=["Class"], group_order = ["Class"], ticks=True, color = "True class", palette = "tab10"
    )
    if save_as is not None:
        plt.savefig(f"images/{save_as}.png")
    plt.show()


def write_em_results(alpha, pi, tau, path):
    with open(os.path.join(RESULTS_PATH, path, "alpha.pkl"), "wb") as f:
        pickle.dump(alpha, f)
    with open(os.path.join(RESULTS_PATH, path, "pi.pkl"), "wb") as f:
        pickle.dump(pi, f)
    with open(os.path.join(RESULTS_PATH, path, "tau.pkl"), "wb") as f:
        pickle.dump(tau, f)


def load_em_results(path):
    with open(os.path.join(RESULTS_PATH, path, "alpha.pkl"), "rb") as f:
        alpha = pickle.load(f)
    with open(os.path.join(RESULTS_PATH, path, "pi.pkl"), "rb") as f:
        pi = pickle.load(f)
    with open(os.path.join(RESULTS_PATH, path, "tau.pkl"), "rb") as f:
        tau = pickle.load(f)
    return alpha, pi, tau
