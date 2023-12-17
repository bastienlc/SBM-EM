import os
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx

from .data import graph_and_params_from_archive, DATA_PATH
from .em import em_algorithm, DecreasingLogLikelihoodException
from .metrics import normalized_mutual_information, rand_index, modularity
from .implementations import IMPLEMENTATIONS


RESULTS_PATH = "results/"


def compare_elbos(
    X, alpha, pi, Z, alpha_pred, pi_pred, tau_pred, implementation="pytorch"
):
    implementation = IMPLEMENTATIONS[implementation]
    X, alpha, pi, Z, alpha_pred, pi_pred, tau_pred = map(
        implementation.input, [X, alpha, pi, Z, alpha_pred, pi_pred, tau_pred]
    )
    true_elbo = implementation.log_likelihood(X, alpha, pi, Z, elbo=True)
    pred_elbo = implementation.log_likelihood(
        X, alpha_pred, pi_pred, tau_pred, elbo=True
    )
    return (true_elbo - pred_elbo) / true_elbo


def write_results(
    experiment, elbo_diffs, nmis, rands, true_modularities, pred_modularities
):
    experiment_path = os.path.join("SBM", f"experiment_{experiment}")
    np.savez(
        os.path.join(RESULTS_PATH, experiment_path, "results.npz"),
        elbo_diffs=elbo_diffs,
        nmis=nmis,
        rands=rands,
        true_modularities=true_modularities,
        pred_modularities=pred_modularities,
    )


def write_report(experiment):
    experiment_path = os.path.join("SBM", f"experiment_{experiment}")
    with np.load(os.path.join(RESULTS_PATH, experiment_path, "results.npz")) as results:
        elbo_diffs = results["elbo_diffs"]
        nmis = results["nmis"]
        rands = results["rands"]
        true_modularities = results["true_modularities"]
        pred_modularities = results["pred_modularities"]

        with open(
            os.path.join(RESULTS_PATH, experiment_path, "report.txt"), "w"
        ) as report:
            report.write(f"Experiment {experiment}\n")
            report.write(f"Number of graphs passed: {len(nmis)}\n")
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

            report.write("Metrics with best graph:\n")
            best_graph_elbo = np.argmax(elbo_diffs)
            report.write(f"Best graph for elbo: {best_graph_elbo}\n")
            report.write(f"ELBO difference: {elbo_diffs[best_graph_elbo]}\n")
            best_graph_nmi = np.argmax(nmis)
            report.write(f"Best graph for NMI: {best_graph_nmi}\n")
            report.write(f"NMI: {nmis[best_graph_nmi]}\n")
            best_graph_rand = np.argmax(rands)
            report.write(f"Best graph for Rand index: {best_graph_rand}\n")
            report.write(f"Rand index: {rands[best_graph_rand]}\n")
            best_graph_modularity = np.argmax(
                np.abs(true_modularities) - np.abs(pred_modularities)
            )
            report.write(
                f"Best graph for absolute difference in modularity: {best_graph_modularity}\n"
            )
            report.write(
                f"True modularity: {true_modularities[best_graph_modularity]}\n"
            )
            report.write(
                f"Pred modularity: {pred_modularities[best_graph_modularity]}\n"
            )


def launch_experiment(experiment=1, n_init=10, n_iter=200, implementation="pytorch"):
    experiment_path = os.path.join("SBM", f"experiment_{experiment}")
    with np.load(os.path.join(DATA_PATH, experiment_path, "params.npz")) as params:
        n_graphs = params["n_graphs"]
        n = params["n"]
        Q = params["Q"]

    elbo_diffs = []
    nmis = []
    rands = []
    true_modularities = []
    pred_modularities = []
    for graph_idx in range(n_graphs):
        try:
            if (n_graphs >= 10) and graph_idx % (n_graphs // 10) == 0:
                print(f"{(100*graph_idx)//n_graphs}% complete...")
            X, Z, alpha, pi = graph_and_params_from_archive(
                os.path.join(experiment_path, f"graph_{graph_idx}.npz")
            )
            alpha_pred, pi_pred, tau_pred = em_algorithm(
                X,
                Q=Q,
                n_init=n_init,
                iterations=n_iter,
                implementation=implementation,
                verbose=False,
            )

            # Fitting quality metrics
            elbo_diffs.append(
                IMPLEMENTATIONS[implementation].output(
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

            # Clustering quality metrics
            true_labels = np.argmax(Z, axis=1)
            pred_labels = np.argmax(tau_pred, axis=1)
            nmis.append(normalized_mutual_information(true_labels, pred_labels))
            rands.append(rand_index(true_labels, pred_labels))
            true_clustering = np.array([true_labels == q for q in range(Q)])
            pred_clustering = np.array([pred_labels == q for q in range(Q)])
            true_modularities.append(modularity(X, true_clustering))
            pred_modularities.append(modularity(X, pred_clustering))
        except DecreasingLogLikelihoodException:
            print("!! Skipped one graph due to decreasing log likelihood !!")
            continue

    elbo_diffs = np.array(elbo_diffs)
    nmis = np.array(nmis)
    rands = np.array(rands)
    true_modularities = np.array(true_modularities)
    pred_modularities = np.array(pred_modularities)
    write_results(
        experiment,
        elbo_diffs,
        nmis,
        rands,
        true_modularities,
        pred_modularities,
    )


def initialization_sensitivity(n_init, n_iter=100, implementation="pytorch"):
    # edges_data = "data/cora/cora.cites"

    # with open(edges_data) as edgelist:
    #     G = nx.read_edgelist(edgelist)
    # X = nx.adjacency_matrix(G).todense()
    # Q = 7

    X, Z, _, _ = graph_and_params_from_archive(f"sample_graph.npz")
    Q = Z.shape[1]
    for i in range(n_init):
        print(f"Initialization {i+1}/{n_init}...")
        ll_log = em_algorithm(
            X,
            Q=Q,
            n_init=1,
            iterations=n_iter,
            implementation=implementation,
            verbose=False,
            diagnostic=True,
        )
        plt.plot(np.arange(n_iter), ll_log)
    plt.xlabel("EM iteration")
    plt.xscale("log")
    plt.ylabel("Log likelihood")
    plt.title(f"Log likelihood for {n_init} initializations")
    plt.show()
