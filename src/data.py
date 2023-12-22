from typing import Optional, Tuple
import os
import numpy as np
import networkx as nx
import pandas as pd

DATA_PATH = "data/"


def intersect_lists(list1: list, list2: list) -> list:
    """
    Intersection of two lists.

    Parameters
    ----------
    list1 : list
        First list.
    list2 : list
        Second list.

    Returns
    -------
    list
        Intersection of list1 and list2.
    """
    set1 = set(list1)
    set2 = set(list2)
    return list(set1.intersection(set2))


def edge_list_from_reactions_list(
    reactions_file_name: str, edge_list_file_name: str
) -> None:
    """
    Write a file that contains the edge list of a graph from a file that contains the list of reactions.
    Used for generating the Escherichia-Coli dataset.

    Parameters
    ----------
    reactions_file_name : str
        Name of the file that contains the list of reactions.
    edge_list_file_name : str
        Name of the file that will contain the edge list.
    """
    with open(os.path.join(DATA_PATH, reactions_file_name), "r") as reactions_file:
        with open(os.path.join(DATA_PATH, edge_list_file_name), "w") as edge_list_file:
            lines = reactions_file.readlines()[1:]
            edge_list_file.write(f"Nb_nodes:{len(lines)}\n")
            reactions = []
            for i, line in enumerate(lines):
                line.strip("\n")
                if " -> " in line:
                    reactants, products = line.split("  ->  ")
                elif " <--> " in line:
                    reactants, products = line.split("  <-->  ")
                elif " <- " in line:
                    products, reactants = line.split("  <-  ")
                else:
                    print(f"Skipping reaction number {i}...")
                    continue
                reactants = reactants.split(" + ")
                products = products.split(" + ")
                reactions.append([reactants, products])
            n = len(reactions)
            for i in range(n):
                for j in range(i + 1, n):
                    if (
                        len(intersect_lists(reactions[i][0], reactions[j][1])) > 0
                        or len(intersect_lists(reactions[i][1], reactions[j][0])) > 0
                    ):
                        edge_list_file.write(f"{i} {j}\n")


def save_archive_from_graph_and_params(
    X: np.ndarray, Z: np.ndarray, alpha: np.ndarray, pi: np.ndarray, file_name: str
) -> None:
    """
    Save a graph and its parameters in a .npz archive.

    Parameters
    ----------
    X : np.array
        Adjacency matrix of the graph.
    Z : np.array
        Z latent variable.
    alpha : np.array
        alpha parameter.
    pi : np.array
        pi parameter.
    file_name : str
        Name of the archive to save to.
    """
    np.savez(os.path.join(DATA_PATH, file_name), X=X, Z=Z, alpha=alpha, pi=pi)


def graph_and_params_from_archive(
    file_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a graph and its parameters from a .npz archive.

    Parameters
    ----------
    file_name : str
        Name of the archive to load from.

    Returns
    -------
    Tuple[np.array, np.array, np.array, np.array]
        A tuple containing the adjacency matrix, the Z latent variable, the alpha parameter, and the pi parameter.
    """
    with np.load(os.path.join(DATA_PATH, f"{file_name}")) as data:
        X = data["X"]
        Z = data["Z"]
        alpha = data["alpha"]
        pi = data["pi"]
        return X, Z, alpha, pi


def graph_from_edge_list(file_name: str) -> np.ndarray:
    """
    Load a graph from an edge list file.

    Parameters
    ----------
    file_name : str
        Name of the edge list file to load from.

    Returns
    -------
    np.ndarray
        A tuple containing the adjacency matrix, the Z latent variable, the alpha parameter, and the pi parameter.
    """
    with open(os.path.join(DATA_PATH, f"{file_name}"), "r") as file:
        n = int(file.readline().strip("Nb_nodes:"))
        lines = file.readlines()
        for line in lines:
            i, j = line.split(" ")
            i = int(i)
            j = int(j)
        X = np.zeros((n, n))
        for line in lines:
            i, j = line.split(" ")
            i = int(i)
            j = int(j)
            X[i, j] = 1
            X[j, i] = 1
    return X


def generate_SBM_dataset(
    experiment: Optional[int] = 1, n_graphs: Optional[int] = 10
) -> None:
    """
    Generate a dataset of graphs from a Stochastic Block Model and save them in data folder.
    Refer to the experiments.md file in the src/data folder for details on the experiments.

    Parameters
    ----------
    experiment : int, optional
        The experiment number, by default 1.
    n_graphs : int, optional
        The number of graphs to generate, by default 10.
    """
    eps1 = 0.1
    eps2 = 0.01
    a = 0.7
    b = 0.8
    occupation = 0.5

    if experiment == 1:
        n = 30
        Q = 3
        alpha = None  # alpha is random in this experiment
        pi = None  # pi is random in this experiment
    elif experiment == 2:
        n = 500
        Q = 3
        alpha = None  # alpha is random in this experiment
        pi = None  # pi is random in this experiment
    elif experiment == 3:
        n = 150
        Q = 3
        alpha = np.ones(Q) / Q
        pi = eps2 * np.ones((Q, Q)) + (1 - eps1 - eps2) * np.eye(Q)
    elif experiment == 4:
        n = 150
        Q = 5
        alpha = np.ones(Q) / Q
        cluster_densities = np.zeros((Q, Q))
        np.fill_diagonal(cluster_densities, 0.5 + 0.5 * np.random.random(Q))
        pi = eps2 * np.ones((Q, Q)) - eps2 * np.eye(Q) + cluster_densities
    elif experiment == 5:
        n = 150
        Q = 2
        alpha = np.array([9 / 10, 1 / 10])
        pi = np.array([[eps2, a], [a, b]])
    elif experiment == 6:
        n = 150
        Q = 3
        alpha = np.ones(Q) / Q
        pi = (1 - eps1) * np.ones((Q, Q)) + (eps2 - (1 - eps1)) * np.eye(Q)
    else:
        raise ValueError("Invalid experiment number.")

    print("Generating dataset...")
    experiment_path = os.path.join("SBM", f"experiment_{experiment}")
    np.savez(
        os.path.join(DATA_PATH, experiment_path, "params.npz"),
        n_graphs=n_graphs,
        n=n,
        Q=Q,
    )
    for graph_idx in range(n_graphs):
        if experiment in [1, 2]:  # Experiments 1 and 2 involve random params
            alpha = np.random.dirichlet([1.5] * Q)
            pi = np.random.rand(Q, Q)
            pi = pi @ np.transpose(pi)
            pi = pi / np.sum(pi, axis=0) * occupation
        if (n_graphs >= 10) and graph_idx % (n_graphs // 10) == 0:
            print(f"{(100*graph_idx)//n_graphs}% complete...")
        X = np.zeros((n, n))
        Z = np.zeros((n, Q))
        for i in range(n):
            Z[i, :] = np.random.multinomial(1, alpha)
        for i in range(n):
            for j in range(i + 1, n):
                i_class = np.where(Z[i, :] == 1)[0][0]
                j_class = np.where(Z[j, :] == 1)[0][0]
                X[i, j] = np.random.binomial(1, pi[i_class, j_class])
                X[j, i] = X[i, j]
        save_archive_from_graph_and_params(
            X,
            Z,
            alpha,
            pi,
            os.path.join(experiment_path, f"graph_{graph_idx}"),
        )
    print("Done.")


def load_karate_club() -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the karate club dataset.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the adjacency matrix and the labels.
    """
    # Loads the karate network
    G = nx.read_weighted_edgelist(
        "./data/karate.edgelist", delimiter=" ", nodetype=int, create_using=nx.Graph()
    )
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())

    # Loads the class labels
    class_labels = np.loadtxt("./data/karate_labels.txt", delimiter=",", dtype=np.int32)
    idx_to_class_label = dict()
    for i in range(class_labels.shape[0]):
        idx_to_class_label[class_labels[i, 0]] = class_labels[i, 1]

    y = list()
    for node in G.nodes():
        y.append(idx_to_class_label[node])

    y = np.array(y)

    X = nx.adjacency_matrix(G).todense()
    return X, y


def load_cora_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the Cora dataset.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the adjacency matrix and the labels.
    """
    edges_data = "data/cora/cora.cites"
    labels_data = "data/cora/nodes.csv"

    # Edges
    with open(edges_data) as edgelist:
        G = nx.read_edgelist(edgelist, nodetype=int)

    # Labels
    labels = pd.read_csv(labels_data, usecols=["nodeId", "subject"])
    subject_to_label_id = {
        "Case_Based": 0,
        "Genetic_Algorithms": 1,
        "Neural_Networks": 2,
        "Probabilistic_Methods": 3,
        "Reinforcement_Learning": 4,
        "Rule_Learning": 5,
        "Theory": 6,
    }
    assert len(labels) == len(G.nodes())
    for row in labels.itertuples():
        G.nodes[row.nodeId]["label"] = subject_to_label_id[row.subject]
    y = np.array([G.nodes[node]["label"] for node in G.nodes()])

    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())

    X = nx.adjacency_matrix(G).todense()

    return X, y
