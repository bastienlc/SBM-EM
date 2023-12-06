import numpy as np
import os



DATA_PATH = "data/"


def intersect_lists(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return list(set1.intersection(set2))


def edge_list_from_reactions_list(reactions_file_name, edge_list_file_name):
    with open(os.path.join(DATA_PATH, reactions_file_name), 'r') as reactions_file:
        with open(os.path.join(DATA_PATH, edge_list_file_name), 'w') as edge_list_file:
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
                    if len(intersect_lists(reactions[i][0], reactions[j][1])) > 0 or len(intersect_lists(reactions[i][1], reactions[j][0])) > 0:
                        edge_list_file.write(f"{i} {j}\n")


def save_archive_from_graph_and_params(X, Z, alpha, pi, file_name):
    np.savez(os.path.join(DATA_PATH, file_name), X=X, Z=Z, alpha=alpha, pi=pi)
        
        
def graph_and_params_from_archive(filename):
    with np.load(os.path.join(DATA_PATH, f"{filename}")) as data:
        X = data["X"]
        Z = data["Z"]
        alpha = data["alpha"]
        pi = data["pi"]
        return X, Z, alpha, pi


def graph_from_edge_list(filename):
    with open(os.path.join(DATA_PATH, f"{filename}"), "r") as file:
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
