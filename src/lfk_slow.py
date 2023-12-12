import numpy as np


def internal_degree(X, community):
    return np.sum(X[community, :][:, community])


def external_degree(X, community):
    return np.sum(X[community, :][:, np.logical_not(community)])


def fitness(X, community, alpha=1):
    """cluster is a mask indicating what nodes are in the considered cluster"""
    k_in = internal_degree(X, community)
    k_out = external_degree(X, community)
    return k_in / ((k_in + k_out) ** alpha)


def node_fitness(X, community, node, alpha=1):
    original_value = community[node]
    community[node] = True
    fit_with = fitness(X, community, alpha)
    community[node] = False
    fit_without = fitness(X, community, alpha)
    community[node] = original_value
    return fit_with - fit_without


def update_neighbors(X, community, neighbors, node):
    for neighbor in np.where(X[node])[0]:
        if not community[neighbor]:
            neighbors.add(neighbor)


def grow_community(X, community, neighbors, alpha=1.0):
    """Grow the community, adding the neighbor of maximal fitness. Returns False if no neighbor has a positive fitness.
    Args:
        X (np.array): adjacency matrix of the graph
        community (np.array): array of booleans indicating what nodes are in the considered community
        neighbors (np.array): array of indices of nodes adjacent to the community
        alpha (float, optional): Alpha exponent for the fitness. Defaults to 1.
    """
    best_neighbor, best_fit = None, -np.inf
    for neighbor in neighbors:
        fit = node_fitness(X, community, neighbor, alpha)
        if fit > best_fit:
            best_neighbor, best_fit = neighbor, fit
    if best_fit <= 0:
        return False
    community[best_neighbor] = True
    neighbors.remove(best_neighbor)
    update_neighbors(X, community, neighbors, best_neighbor)
    return True


def clean_community(X, community, alpha=1.0):
    """Remove nodes that have a negative fitness from the community."""
    exit = False
    while not exit:
        exit = True
        for node in np.where(community)[0]:
            if node_fitness(X, community, node, alpha) < 0:
                community[node] = False
                exit = False


def natural_community(X, node, alpha=1.0):
    """Compute the natural community of a node, starting from the isolated node and growing it iteratively with neighbors of maximal fitness."""
    community = np.zeros(X.shape[0], dtype=bool)
    community[node] = True
    neighbors = set()
    update_neighbors(X, community, neighbors, node)
    while True:
        if not grow_community(X, community, neighbors, alpha):
            break
        clean_community(X, community, alpha)
    return community


def lfk(X, alpha=1.0):
    """Apply Lancichinetti-Fortunato-Kertész clustering algorithm to the graph X."""
    assigned_nodes = np.zeros(X.shape[0], dtype=bool)
    communities = []
    while not assigned_nodes.all():
        node = np.random.choice(np.where(np.logical_not(assigned_nodes))[0])
        community = natural_community(X, node, alpha)
        assigned_nodes = np.logical_or(assigned_nodes, community)
        communities += [community]
    return communities
