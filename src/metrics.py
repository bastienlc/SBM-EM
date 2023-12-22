import numpy as np
from sklearn import metrics

# Supervised metrics


def rand_index(gt, pred):
    """Measures similarity between two clusterings by comparing
    the number of pairs of elements that are assigned in the same or in different clusters,
    in the predicted and true clusterings.
    """
    grouped_gt = gt[:, None] == gt[None, :]
    grouped_pred = pred[:, None] == pred[None, :]
    _a = grouped_gt & grouped_pred
    a = (
        np.sum(_a) - np.sum(np.diagonal(_a))
    ) / 2  # Number of pairs of elements that are assigned to the same cluster in both predicted and true clusterings
    _b = ~grouped_gt & ~grouped_pred
    b = (
        np.sum(_b) - np.sum(np.diagonal(_b))
    ) / 2  # Number of pairs of elements that are assigned to different clusters in both predicted and true clusterings
    _c = grouped_gt & ~grouped_pred
    c = (
        np.sum(_c) - np.sum(np.diagonal(_c))
    ) / 2  # Number of pairs of elements that are assigned to the same cluster in the true clustering and to different clusters in the predicted clustering
    _d = ~grouped_gt & grouped_pred
    d = (
        np.sum(_d) - np.sum(np.diagonal(_d))
    ) / 2  # Number of pairs of elements that are assigned to the same cluster in the true clustering and to different clusters in the predicted clustering
    assert a + b + c + d == gt.shape[0] * (gt.shape[0] - 1) / 2  # Sanity check
    return (a + b) / (a + b + c + d)


def entropy(p):
    p[p == 0] = 1  # Avoid division by zero
    return -np.sum(p * np.log(p))


def mutual_information(gt, pred, Q=None):
    """Measures the similarity between two clusterings by assessing how much information is needed to infer one clustering from the other."""
    if Q is None:
        Q = max(gt.max(), pred.max()) + 1
    p_gt = np.bincount(gt, minlength=Q) / gt.shape[0]
    p_pred = np.bincount(pred, minlength=Q) / pred.shape[0]
    mask_gt = (gt[:, None] == np.arange(Q)).astype(int)
    mask_pred = (pred[:, None] == np.arange(Q)).astype(int)
    p_joint = np.dot(mask_gt.T, mask_pred) / gt.shape[0]
    denom_log = p_gt[:, None] * p_pred[None, :]
    denom_mask = denom_log != 0  # Avoid division by 0
    arg_log = np.ones((Q, Q))
    arg_log[denom_mask] = p_joint[denom_mask] / denom_log[denom_mask]
    arg_log[arg_log == 0] = 1  # Avoid log(0)
    return np.sum(p_joint * np.log(arg_log))


def normalized_mutual_information(gt, pred, Q=None):
    """Normalized version of mutual information."""
    if Q is None:
        Q = max(gt.max(), pred.max()) + 1
    p_gt = np.bincount(gt) / gt.shape[0]
    p_pred = np.bincount(pred) / pred.shape[0]
    denom = entropy(p_gt) + entropy(p_pred)
    if denom == 0:
        return 0
    return (2 * mutual_information(gt, pred, Q)) / denom


# Unsupervised metrics


def SBM_clustering_coefficient(alpha, pi):
    return np.einsum(
        "q, l, m, ql, qm, lm ->", alpha, alpha, alpha, pi, pi, pi
    ) / np.einsum("q, l, m, ql, qm ->", alpha, alpha, alpha, pi, pi)


def clustering_coefficient(X, cluster=None):
    """Compute the clustering coefficient over the specified cluster. Also called "transitivity".
    cluster is the mask indicating which nodes are in the considered cluster.
    If cluster is None, compute the clustering coefficient over the whole graph.
    """
    sub_X = X if cluster is None else X[cluster, :][:, cluster]
    degrees = np.sum(sub_X, axis=1)
    denom = np.sum(degrees * (degrees - 1))
    if denom == 0:
        return 0
    else:
        return np.einsum("ij, jk, ki ->", sub_X, sub_X, sub_X) / denom


def intra_cluster_density(X, cluster, weighted=False):
    """cluster is the mask indicating which nodes are in the considered cluster"""
    if weighted:
        nb_internal_edges = np.sum(X[cluster, :][:, cluster]) / 2
    else:
        nb_internal_edges = np.sum(X[cluster, :][:, cluster] > 0) / 2
    nb_nodes = np.sum(cluster)
    return nb_internal_edges / (nb_nodes * (nb_nodes - 1) / 2)


def inter_cluster_density(X, cluster, weighted=False):
    """cluster is the mask indicating which nodes are in the considered cluster"""
    if weighted:
        nb_external_edges = np.sum(X[cluster, :][:, np.logical_not(cluster)])
    else:
        nb_external_edges = np.sum(X[cluster, :][:, np.logical_not(cluster)] > 0)
    n = X.shape[0]
    nb_nodes = np.sum(cluster)
    return nb_external_edges / (nb_nodes * (n - nb_nodes))


def conductance(X, cluster):
    """A good cluster should have low conductance, especially compared to the lowest conductance on the graph.
    cluster is the list of indices of nodes in the considered cluster.
    """
    a = np.sum(X[cluster])
    b = np.sum(X) - np.sum(X[np.logical_not(cluster)])
    return inter_cluster_density(X, cluster) / min(a, b)


def modularity_simple(X, clustering):
    """The modularity of a clustering compares the observed connectivity within clusters to
    its expected value for a random graph with the same degree distribution.
    Implemented as presented by M. E. J. Newman and M. Girvan in https://arxiv.org/abs/cond-mat/0308217.
    """
    m = (X.trace() + np.sum(X)) / 2
    c = clustering.shape[0]
    e = np.zeros((c, c))
    for i in range(c):
        e[i, i] = np.sum(X[clustering[i], :][:, clustering[i]]) / 2
    for i in range(c):
        for j in range(i + 1, c):
            e[i, j] = np.sum(X[clustering[i], :][:, clustering[j]])
            e[j, i] = e[i, j]
    e /= m
    return e.trace() - np.sum(e @ e)


def modularity(X, clustering):
    """The modularity of a clustering compares the observed connectivity within clusters to
    its expected value for a random graph with the same degree distribution.
    We here make the common assumption that the random graph resulting from the configuration model contains no self-loops and no multiple edges.
    Implemented as presented in matrix form by M. E. J. Newman in https://arxiv.org/abs/physics/0602124
    """
    m = (X.trace() + np.sum(X)) / 2
    S = np.transpose(clustering).astype(int)
    degrees = np.sum(X, axis=1)
    B = X - (degrees[:, None] @ degrees[None, :] / (2 * m))
    return 1 / (2 * m) * np.trace(S.T @ B @ S)


def modularity_v2(X, clustering):
    """The modularity of a clustering compares the observed connectivity within clusters to
    its expected value for a random graph with the same degree distribution.
    We here make the common assumption that the random graph resulting from the configuration model contains no self-loops and no multiple edges.
    Implemented as presented by B.H.Good, Y.A. de Montjoye and A. Clauset in https://arxiv.org/abs/0910.0165.pdf.
    """
    m = (X.trace() + np.sum(X)) / 2
    c = len(clustering)
    e = np.zeros(c)
    d = np.zeros(c)
    for i in range(c):
        e[i] = np.sum(X[clustering[i], :][:, clustering[i]]) / 2
        d[i] = np.sum(X[clustering[i], :])
    e /= m
    d /= 2 * m
    return np.sum(e - d**2)
