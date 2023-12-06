from sklearn import metrics
import numpy as np



# Supervised metrics

def metrics_from_ground_truth(gt, pred):
    report = dict()
    report['Rand index'] = metrics.rand_score(gt, pred)
    report['AMI'] = metrics.adjusted_mutual_info_score(gt, pred)
    report['V measure'] = metrics.v_measure_score(gt, pred)
    report['Fowlkes-Mallows'] = metrics.fowlkes_mallows_score(gt, pred)
    return report


def rand_index(gt, pred):
    """ Measures similarity between two clusterings by comparing
    the number of pairs of elements that are assigned in the same or in different clusters,
    in the predicted and true clusterings.
    """
    # TODO: Can be improved with Adjusted Rand Index
    grouped_gt = gt[:, None] == gt[None, :]
    grouped_pred = pred[:, None] == pred[None, :]
    _a = grouped_gt & grouped_pred
    a = ( np.sum(_a) - np.sum(np.diagonal(_a)) ) / 2 # Number of pairs of elements that are assigned to the same cluster in both predicted and true clusterings
    _b = ~grouped_gt & ~grouped_pred
    b = ( np.sum(_b) - np.sum(np.diagonal(_b)) ) / 2 # Number of pairs of elements that are assigned to different clusters in both predicted and true clusterings
    _c = grouped_gt & ~grouped_pred
    c = ( np.sum(_c) - np.sum(np.diagonal(_c)) ) / 2 # Number of pairs of elements that are assigned to the same cluster in the true clustering and to different clusters in the predicted clustering
    _d = ~grouped_gt & grouped_pred
    d = ( np.sum(_d) - np.sum(np.diagonal(_d)) ) / 2 # Number of pairs of elements that are assigned to the same cluster in the true clustering and to different clusters in the predicted clustering
    assert a + b + c + d  == gt.shape[0]*(gt.shape[0]-1)/2  # Sanity check
    return (a + b) / (a + b + c + d)
    

def entropy(p):
    p[p == 0] = 1  # Avoid division by zero
    return -np.sum(p * np.log(p))


def mutual_information(gt, pred, Q=None):
    """ Measures the similarity between two clusterings by assessing how much information is needed to infer one clustering from the other.
    """
    if Q is None:
        Q = max(gt.max(), pred.max()) + 1
    p_gt = np.bincount(gt, minlength=Q) / gt.shape[0]
    p_pred = np.bincount(pred, minlength=Q) / pred.shape[0]
    p_gt[p_gt == 0] = 1  # Avoid division by zero
    p_pred[p_pred == 0] = 1
    p_joint, _, _ = np.histogram2d(gt, pred, bins=Q)
    p_joint /= gt.shape[0]
    arg_log = p_joint / (p_gt[:, None] * p_pred[None, :])
    arg_log[arg_log == 0] = 1  # Avoid log(0)
    return np.sum(p_joint * np.log(arg_log))
    
    
def normalized_mutual_information(gt, pred, Q=None):
    """ Normalized version of mutual information."""
    # TODO: Can be improved with Adjusted Mutual Information
    if Q is None:
        Q = max(gt.max(), pred.max()) + 1
    p_gt = np.bincount(gt) / gt.shape[0]
    p_pred = np.bincount(pred) / pred.shape[0]
    denom = entropy(p_gt) + entropy(p_pred)
    if denom == 0:
        return 0
    return (2*mutual_information(gt, pred, Q)) / denom
    

# Unsupervised metrics

def SBM_clustering_coefficient(alpha, pi):
    return np.einsum("q, l, m, ql, qm, lm ->", alpha, alpha, alpha, pi, pi, pi)/np.einsum("q, l, m, ql, qm ->", alpha, alpha, alpha, pi, pi)
    
    
def slow_SBM_clustering_coefficient(alpha, pi):
    Q = alpha.shape[0]
    sum_1 = 0
    sum_2 = 0
    for q in range(Q):
        for l in range(Q):
            for m in range(Q):
                prod_2 = alpha[q] * alpha[l] * alpha[m] * pi[q, l] * pi[q, m]
                sum_2 += prod_2
                sum_1 += prod_2 * pi[l, m]
    return sum_1 / sum_2
