import numpy as np
import sklearn.metrics as sk_metrics
import networkx as nx

from ..metrics import (
    mutual_information,
    normalized_mutual_information,
    rand_index,
    clustering_coefficient,
)

n = 5
Q = 3
true_labels = np.random.randint(0, Q, n)
predictions = true_labels.copy()
for i in np.random.choice(np.arange(n), 3, replace=False):
    predictions[i] = np.random.randint(0, Q)
close_epsilon = 1e-5

G = nx.erdos_renyi_graph(n, 0.5)
X = nx.adjacency_matrix(G).todense()


class TestMetrics:
    def test_rand_index(self):
        assert (
            abs(
                sk_metrics.rand_score(true_labels, predictions)
                - rand_index(true_labels, predictions)
            )
            < close_epsilon
        )

    def test_mutual_information(self):
        assert (
            abs(
                sk_metrics.mutual_info_score(true_labels, predictions)
                - mutual_information(true_labels, predictions, Q=Q)
            )
            < close_epsilon
        )

    def test_normalized_mutual_information(self):
        assert (
            abs(
                sk_metrics.normalized_mutual_info_score(true_labels, predictions)
                - normalized_mutual_information(true_labels, predictions)
            )
            < close_epsilon
        )

    def test_clustering_coefficient(self):
        assert abs(nx.transitivity(G) - clustering_coefficient(X, None)) < close_epsilon
