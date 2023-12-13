import numpy as np
import sklearn.metrics as sk_metrics

from ..lfk_slow import lfk, grow_community, natural_community, node_fitness


X = np.array(
    [
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 1, 0],
    ]
)


class TestLFK:
    def test_grow_community(self):
        community = np.array([True, True, False, False, False, False, False])
        neigbors = {2, 3}
        assert node_fitness(X, community, 2) > node_fitness(X, community, 3)

        grow_community(X, community, neigbors)
        assert np.all(
            community == np.array([True, True, True, False, False, False, False])
        )
        assert neigbors == {3}

        grow_community(X, community, neigbors)
        assert np.all(
            community == np.array([True, True, True, True, False, False, False])
        )
        assert neigbors == {4}

    def test_natural_community(self):
        assert np.all(
            natural_community(X, 0)
            == np.array([True, True, True, True, False, False, False])
        )

    def test_lfk(self):
        community_1 = np.array([True, True, True, True, False, False, False])
        community_2 = np.array([False, False, False, False, True, True, True])
        communities = lfk(X)
        print(communities)
        assert len(communities) == 2
        assert (
            np.all(communities[0] == community_1)
            and np.all(communities[1] == community_2)
        ) or (
            np.all(communities[1] == community_1)
            and np.all(communities[0] == community_2)
        )
