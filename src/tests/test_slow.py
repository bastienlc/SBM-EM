import numpy as np

from ..slow import b, e_step, fixed_point_iteration, init_tau, log_likelihood, m_step

n = 5
Q = 5
X = np.random.random((n, n))
alpha = np.random.random(Q)
alpha = alpha / np.sum(alpha)
pi = np.random.random((Q, Q))
pi = pi @ np.transpose(pi)
pi = pi / np.sum(pi, axis=0)
tau = init_tau(n, Q)


class TestSlow:
    def test_fixed_point_iteration(self):
        previous_tau = np.copy(tau)
        computed_tau = fixed_point_iteration(tau, X, alpha, pi)
        for i in range(n):
            for q in range(Q):
                tau[i, q] = alpha[q]
                for j in range(n):
                    if i != j:
                        for l in range(Q):
                            tau[i, q] *= b(X[i, j], pi[q, l]) ** previous_tau[j, l]
            tau[i, :] /= np.sum(tau[i, :])

        assert np.allclose(tau, computed_tau)

    def test_e_step(self):
        tau = e_step(X, alpha, pi)
        assert tau.shape == (n, Q)
        assert np.allclose(np.sum(tau, axis=1), np.ones(n))

    def test_m_step(self):
        computed_alpha, computed_pi = m_step(X, tau)
        expected_alpha = np.sum(tau, axis=0) / n
        expected_pi = np.zeros((Q, Q))
        for q in range(Q):
            for l in range(Q):
                normalization_factor = 0
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            expected_pi[q, l] += tau[i, q] * tau[j, l] * X[i, j]
                            normalization_factor += tau[i, q] * tau[j, l]
                expected_pi[q, l] /= normalization_factor
        assert np.allclose(expected_alpha, computed_alpha)
        assert np.allclose(expected_pi, computed_pi)

    def test_log_likelihood(self):
        computed_ll = log_likelihood(X, alpha, pi, tau)
        expected_ll = 0
        for i in range(n):
            for q in range(Q):
                expected_ll += tau[i, q] * np.log(alpha[q]) - tau[i, q] * np.log(
                    tau[i, q]
                )
                for j in range(n):
                    if i != j:
                        for l in range(Q):
                            expected_ll += (
                                (1 / 2)
                                * tau[i, q]
                                * tau[j, l]
                                * np.log(b(X[i, j], pi[q, l]))
                            )
        assert np.allclose(expected_ll, computed_ll)
