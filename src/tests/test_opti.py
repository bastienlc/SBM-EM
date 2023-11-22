import numpy as np
import torch

from faster import b, init_tau
from opti import (
    DEVICE,
    compute_b,
    e_step,
    fixed_point_iteration,
    log_likelihood,
    m_step,
)

n = 5
Q = 5
X = np.random.random((n, n))
alpha = np.random.random(Q)
alpha = alpha / np.sum(alpha)
pi = np.random.random((Q, Q))
pi = pi @ np.transpose(pi)
pi = pi / np.sum(pi, axis=0)
tau = init_tau(n, Q)


class TestFaster:
    def test_compute_b(self):
        computed_values = compute_b(
            torch.tensor(X, device=DEVICE), torch.tensor(pi, device=DEVICE)
        )
        assert computed_values.shape == (n, Q, n, Q)
        for i in range(n):
            for j in range(n):
                for q in range(Q):
                    for l in range(Q):
                        if i != j:
                            expected_value = pi[q, l] ** X[i, j] * (1 - pi[q, l]) ** (
                                1 - X[i, j]
                            )
                            assert np.allclose(
                                computed_values[i, q, j, l].cpu().item(), expected_value
                            )
                        else:
                            expected_value = 1
                            assert np.allclose(
                                computed_values[i, q, j, l].cpu().item(), expected_value
                            )

    def test_fixed_point_iteration(self):
        previous_tau = np.copy(tau)
        computed_tau = fixed_point_iteration(
            torch.tensor(tau, device=DEVICE),
            torch.tensor(X, device=DEVICE),
            torch.tensor(alpha, device=DEVICE),
            torch.tensor(pi, device=DEVICE),
        )
        for i in range(n):
            for q in range(Q):
                tau[i, q] = alpha[q]
                for j in range(n):
                    if i != j:
                        for l in range(Q):
                            tau[i, q] *= b(X[i, j], pi[q, l]) ** previous_tau[j, l]
            tau[i, :] /= np.sum(tau[i, :])

        assert np.allclose(tau, computed_tau.cpu().detach().numpy())

    def test_e_step(self):
        tau = (
            e_step(
                torch.tensor(X, device=DEVICE),
                torch.tensor(alpha, device=DEVICE),
                torch.tensor(pi, device=DEVICE),
            )
            .cpu()
            .detach()
            .numpy()
        )
        assert tau.shape == (n, Q)
        assert np.allclose(np.sum(tau, axis=1), np.ones(n))

    def test_m_step(self):
        computed_alpha, computed_pi = m_step(
            torch.tensor(X, device=DEVICE),
            torch.tensor(tau, device=DEVICE),
        )
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
        assert np.allclose(expected_alpha, computed_alpha.cpu().detach().numpy())
        assert np.allclose(expected_pi, computed_pi.cpu().detach().numpy())

    def test_log_likelihood(self):
        computed_ll = log_likelihood(
            torch.tensor(X, device=DEVICE),
            torch.tensor(alpha, device=DEVICE),
            torch.tensor(pi, device=DEVICE),
            torch.tensor(tau, device=DEVICE),
        )
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
        assert np.allclose(expected_ll, computed_ll.cpu().item())
