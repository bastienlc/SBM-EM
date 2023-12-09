import numpy as np

from ..constants import *
from .generic import GenericImplementation


class NumpyImplementation(GenericImplementation):
    def input(self, array):
        return array

    def output(self, array):
        return array

    def m_step(self, X, tau):
        n = X.shape[0]
        alpha = np.sum(tau, axis=0) / n
        pi = (
            np.einsum("ij,iq,jl->ql", X, tau, tau)
            - np.einsum("ii,iq,il->ql", X, tau, tau)
        ) / (np.einsum("iq,jl->ql", tau, tau) - np.einsum("iq,il->ql", tau, tau))
        return alpha, pi

    def init_tau(self, n, Q):
        tau = np.random.rand(n, Q)
        return tau / np.sum(tau, axis=1)[:, np.newaxis]

    def _compute_b(self, X: np.ndarray, pi: np.ndarray):
        # returns an array of shape (n, Q, n, Q)
        n = X.shape[0]
        Q = pi.shape[0]
        repeated_pi = np.repeat(pi[np.newaxis, :, np.newaxis, :], n, axis=0).repeat(
            n, axis=2
        )
        repeated_X = np.repeat(X[:, np.newaxis, :, np.newaxis], Q, axis=1).repeat(
            Q, axis=3
        )
        b_values = repeated_pi**repeated_X * (1 - repeated_pi) ** (1 - repeated_X)
        b_values[np.arange(n), :, np.arange(n), :] = 1
        return b_values

    def fixed_point_iteration(self, tau, X, alpha, pi):
        b_values = self._compute_b(X, pi)
        tau = alpha[None, :] * np.prod(b_values**tau, axis=(2, 3))
        tau /= np.sum(tau, axis=1)[:, None]

        return tau

    def e_step(self, X, alpha, pi):
        n = X.shape[0]
        Q = alpha.shape[0]
        tau = self.init_tau(n, Q)
        for _ in range(MAX_FIXED_POINT_ITERATIONS):
            previous_tau = np.copy(tau)
            tau = self.fixed_point_iteration(tau, X, alpha, pi)

            if np.linalg.norm(previous_tau - tau, ord=1) < EPSILON:
                break
        return tau

    def log_likelihood(self, X, alpha, pi, tau):
        n = X.shape[0]
        ll = 0
        ll += np.sum(
            tau * np.repeat(np.log(alpha)[np.newaxis, :], n, axis=0), axis=(0, 1)
        )
        ll -= np.sum(tau * np.log(tau), axis=(0, 1))
        b_values = self._compute_b(X, pi)
        log_b_values = np.log(b_values)
        ll += 1 / 2 * np.einsum("iq,jl,iqjl->", tau, tau, log_b_values)
        return ll

    def parameters_are_ok(self, alpha, pi, tau):
        if np.abs(np.sum(alpha) - 1) > PRECISION:
            return False
        if np.any(alpha < 0):
            return False
        if np.any(alpha > 1):
            return False
        if np.any(pi < 0):
            return False
        if np.any(pi > 1):
            return False
        if np.any(tau < 0):
            return False
        if np.any(tau > 1):
            return False
        if np.any((np.sum(tau, axis=1) - 1) > PRECISION):
            return False
        if np.any(pi - pi.transpose() > PRECISION):
            return False
        return True
