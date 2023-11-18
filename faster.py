import numpy as np

from constants import *


def init_X(X):
    return X


def m_step(X, tau):
    n = X.shape[0]
    alpha = np.sum(tau, axis=0) / n
    pi = np.einsum("ij,iq,jl->ql", X, tau, tau) / (
        np.einsum("iq,jl->ql", tau, tau) - np.einsum("iq,il->ql", tau, tau)
    )
    return alpha, pi


def b(x, pi):
    return pi**x * (1 - pi) ** (1 - x)


def init_tau(n, Q):
    tau = np.random.rand(n, Q)
    return tau / np.sum(tau, axis=1)[:, np.newaxis]


def compute_b(X: np.ndarray, pi: np.ndarray):
    # returns a tensor of shape (n, Q, n, Q)
    n = X.shape[0]
    Q = pi.shape[0]
    repeated_pi = np.repeat(pi[np.newaxis, :, np.newaxis, :], n, axis=0).repeat(
        n, axis=2
    )
    repeated_X = np.repeat(X[:, np.newaxis, :, np.newaxis], Q, axis=1).repeat(Q, axis=3)
    b_values = repeated_pi**repeated_X * (1 - repeated_pi) ** (1 - repeated_X)
    b_values[np.arange(n), :, np.arange(n), :] = 1
    return b_values


def e_step(X, alpha, pi):
    n = X.shape[0]
    Q = alpha.shape[0]
    tau = init_tau(n, Q)
    for _ in range(MAX_FIXED_POINT_ITERATIONS):
        previous_tau = np.copy(tau)
        b_values = compute_b(X, pi)
        for i in range(n):
            for q in range(Q):
                tau[i, q] = alpha[q]
                tau[i, q] *= np.prod(b_values[i, q, :, :] ** previous_tau, axis=(0, 1))
            tau[i, :] /= np.sum(tau[i, :])

        if np.linalg.norm(previous_tau - tau, ord=1) < EPSILON:
            break
    return tau


def log_likelihood(X, alpha, pi, tau):
    n = X.shape[0]
    ll = 0
    ll += np.sum(tau * np.repeat(np.log(alpha)[np.newaxis, :], n, axis=0), axis=(0, 1))
    ll -= np.sum(tau * np.log(tau), axis=(0, 1))
    b_values = compute_b(X, pi)
    log_b_values = np.log(b_values)
    ll += 1 / 2 * np.einsum("iq,jl,iqjl->", tau, tau, log_b_values)
    return ll


def parameters_are_ok(alpha, pi, tau):
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
