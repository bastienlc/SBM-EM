import numpy as np

from constants import *


def init_X(X):
    return X


def m_step(X, tau):
    alpha = np.mean(tau, axis=0)
    Q = tau.shape[1]
    n = X.shape[0]
    pi = np.zeros((Q, Q))
    for q in range(Q):
        for l in range(Q):
            num = 0
            denum = 0
            for i in range(n):
                for j in range(n):
                    if i != j:
                        num += tau[i, q] * tau[j, l] * X[i, j]
                        denum += tau[i, q] * tau[j, l]
            pi[q, l] = num / denum
    return alpha, pi


def b(x, pi):
    return pi**x * (1 - pi) ** (1 - x)


def init_tau(n, Q):
    tau = np.random.rand(n, Q)
    return tau / np.sum(tau, axis=1)[:, np.newaxis]


def e_step(X, alpha, pi):
    n = X.shape[0]
    Q = alpha.shape[0]
    tau = init_tau(n, Q)
    for _ in range(MAX_FIXED_POINT_ITERATIONS):
        previous_tau = np.copy(tau)
        for i in range(n):
            for q in range(Q):
                tau[i, q] = alpha[q]
                for j in range(n):
                    if i != j:
                        for l in range(Q):
                            tau[i, q] *= b(X[i, j], pi[q, l]) ** previous_tau[j, l]
            tau[i, :] /= np.sum(tau[i, :])

        if np.linalg.norm(previous_tau - tau, ord=1) < EPSILON:
            break
    return tau


def log_likelihood(X, alpha, pi, tau):
    n = X.shape[0]
    Q = alpha.shape[0]
    ll = 0
    for i in range(n):
        for q in range(Q):
            ll += tau[i, q] * np.log(alpha[q]) - tau[i, q] * np.log(tau[i, q])
            for j in range(n):
                if i != j:
                    for l in range(Q):
                        ll += (
                            (1 / 2)
                            * tau[i, q]
                            * tau[j, l]
                            * np.log(b(X[i, j], pi[q, l]))
                        )
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
