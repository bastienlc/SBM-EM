import numpy as np

def log_likelihood(A, pi, q, theta):
    # A : n x n
    # pi : Q
    # q : n x Q
    # theta : Q x n

    n = A.shape[0]
    Q = pi.shape[0]

    output = 0

    for i in range(n):
        for r in range(Q):
            term = 0
            term += np.log(pi[r])
            for j in range(n):
                term += A[i, j] * np.log(theta[r, j] + 1e-6)
            output += q[i, r] * term

    return output

def e_step(A, pi, theta):
    # A : n x n
    # pi : Q
    # theta : Q x n
    n, Q = A.shape[0], pi.shape[0]
    q = np.zeros((n, Q))
    theta = theta + 1e-6
    pi = pi + 1e-6
    for i in range(n):
        for r in range(Q):
            term = pi[r]
            for j in range(n):
                term *= theta[r, j]**A[i, j]
            q[i, r] = term
        q[i] = q[i] / np.sum(q[i])

    return q

def e_step_log(A, pi, theta):
    # A : n x n
    # pi : Q
    # theta : Q x n
    n, Q = A.shape[0], pi.shape[0]
    q = np.zeros((n, Q))
    for i in range(n):
        for r in range(Q):
            term = np.log(pi[r] + 1e-6)
            for j in range(n):
                term += A[i, j]*np.log(theta[r, j] + 1e-6)
            q[i, r] = term
        q[i] = np.exp(q[i])
        q[i] = q[i] / np.sum(q[i])

    return q