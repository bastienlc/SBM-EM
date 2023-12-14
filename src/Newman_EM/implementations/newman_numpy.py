import numpy as np

def log_likelihood_numpy(A, pi, q, theta):
    # A : n x n
    # pi : Q
    # q : n x Q
    # theta : Q x n

    B = np.einsum('ij,rj -> ir', A, np.log(theta+ 1e-6))
    B = np.log(pi)[None, :] + B
    B = np.multiply(B, q)
    B = np.sum(B)
    return B

def e_step_numpy(A, pi, theta):
    # A : n x n
    # pi : Q
    # theta : Q x n

    q = np.log(pi + 1e-6) + np.einsum('ij,rj -> ir', A, np.log(theta + 1e-6))
    q = np.exp(q)
    q = q / np.sum(q, axis=1, keepdims=True)

    return q

def m_step_numpy(A, q):
    # A : n x n
    # pi : Q
    # q : n x Q
    n, Q = A.shape[0], q.shape[1]
    pi = np.zeros(Q)
    theta = np.zeros((Q, n))

    pi = np.sum(q, axis=0) / n
    theta = np.einsum('ij,ir -> rj', A, q) / np.einsum('ij,ir -> r', A, q)[:, None]

    return pi, theta