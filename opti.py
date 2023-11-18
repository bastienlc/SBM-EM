import torch

from constants import *

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
torch.no_grad()
print(f"Using device {DEVICE}")


def init_X(X):
    return torch.tensor(X, device=DEVICE).float()


def m_step(X: torch.Tensor, tau: torch.Tensor):
    n = X.shape[0]
    alpha = torch.sum(tau, axis=0) / n
    pi = torch.einsum("ij,iq,jl->ql", X, tau, tau) / (
        torch.einsum("iq,jl->ql", tau, tau) - torch.einsum("iq,il->ql", tau, tau)
    )

    return alpha, pi


def init_tau(n, Q):
    tau = torch.rand(n, Q, device=DEVICE)
    return tau / torch.sum(tau, dim=1, keepdim=True)


def compute_b(X: torch.Tensor, pi: torch.Tensor):
    # returns a tensor of shape (n, Q, n, Q)
    n = X.shape[0]
    Q = pi.shape[0]
    repeated_pi = pi.unsqueeze(0).unsqueeze(2).expand(n, -1, n, -1)
    repeated_X = X.unsqueeze(1).unsqueeze(3).expand(-1, Q, -1, Q)
    b_values = repeated_pi**repeated_X * (1 - repeated_pi) ** (1 - repeated_X)
    b_values[torch.arange(n), :, torch.arange(n), :] = 1
    return b_values


def e_step(X: torch.Tensor, alpha: torch.Tensor, pi: torch.Tensor):
    n = X.shape[0]
    Q = alpha.shape[0]
    tau = init_tau(n, Q)
    for _ in range(MAX_FIXED_POINT_ITERATIONS):
        previous_tau = tau.clone()
        b_values = compute_b(X, pi)
        for i in range(n):
            for q in range(Q):
                tau[i, q] = alpha[q]
                tau[i, q] *= torch.prod(
                    torch.prod(b_values[i, q, :, :] ** previous_tau, dim=0), dim=0
                )
            tau[i, :] /= torch.sum(tau[i, :])

        if torch.linalg.norm(previous_tau - tau, ord=1) < EPSILON:
            break
    return tau


def log_likelihood(
    X: torch.Tensor, alpha: torch.Tensor, pi: torch.Tensor, tau: torch.Tensor
):
    n = X.shape[0]
    ll = 0
    ll += torch.sum(tau * torch.log(alpha).expand(n, -1), dim=[0, 1])
    ll -= torch.sum(tau * torch.log(tau), dim=[0, 1])
    b_values = compute_b(X, pi)
    log_b_values = torch.log(b_values)
    ll += 1 / 2 * torch.einsum("iq,jl,iqjl->", tau, tau, log_b_values)
    return ll


def parameters_are_ok(alpha: torch.Tensor, pi: torch.Tensor, tau: torch.Tensor):
    if torch.abs(torch.sum(alpha) - 1) > PRECISION:
        return False
    if torch.any(alpha < 0):
        return False
    if torch.any(alpha > 1):
        return False
    if torch.any(pi < 0):
        return False
    if torch.any(pi > 1):
        return False
    if torch.any(tau < 0):
        return False
    if torch.any(tau > 1):
        return False
    if torch.any((torch.sum(tau, axis=1) - 1) > PRECISION):
        return False
    if torch.any(pi - torch.transpose(pi, 0, 1) > PRECISION):
        return False
    return True
