import torch

from ..constants import *
from .generic import GenericImplementation

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
torch.no_grad()
torch.set_default_dtype(
    torch.float64
)  # torch.float32 is not precise enough for our needs (nans show up)


def normalize_matrix_with_nan_handling(tau):
    normalized_tau = tau / torch.sum(tau, dim=1, keepdim=True)
    if torch.any(torch.isnan(normalized_tau)):
        _, max_indices = torch.max(tau, dim=1, keepdim=True)
        replacement = torch.full_like(tau, EPSILON)
        replacement.scatter_(1, max_indices, 1 - (tau.shape[1] - 1) * EPSILON)
        nan_columns = torch.isnan(normalized_tau)
        normalized_tau = torch.where(nan_columns, replacement, normalized_tau)

    return normalized_tau


class PytorchImplementation(GenericImplementation):
    def input(self, array):
        return torch.tensor(array, device=DEVICE, dtype=torch.float64)

    def output(self, array):
        if isinstance(array, torch.Tensor):
            return array.cpu().detach().numpy()
        else:
            return array

    def m_step(self, X: torch.Tensor, tau: torch.Tensor):
        n = X.shape[0]
        alpha = torch.sum(tau, axis=0) / n
        pi = (
            torch.einsum("ij,iq,jl->ql", X, tau, tau)
            - torch.einsum("ii,iq,il->ql", X, tau, tau)
        ) / (torch.einsum("iq,jl->ql", tau, tau) - torch.einsum("iq,il->ql", tau, tau))

        return alpha, pi

    def init_tau(self, n, Q):
        tau = torch.rand(n, Q, device=DEVICE)
        return tau / torch.sum(tau, dim=1, keepdim=True)

    def _compute_b(self, X: torch.Tensor, pi: torch.Tensor):
        # returns a tensor of shape (n, Q, n, Q)
        n = X.shape[0]
        Q = pi.shape[0]
        repeated_pi = pi.unsqueeze(0).unsqueeze(2).expand(n, -1, n, -1)
        repeated_X = X.unsqueeze(1).unsqueeze(3).expand(-1, Q, -1, Q)
        b_values = repeated_pi**repeated_X * (1 - repeated_pi) ** (1 - repeated_X)
        b_values[torch.arange(n), :, torch.arange(n), :] = 1
        return b_values

    def fixed_point_iteration(self, tau, X, alpha, pi):
        n = X.shape[0]
        b_values = self._compute_b(X, pi)
        tau = alpha.unsqueeze(0).expand(n, -1) * torch.prod(
            torch.prod(torch.pow(b_values, tau), dim=3), dim=2
        )
        tau /= torch.sum(tau, dim=1, keepdim=True)

        return tau

    def e_step(self, X: torch.Tensor, alpha: torch.Tensor, pi: torch.Tensor):
        n = X.shape[0]
        Q = alpha.shape[0]
        n_inits = 0
        norm_change = 1 / EPSILON
        while norm_change > EPSILON:
            if n_inits >= MAX_FIXED_POINT_INITS:
                if RAISE_ERROR_ON_FIXED_POINT:
                    raise TimeoutError(
                        f"Fixed points iteration did not converge after {n_inits} initializations."
                    )
                else:
                    print(
                        f"Fixed points iteration did not converge after {n_inits} initializations. Restarting."
                    )
                    n_inits = 0
            tau = self.init_tau(n, Q)
            n_inits += 1
            for _ in range(MAX_FIXED_POINT_ITERATIONS):
                previous_tau = tau.clone()
                tau = 0.9 * tau + 0.1 * self.fixed_point_iteration(tau, X, alpha, pi)
                norm_change = torch.linalg.norm(previous_tau - tau, ord=1)

                if norm_change < EPSILON:
                    break
        return tau

    def log_likelihood(
        self,
        X: torch.Tensor,
        alpha: torch.Tensor,
        pi: torch.Tensor,
        tau: torch.Tensor,
        elbo: bool = True,
    ):
        n = X.shape[0]
        ll = 0
        ll += torch.sum(tau * torch.log(alpha).expand(n, -1), dim=[0, 1])
        if elbo:
            tau_log = tau * torch.log(tau)
            tau_log = torch.nan_to_num(tau_log, nan=0.0)  # Avoid NaN due to log(0)
            ll -= torch.sum(tau_log, dim=[0, 1])
        b_values = self._compute_b(X, pi)
        log_b_values = torch.log(b_values)
        ll += 1 / 2 * torch.einsum("iq,jl,iqjl->", tau, tau, log_b_values)
        return ll

    def check_parameters(
        self, alpha: torch.Tensor, pi: torch.Tensor, tau: torch.Tensor
    ):
        if torch.any(torch.isnan(alpha)):
            raise ValueError(f"Some alphas are nan")
        if torch.any(torch.isnan(pi)):
            raise ValueError(f"Some pis are nan")
        if torch.any(torch.isnan(tau)):
            raise ValueError(f"Some taus are nan")
        if torch.abs(torch.sum(alpha) - 1) > PRECISION:
            raise ValueError(f"Sum of alpha is {torch.sum(alpha)}")
        if torch.any(alpha < 0):
            raise ValueError(f"Some alphas are negative")
        if torch.any(alpha > 1):
            raise ValueError(f"Some alphas are greater than 1")
        if torch.any(pi < 0):
            raise ValueError(f"Some pis are negative")
        if torch.any(pi > 1):
            raise ValueError(f"Some pis are greater than 1")
        if torch.any(tau < 0):
            raise ValueError(f"Some taus are negative")
        if torch.any(tau > 1):
            raise ValueError(f"Some taus are greater than 1")
        if torch.any((torch.sum(tau, axis=1) - 1) > PRECISION):
            raise ValueError(f"Some taus do not sum to 1")
        if torch.any(pi - torch.transpose(pi, 0, 1) > PRECISION):
            raise ValueError(f"Pi is not symmetric")
        return True


class PytorchLowMemoryImplementation(PytorchImplementation):
    def _partial_compute_b(self, X: torch.Tensor, pi: torch.Tensor):
        # returns a tensor of shape (n, Q, Q)
        n = X.shape[0]
        Q = pi.shape[0]
        repeated_pi = pi.unsqueeze(0).expand(n, -1, -1)
        repeated_X = X.unsqueeze(1).unsqueeze(2).expand(-1, Q, Q)
        b_values = repeated_pi**repeated_X * (1 - repeated_pi) ** (1 - repeated_X)

        return b_values

    def fixed_point_iteration(self, tau, X, alpha, pi):
        n = X.shape[0]
        Q = pi.shape[0]
        previous_tau_repeated = tau.clone().unsqueeze(1).expand(-1, Q, -1)
        for i in range(n):
            b_values = self._partial_compute_b(X[i, :], pi)
            b_values[i, :, :] = 1
            tau[i, :] = alpha * torch.prod(
                torch.prod(torch.pow(b_values, previous_tau_repeated), dim=2), dim=0
            )

        tau /= torch.sum(tau, dim=1, keepdim=True)

        return tau

    def log_likelihood(
        self, X: torch.Tensor, alpha: torch.Tensor, pi: torch.Tensor, tau: torch.Tensor
    ):
        n = X.shape[0]
        ll = 0
        ll += torch.sum(tau * torch.log(alpha).expand(n, -1), dim=[0, 1])
        ll -= torch.sum(tau * torch.log(tau), dim=[0, 1])
        for i in range(n):
            b_values = self._partial_compute_b(X[i, :], pi)
            b_values[i, :, :] = 1
            log_b_values = torch.log(b_values)
            ll += 1 / 2 * torch.einsum("q,jl,jql->", tau[i, :], tau, log_b_values)
        return ll


class PytorchLogImplementation(PytorchImplementation):
    def fixed_point_iteration(self, tau, X, alpha, pi):
        Q = pi.shape[0]
        previous_tau = tau.clone()
        for q in range(Q):
            tau[:, q] = (
                torch.log(alpha[q])
                + torch.einsum("jl,ij,l->i", previous_tau, X, torch.log(pi[q, :]))
                + torch.einsum(
                    "jl,ij,l->i", previous_tau, 1 - X, torch.log(1 - pi[q, :])
                )
                - torch.einsum(
                    "il,i,l->i", previous_tau, torch.diagonal(X), torch.log(pi[q, :])
                )
                - torch.einsum(
                    "il,i,l->i",
                    previous_tau,
                    torch.diagonal(1 - X),
                    torch.log(1 - pi[q, :]),
                )
            )

        tau = torch.exp(tau)
        tau = normalize_matrix_with_nan_handling(tau)

        return tau

    def log_likelihood(
        self, X: torch.Tensor, alpha: torch.Tensor, pi: torch.Tensor, tau: torch.Tensor
    ):
        n = X.shape[0]
        Q = alpha.shape[0]
        ll = 0
        ll += torch.sum(tau * torch.log(alpha).expand(n, -1), dim=[0, 1])
        ll -= torch.sum(tau * torch.log(tau), dim=[0, 1])
        for q in range(Q):
            ll += (
                1
                / 2
                * torch.einsum("i,jl,ij,l->", tau[:, q], tau, X, torch.log(pi[q, :]))
            )
            ll += (
                1
                / 2
                * torch.einsum(
                    "i,jl,ij,l->", tau[:, q], tau, 1 - X, torch.log(1 - pi[q, :])
                )
            )
            ll -= (
                1
                / 2
                * torch.einsum(
                    "i,il,i,l->", tau[:, q], tau, torch.diagonal(X), torch.log(pi[q, :])
                )
            )
            ll -= (
                1
                / 2
                * torch.einsum(
                    "i,il,i,l->",
                    tau[:, q],
                    tau,
                    torch.diagonal(1 - X),
                    torch.log(1 - pi[q, :]),
                )
            )
        return ll
