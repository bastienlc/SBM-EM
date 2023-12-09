import torch

from ..constants import *
from .generic import GenericImplementation

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
torch.no_grad()
torch.set_default_dtype(
    torch.float64
)  # torch.float32 is not precise enough for our needs (nans show up)


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
        norm_change = 1e100
        while norm_change > EPSILON:
            if n_inits >= MAX_FIXED_POINT_INITS:
                raise TimeoutError(
                    f"Fixed points iteration did not converge after {n_inits} initializations."
                )
            tau = self.init_tau(n, Q)
            n_inits += 1
            for _ in range(MAX_FIXED_POINT_ITERATIONS):
                previous_tau = tau.clone()
                tau = self.fixed_point_iteration(tau, X, alpha, pi)
                norm_change = torch.linalg.norm(previous_tau - tau, ord=1)

                if norm_change < EPSILON:
                    break
        return tau

    def log_likelihood(
        self, X: torch.Tensor, alpha: torch.Tensor, pi: torch.Tensor, tau: torch.Tensor
    ):
        n = X.shape[0]
        ll = 0
        ll += torch.sum(tau * torch.log(alpha).expand(n, -1), dim=[0, 1])
        ll -= torch.sum(tau * torch.log(tau), dim=[0, 1])
        b_values = self._compute_b(X, pi)
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
