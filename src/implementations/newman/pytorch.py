import numpy as np
import torch

from ...constants import *
from ..generic import GenericImplementation

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

    def init_tau(self, n, Q):
        tau = np.random.rand(n, Q)
        tau = tau / np.sum(tau, axis=1).reshape(-1, 1)

        tau = torch.from_numpy(tau).type(torch.DoubleTensor)
        return tau

    def log_likelihood(
        self, X: torch.Tensor, alpha: torch.Tensor, pi: torch.Tensor, tau: torch.Tensor
    ):
        # A : n x n
        # pi : Q
        # q : n x Q
        # theta : Q x n

        B = torch.einsum("ij,rj -> ir", X, torch.log(pi + 1e-6))
        B = torch.log(alpha)[None, :] + B
        B = torch.multiply(B, tau)
        B = torch.sum(B)
        return B

    def e_step(self, X: torch.Tensor, alpha: torch.Tensor, pi: torch.Tensor):
        # A : n x n
        # pi : Q
        # theta : Q x n
        C = 10

        tau = torch.log(alpha + 1e-6) + torch.einsum(
            "ij,rj -> ir", X, torch.log(pi + 1e-6)
        )
        max_tau = torch.max(tau, axis=1).values
        # max_tau = max_tau.values
        # max_tau = max_tau.cpu().detach().numpy()
        min_tau = torch.min(tau, axis=1).values
        # min_tau = min_tau.values
        # min_tau = min_tau.cpu().detach().numpy()
        mask_tau_min = min_tau < -500.0
        max_tau = max_tau[mask_tau_min]
        min_tau = min_tau[mask_tau_min]
        ranges = torch.stack([max_tau, min_tau], axis=1)
        # print(ranges[ranges[:, 1] - ranges[:, 0] > 500.])
        C = torch.zeros(tau.shape[0], device=DEVICE, dtype=torch.float64)
        C[mask_tau_min] = -max_tau
        # print(C[mask_tau_min])
        tau = tau + C[:, None]
        tau = torch.exp(tau)
        tau = tau / torch.sum(tau, axis=1, keepdims=True)
        # print(tau)
        # # print where tau is nan
        # arg_tau_nan = torch.argwhere(torch.isnan(tau))
        # print(arg_tau_nan)
        # # take an example
        # exple_tau_nan = arg_tau_nan[0]
        # print(exple_tau_nan)
        # print(tau[exple_tau_nan[0], exple_tau_nan[1]])
        # # retrace the calculation of tau[exple_tau_nan[0], exple_tau_nan[1]]
        # print("alpha : ", alpha)
        # print(torch.log(alpha[exple_tau_nan[1]] + 1e-6))
        # print(X[exple_tau_nan[0], :])
        # print(pi[exple_tau_nan[1], :] + 1e-6)
        # print(torch.log(pi[exple_tau_nan[1], :] + 1e-6))
        # term1 = torch.log(alpha[exple_tau_nan[1]] + 1e-6 + torch.sum(X[exple_tau_nan[0], :] * torch.log(pi[exple_tau_nan[1], :] + 1e-6)))
        # print("term1 : ", term1)
        # term2 = torch.exp(term1)
        # print("term2 : ", term2)
        # term3 = torch.sum(term2)
        # print("term3 : ", term3)
        # term4 = term2 / term3
        # print("term4 : ", term4)
        if torch.any(torch.isnan(tau)):
            print(f"tau is nan")
            arg_tau_nan = torch.argwhere(torch.isnan(tau))
            exple_tau_nan = arg_tau_nan[0]
            print("exple_tau_nan : ", exple_tau_nan)
            print("tau", tau[exple_tau_nan[0], exple_tau_nan[1]])
            term11 = torch.log(alpha[exple_tau_nan[1]] + 1e-6)
            print("term11 : ", term11)
            term12 = X[exple_tau_nan[0], :] * torch.log(pi[exple_tau_nan[1], :] + 1e-6)
            print("term12 : ", term12)
            term13 = torch.sum(term12)
            print("term13 : ", term13)
            term1 = term11 + term13
            print("term1 : ", term1)
            term2 = C[exple_tau_nan[0]]
            print("term2 : ", term2)
            term3 = term1 + term2
            print("term3 : ", term3)
            term4 = torch.exp(term3)
            print("term4 : ", term4)
        else:
            # print(f"tau is not nan")
            pass
        return tau

    def m_step(self, X: torch.Tensor, tau: torch.Tensor):
        # A : n x n
        # pi : Q
        # q : n x Q
        alpha = torch.mean(tau, axis=0)
        pi = (
            torch.einsum("ij,ir -> rj", X, tau)
            / torch.einsum("ij,ir -> r", X, tau)[:, None]
        )

        return alpha, pi

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
        return True
