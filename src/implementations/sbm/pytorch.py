from typing import Optional, Tuple

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


def normalize_matrix_with_nan_handling(tau: torch.Tensor) -> torch.Tensor:
    """
    Normalizes the tau matrix while handling NaN values.

    Parameters
    ----------
    tau : torch.Tensor
        The tau matrix.

    Returns
    -------
    torch.Tensor
        Normalized tau matrix.
    """
    normalized_tau = tau / torch.sum(tau, dim=1, keepdim=True)

    if torch.any(torch.isnan(normalized_tau)):
        _, max_indices = torch.max(tau, dim=1, keepdim=True)
        replacement = torch.full_like(tau, EPSILON)
        replacement.scatter_(1, max_indices, 1 - (tau.shape[1] - 1) * EPSILON)
        nan_columns = torch.isnan(normalized_tau)
        normalized_tau = torch.where(nan_columns, replacement, normalized_tau)

    return normalized_tau


class PytorchImplementation(GenericImplementation):
    """
    PytorchImplementation class for the EM algorithm. This class implements the EM algorithm using PyTorch.

    Methods
    -------
    e_step(X, alpha, pi)
        Performs the E-step of the EM algorithm.
    init_tau(n, Q)
        Initializes the tau matrix.
    m_step(X, tau)
        Performs the M-step of the EM algorithm.
    log_likelihood(X, alpha, pi, tau)
        Computes the log-likelihood.
    check_parameters(alpha, pi, tau)
        Checks if the parameters are valid.
    fixed_point_iteration(tau, X, alpha, pi)
        Performs a fixed-point iteration of the E-step.
    input(array)
        Processes input arrays.
    output(array)
        Processes output arrays.
    """

    def input(self, array: np.ndarray) -> torch.Tensor:
        """
        Processes the input array.

        Parameters
        ----------
        array : np.ndarray
            Input data to be processed.

        Returns
        -------
        torch.Tensor
            Processed input data. It can then be fed to the other methods.
        """
        return torch.tensor(array, device=DEVICE, dtype=torch.float64)

    def output(self, array) -> np.ndarray:
        """
        Processes the output array.

        Parameters
        ----------
        array
            Output data to be processed.

        Returns
        -------
        np.ndarray
            Processed output data. Returns a numpy array.
        """
        if isinstance(array, torch.Tensor):
            return array.cpu().detach().numpy()
        else:
            return array

    def m_step(
        self, X: torch.Tensor, tau: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the M-step of the EM algorithm.

        Parameters
        ----------
        X : torch.Tensor
            The adjacency matrix of the graph.
        tau : torch.Tensor
            Current tau matrix.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Estimated alpha and pi parameters.
        """
        n = X.shape[0]
        alpha = torch.sum(tau, dim=0) / n
        pi = (
            torch.einsum("ij,iq,jl->ql", X, tau, tau)
            - torch.einsum("ii,iq,il->ql", X, tau, tau)
        ) / (torch.einsum("iq,jl->ql", tau, tau) - torch.einsum("iq,il->ql", tau, tau))
        return alpha, pi

    def init_tau(self, n: int, Q: int) -> torch.Tensor:
        """
        Initializes the tau matrix.

        Parameters
        ----------
        n : int
            Number of nodes.
        Q : int
            Number of classes.

        Returns
        -------
        torch.Tensor
            Initialized tau matrix.
        """
        tau = torch.rand(n, Q, device=DEVICE)
        return tau / torch.sum(tau, dim=1, keepdim=True)

    def _compute_b(self, X: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        """
        Computex the array of shape (n, Q, n, Q) used in fixed-point iterations.

        Parameters
        ----------
        X : torch.Tensor
            The adjacency matrix of the graph.
        pi : torch.Tensor
            Estimated pi parameters.

        Returns
        -------
        torch.Tensor
            Computed array.
        """
        n = X.shape[0]
        Q = pi.shape[0]
        repeated_pi = pi.unsqueeze(0).unsqueeze(2).expand(n, -1, n, -1)
        repeated_X = X.unsqueeze(1).unsqueeze(3).expand(-1, Q, -1, Q)
        b_values = repeated_pi**repeated_X * (1 - repeated_pi) ** (1 - repeated_X)
        b_values[torch.arange(n), :, torch.arange(n), :] = 1
        return b_values

    def fixed_point_iteration(
        self, tau: torch.Tensor, X: torch.Tensor, alpha: torch.Tensor, pi: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs a fixed-point iteration of the E-step.

        Parameters
        ----------
        tau : torch.Tensor
            Current tau matrix.
        X : torch.Tensor
            The adjacency matrix of the graph.
        alpha : torch.Tensor
            Estimated alpha parameters.
        pi : torch.Tensor
            Estimated pi parameters.

        Returns
        -------
        torch.Tensor
            Updated tau matrix.
        """
        n = X.shape[0]
        b_values = self._compute_b(X, pi)
        tau = alpha.unsqueeze(0).expand(n, -1) * torch.prod(
            torch.prod(torch.pow(b_values, tau), dim=3), dim=2
        )
        tau /= torch.sum(tau, dim=1, keepdim=True)

        return tau

    def e_step(
        self, X: torch.Tensor, alpha: torch.Tensor, pi: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs the E-step of the EM algorithm. This method uses fixed-point iterations.

        Parameters
        ----------
        X : torch.Tensor
            The adjacency matrix of the graph.
        alpha : torch.Tensor
            Estimated alpha parameters.
        pi : torch.Tensor
            Estimated pi parameters.

        Returns
        -------
        torch.Tensor
            Updated tau matrix.
        """
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
        elbo: Optional[bool] = True,
    ) -> torch.Tensor:
        """
        Computes the log-likelihood.

        Parameters
        ----------
        X : torch.Tensor
            The adjacency matrix of the graph.
        alpha : torch.Tensor
            Estimated alpha parameters.
        pi : torch.Tensor
            Estimated pi parameters.
        tau : torch.Tensor
            Current tau matrix.
        elbo : bool, optional
            If True, calculates the evidence lower bound, by default True.

        Returns
        -------
        torch.Tensor
            Log-likelihood value.
        """
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
    ) -> bool:
        """
        Checks if the parameters are valid.

        Parameters
        ----------
        alpha : torch.Tensor
            Estimated alpha parameters.
        pi : torch.Tensor
            Estimated pi parameters.
        tau : torch.Tensor
            Estimated tau matrix.

        Returns
        -------
        bool
            True if parameters are valid, False otherwise.
        """
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


class PytorchLogImplementation(PytorchImplementation):
    """
    PytorchLogImplementation class for the EM algorithm. This class implements the EM algorithm using PyTorch and logs to transform products into sums. This implementation is a bit slower than the PytorchImplementation class but use less memory.

    Methods
    -------
    e_step(X, alpha, pi)
        Performs the E-step of the EM algorithm.
    init_tau(n, Q)
        Initializes the tau matrix.
    m_step(X, tau)
        Performs the M-step of the EM algorithm.
    log_likelihood(X, alpha, pi, tau)
        Computes the log-likelihood.
    check_parameters(alpha, pi, tau)
        Checks if the parameters are valid.
    fixed_point_iteration(tau, X, alpha, pi)
        Performs a fixed-point iteration of the E-step.
    input(array)
        Processes input arrays.
    output(array)
        Processes output arrays.
    """

    def fixed_point_iteration(
        self, tau: torch.Tensor, X: torch.Tensor, alpha: torch.Tensor, pi: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs a fixed-point iteration of the E-step.

        Parameters
        ----------
        tau : torch.Tensor
            Current tau matrix.
        X : torch.Tensor
            The adjacency matrix of the graph.
        alpha : torch.Tensor
            Estimated alpha parameters.
        pi : torch.Tensor
            Estimated pi parameters.

        Returns
        -------
        torch.Tensor
            Updated tau matrix.
        """
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
    ) -> float:
        """
        Computes the log-likelihood.

        Parameters
        ----------
        X : torch.Tensor
            The adjacency matrix of the graph.
        alpha : torch.Tensor
            Estimated alpha parameters.
        pi : torch.Tensor
            Estimated pi parameters.
        tau : torch.Tensor
            Current tau matrix.

        Returns
        -------
        torch.Tensor
            Log-likelihood value.
        """
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
