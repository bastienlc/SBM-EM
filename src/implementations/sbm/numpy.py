from typing import Tuple

import numpy as np

from ...constants import *
from ..generic import GenericImplementation


class NumpyImplementation(GenericImplementation):
    """
    NumpyImplementation class for the EM algorithm. This implementation is based on numpy arrays so it is not optimized for speed.

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

    def input(self, array: np.ndarray) -> np.ndarray:
        """
        Processes the input array.

        Parameters
        ----------
        array : np.ndarray
            Input data to be processed.

        Returns
        -------
        np.ndarray
            Processed input data.
        """
        return array

    def output(self, array: np.ndarray) -> np.ndarray:
        """
        Processes the output array.

        Parameters
        ----------
        array : np.ndarray
            Output data to be processed.

        Returns
        -------
        np.ndarray
            Processed output data. Returns a numpy array.
        """
        return array

    def m_step(self, X: np.ndarray, tau: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs the M-step of the EM algorithm.

        Parameters
        ----------
        X : np.ndarray
            The adjacency matrix of the graph.
        tau : np.ndarray
            Current tau matrix.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Estimated alpha and pi parameters.
        """
        n = X.shape[0]
        alpha = np.sum(tau, axis=0) / n
        pi = (
            np.einsum("ij,iq,jl->ql", X, tau, tau)
            - np.einsum("ii,iq,il->ql", X, tau, tau)
        ) / (np.einsum("iq,jl->ql", tau, tau) - np.einsum("iq,il->ql", tau, tau))
        return alpha, pi

    def init_tau(self, n: int, Q: int) -> np.ndarray:
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
        np.ndarray
            Initialized tau matrix.
        """
        tau = np.random.rand(n, Q)
        return tau / np.sum(tau, axis=1)[:, np.newaxis]

    def _compute_b(self, X: np.ndarray, pi: np.ndarray) -> np.ndarray:
        """
        Computes the array of shape (n, Q, n, Q) used in fixed-point iterations.

        Parameters
        ----------
        X : np.ndarray
            The adjacency matrix of the graph.
        pi : np.ndarray
            Estimated pi parameters.

        Returns
        -------
        np.ndarray
            Computed array.
        """
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

    def fixed_point_iteration(
        self, tau: np.ndarray, X: np.ndarray, alpha: np.ndarray, pi: np.ndarray
    ) -> np.ndarray:
        """
        Performs a fixed-point iteration.

        Parameters
        ----------
        tau : np.ndarray
            Current tau matrix.
        X : np.ndarray
            The adjacency matrix of the graph.
        alpha : np.ndarray
            Estimated alpha parameters.
        pi : np.ndarray
            Estimated pi parameters.

        Returns
        -------
        np.ndarray
            Updated tau matrix.
        """
        b_values = self._compute_b(X, pi)
        tau = alpha[None, :] * np.prod(b_values**tau, axis=(2, 3))
        tau /= np.sum(tau, axis=1)[:, None]

        return tau

    def e_step(self, X, alpha, pi):
        """
        Performs the E-step of the EM algorithm. This method uses fixed-point iterations.

        Parameters
        ----------
        X : np.ndarray
            The adjacency matrix of the graph.
        alpha : np.ndarray
            Estimated alpha parameters.
        pi : np.ndarray
            Estimated pi parameters.

        Returns
        -------
        np.ndarray
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
                previous_tau = np.copy(tau)
                tau = 0.9 * tau + 0.1 * self.fixed_point_iteration(tau, X, alpha, pi)
                norm_change = np.linalg.norm(previous_tau - tau, ord=1)

                if norm_change < EPSILON:
                    break
        return tau

    def log_likelihood(self, X, alpha, pi, tau, elbo=True):
        """
        Computes the log-likelihood.

        Parameters
        ----------
        X : np.ndarray
            The adjacency matrix of the graph.
        alpha : np.ndarray
            Estimated alpha parameters.
        pi : np.ndarray
            Estimated pi parameters.
        tau : np.ndarray
            Current tau matrix.
        elbo : bool, optional
            If True, calculates the evidence lower bound, by default True.

        Returns
        -------
        np.ndarray
            Log-likelihood value.
        """
        n = X.shape[0]
        ll = 0
        ll += np.sum(
            tau * np.repeat(np.log(alpha)[np.newaxis, :], n, axis=0), axis=(0, 1)
        )
        if elbo:
            tau_log = tau * np.log(tau)
            tau_log = np.nan_to_num(tau_log, nan=0.0)  # Avoid NaN due to log(0)
            ll -= np.sum(tau_log, axis=(0, 1))
        b_values = self._compute_b(X, pi)
        log_b_values = np.log(b_values)
        ll += 1 / 2 * np.einsum("iq,jl,iqjl->", tau, tau, log_b_values)
        return ll

    def check_parameters(self, alpha, pi, tau):
        """
        Checks if the parameters are valid. This method may raise a ValueError if the parameters are not valid.

        Parameters
        ----------
        alpha : np.ndarray
            Estimated alpha parameters.
        pi : np.ndarray
            Estimated pi parameters.
        tau : np.ndarray
            Estimated tau matrix.

        Returns
        -------
        bool
            True if the parameters are valid, False otherwise.
        """
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
