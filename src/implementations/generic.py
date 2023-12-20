import numpy as np


class GenericImplementation:
    """
    An abstract class representing a generic implementation for the EM algorithm. It is used to define the interface of the different implementations. It also provides input and output processing methods to deal with different data types (numpy, torch, etc.).

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

    def __init__(self):
        """
        Initializes a GenericImplementation instance.
        """
        pass

    def e_step(self, X, alpha, pi):
        """
        Performs the E-step of the EM algorithm.

        Parameters
        ----------
        X : array
            The adjacency matrix of the graph.
        alpha : array
            Estimated alpha parameters.
        pi : array
            Estimated pi parameters.

        Returns
        -------
        array
            Updated tau matrix.
        """
        pass

    def init_tau(self, n: int, Q: int):
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
        array
            Initialized tau matrix.
        """
        pass

    def m_step(self, X, tau):
        """
        Performs the M-step of the EM algorithm.

        Parameters
        ----------
        X : array
            The adjacency matrix of the graph.
        tau : array
            Current tau matrix.

        Returns
        -------
        Tuple
            Estimated alpha and pi parameters.
        """
        pass

    def log_likelihood(self, X, alpha, pi, tau):
        """
        Computes the log-likelihood.

        Parameters
        ----------
        X : array
            The adjacency matrix of the graph.
        alpha : array
            Estimated alpha parameters.
        pi : array
            Estimated pi parameters.
        tau : array
            Current tau matrix.

        Returns
        -------
        float
            Log-likelihood value.
        """
        pass

    def check_parameters(self, alpha, pi, tau):
        """
        Checks if the parameters are valid. This method may raise a ValueError if the parameters are not valid.

        Parameters
        ----------
        alpha : array
            Estimated alpha parameters.
        pi : array
            Estimated pi parameters.
        tau : array
            Current tau matrix.

        Returns
        -------
        bool
            True if parameters are valid, False otherwise.
        """
        pass

    def fixed_point_iteration(self, tau, X, alpha, pi):
        """
        Performs a fixed-point iteration of the E-step. Not all implementations need this method.

        Parameters
        ----------
        tau : array
            Current tau matrix.
        X : array
            The adjacency matrix of the graph.
        alpha : array
            Estimated alpha parameters.
        pi : array
            Estimated pi parameters.

        Returns
        -------
        array
            Updated tau matrix.
        """
        pass

    def input(self, array):
        """
        Processes the input array.

        Parameters
        ----------
        array : array
            Input data to be processed.

        Returns
        -------
        array
            Processed input data. It can then be fed to the other methods.
        """
        pass

    def output(self, array) -> np.ndarray:
        """
        Processes the output array.

        Parameters
        ----------
        array : array
            Output data to be processed.

        Returns
        -------
        array
            Processed output data. Returns a numpy array.
        """
        pass
