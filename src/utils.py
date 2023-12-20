from typing import List, Tuple, Union

import numpy as np
import torch

from .constants import *


def sort_parameters(
    alpha: Union[np.ndarray, torch.Tensor], pi: Union[np.ndarray, torch.Tensor]
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Sorts the parameters in ascending order of alpha. This helps enforcing the unicity of the solution.

    Parameters
    ----------
    alpha : Union[np.ndarray, torch.Tensor]
        The alpha parameters of the mixture model.
    pi : Union[np.ndarray, torch.Tensor]
        The pi parameters of the mixture model.

    Returns
    -------
    Union[Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]
        The sorted alpha and pi parameters.
    """
    if isinstance(alpha, np.ndarray):
        sort_indices = np.argsort(alpha)
        if pi.shape[0] == pi.shape[1] and np.all(
            pi - pi.T < PRECISION
        ):  # Symmetric matrix case
            return alpha[sort_indices], pi[sort_indices, :][:, sort_indices]
        else:  # Non-symmetric matrix case
            return alpha[sort_indices], pi[sort_indices, :]

    if isinstance(alpha, torch.Tensor):
        sort_indices = torch.argsort(alpha)
        if pi.shape[0] == pi.shape[1] and torch.all(
            pi - torch.transpose(pi, 0, 1) < PRECISION
        ):  # Symmetric matrix case
            return alpha[sort_indices], pi[sort_indices, :][:, sort_indices]
        else:  # Non-symmetric matrix case
            return alpha[sort_indices], pi[sort_indices, :]


def drop_inits(
    inits: List[int],
    iteration: int,
    max_iterations: int,
    log_likelihoods: np.ndarray,
    inits_to_drop: List[int] = [],
) -> List[int]:
    """
    Drops the initialisations that are not performing well. At the end of the algorithm, only the best initialisation is kept.

    Parameters
    ----------
    inits : List[int]
        The list of current initialisations.
    iteration : int
        The current iteration.
    max_iterations : int
        The maximum number of iterations.
    log_likelihoods : np.ndarray
        The log likelihoods of each initialisation at each iteration.
    inits_to_drop : List[int], optional
        The initialisations to manually drop, if they raise errors for instance.

    Returns : List[int]
        The list of initialisations to keep.
    """

    # Initialisations to manually drop
    for init in inits_to_drop:
        try:
            inits.remove(init)
        except Exception:
            raise ValueError(
                "Tried to remove an initialisation that was already removed."
            )

    # Drops around 10% of the worst initialisations for each 10% of the iterations
    while (max_iterations - iteration) / 10 < len(inits) and len(inits) > 1:
        index_to_drop = np.argmin(log_likelihoods[inits, iteration])
        inits = np.delete(inits, index_to_drop)

    return inits
