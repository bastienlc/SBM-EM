from typing import List

import numpy as np
import torch

from .constants import *


def sort_parameters(alpha, pi):
    if isinstance(alpha, np.ndarray):
        sort_indices = np.argsort(alpha)
        if pi.shape[0] == pi.shape[1] and np.all(pi - pi.T < PRECISION):
            return alpha[sort_indices], pi[sort_indices, :][:, sort_indices]
        else:
            return alpha[sort_indices], pi[sort_indices, :]
    if isinstance(alpha, torch.Tensor):
        sort_indices = torch.argsort(alpha)
        if pi.shape[0] == pi.shape[1] and torch.all(
            pi - torch.transpose(pi, 0, 1) < PRECISION
        ):
            return alpha[sort_indices], pi[sort_indices, :][:, sort_indices]
        else:
            return alpha[sort_indices], pi[sort_indices, :]


def drop_inits(
    inits: List[int],
    iteration: int,
    max_iterations: int,
    log_likelihoods: np.ndarray,
    inits_to_drop: List[int] = [],
):
    for init in inits_to_drop:
        try:
            inits.remove(init)
        except Exception:
            raise ValueError("Removed all initialisations")

    while (max_iterations - iteration) / 10 < len(inits) and len(inits) > 1:
        index_to_drop = np.argmin(log_likelihoods[inits, iteration])
        inits = np.delete(inits, index_to_drop)

    return inits


def b(x, pi):
    return pi**x * (1 - pi) ** (1 - x)
