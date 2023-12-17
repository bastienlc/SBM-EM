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


def drop_init(n_init, tau_list, ll_list, to_drop=None):
    if n_init > 1:
        to_drop = to_drop if to_drop is not None else np.argmin(ll_list)
        return (
            n_init - 1,
            tau_list[:to_drop] + tau_list[to_drop + 1 :],
            np.delete(ll_list, to_drop),
        )
    else:
        return n_init, tau_list, ll_list


def b(x, pi):
    return pi**x * (1 - pi) ** (1 - x)
