import numpy as np
import torch

from .constants import *


def sort_parameters(alpha, pi):
    if isinstance(alpha, np.ndarray):
        sort_indices = np.argsort(alpha)
        return alpha[sort_indices], pi[sort_indices, :][:, sort_indices]
    if isinstance(alpha, torch.Tensor):
        sort_indices = torch.argsort(alpha)
        return alpha[sort_indices], pi[sort_indices, :][:, sort_indices]


def drop_init(n_init, tau_list, ll_list):
    if n_init > 1:
        worst_index = np.argmin(ll_list)
        return (
            n_init - 1,
            tau_list[:worst_index] + tau_list[worst_index + 1 :],
            np.delete(ll_list, worst_index),
        )
    else:
        return n_init, tau_list, ll_list


def b(x, pi):
    return pi**x * (1 - pi) ** (1 - x)
