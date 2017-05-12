import torch
import numpy as np


def normalize(n, minN=-1, maxN=1):
    """
    Normalize between -1 and 1
    """
    if type(n) == np.ndarray:
        min_ = np.min(n)
        max_ = np.max(n)

    elif type(n) == torch.Tensor:
        min_ = n.min()
        max_ = n.max()

    if min_ > max_:
        return None

    return ((maxN - minN) * ((n - min_) / (max_ - min_))) + minN
