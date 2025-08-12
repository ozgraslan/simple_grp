import numpy as np
import torch

def is_uint8(x):
    if isinstance(x, np.ndarray):
        return x.dtype == np.uint8
    elif isinstance(x, torch.Tensor):
        return x.dtype == torch.uint8
    else:
        raise NotImplementedError

