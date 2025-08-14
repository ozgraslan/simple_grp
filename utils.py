import os
import random
import math
import numpy as np
import torch


def save_checkpoint(model, save_dir, step):
    # Create subfolder for the current checkpoint
    ckpt_dir = os.path.join(save_dir, f"{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save model (you can also save tokenizer/config here)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin"))
    print("Model saved to", os.path.join(ckpt_dir, "pytorch_model.bin"))


def is_uint8(x):
    if isinstance(x, np.ndarray):
        return x.dtype == np.uint8
    elif isinstance(x, torch.Tensor):
        return x.dtype == torch.uint8
    else:
        raise NotImplementedError

## adapted from https://github.com/Physical-Intelligence/openpi/blob/36dc3c037eb8a3921be9ecb94369d60cbf56f58f/src/openpi/models/pi0.py#L20
def get_att_mask(mask_arr, inp_mask):
    cumsum = torch.cumsum(mask_arr, dim=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = inp_mask[:, None, :] * inp_mask[:, :, None]
    ## need to add the diagonal to the mask
    ## because older torch versions give nans 
    ## if a token does not attend to anything
    return torch.logical_or(
        torch.logical_and(attn_mask, valid_mask),  
        torch.eye(valid_mask.shape[1], device=mask_arr.device).unsqueeze(0)
    )

## from openvla github
def set_seed_everywhere(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

## function taken from https://github.com/openvla/openvla/blob/main/experiments/robot/libero/libero_utils.py#L77
def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den