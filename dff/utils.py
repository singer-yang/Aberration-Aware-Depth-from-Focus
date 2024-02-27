import torch
import numpy as np

def select_focus_dist(depth, num, mode='linear', center=True):
    """ Select focus distances from depth map. 

    Args:
        depth (tensor): [B, 1, H, W] tensor.
        num (int): focal stack size.
        mode (str, optional):
            "linear": sample linearly to approximate real-world applications.
            "importance": sample more data from close distances.
    """
    assert num > 3, 'Focal stack size is too small'
    B, C, H, W = depth.shape
    mask = (depth > 0)

    avg_depth = torch.sum(depth, dim=(1,2,3)) / torch.sum(mask, dim=(1,2,3))
    depth_max = torch.amax(depth, dim=(1,2,3))
    depth_min = torch.zeros_like(depth_max)
    for i in range(B):
        mask0 = mask[i,...]
        depth0 = depth[i, ...]
        depth_min[i] = torch.min(depth0[mask0>0])

    # Select focus distances
    if mode == 'linear':
        focus_dists = []
        for i in range(num):
            focus_dists.append(depth_min + i * (depth_max - depth_min) / (num - 1))

    elif mode == 'importance':
        focus_dists = [depth_max, depth_min]
        num = num - 2

        while len(focus_dists) < num:
            focus_dist = np.random.rand() * (depth_max - depth_min) + depth_min
            if focus_dist > avg_depth:
                accept_rate = (depth_max - focus_dist) / (depth_max - avg_depth)
            else:
                accept_rate = (focus_dist - depth_min) / (avg_depth - depth_min)

            accept = np.random.rand()
            if accept < accept_rate:
                focus_dists.append(focus_dist)
    else:
        raise NotImplementedError

    focus_dists = torch.stack(focus_dists, dim=1)
    focus_dists = torch.sort(focus_dists, dim=-1)[0]
    return focus_dists