import torch
import numpy as np
import skimage.filters as skf
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# ================================
# Matrics for depth evaluation
# ================================
def abs_rel(est_depth, gt_depth):
    out = np.abs(gt_depth - est_depth) / (gt_depth)
    total_pixels = np.count_nonzero(~np.isinf(out))
    out[np.isinf(out)] = 0
    return np.sum(out) / total_pixels

def sq_rel(est_depth, gt_depth):
    out = np.power((gt_depth - est_depth), 2) / gt_depth
    total_pixels = np.count_nonzero(~np.isinf(out))
    out[np.isinf(out)] = 0
    return np.sum(out) / total_pixels

def mae(est_depth, gt_depth):
    return np.mean(np.abs(gt_depth - est_depth))

def mse(est_depth, gt_depth):
    return np.mean(np.power((gt_depth - est_depth), 2))

def rmse(est_depth, gt_depth):
    return np.sqrt(mse(est_depth, gt_depth))

def rmse_log(est_depth, gt_depth):
    gt_depth = np.log(gt_depth)
    est_depth = np.log(est_depth)
    total_pixels = np.count_nonzero((~np.isinf(est_depth)) * (~np.isinf(gt_depth)))
    out = np.power((gt_depth - est_depth), 2)
    out[np.isinf(out)] = 0
    return np.sqrt(np.sum(out) / total_pixels)

def accuracy_k(est_depth, gt_depth, k):
    thresh = np.maximum(est_depth/gt_depth, gt_depth/est_depth)
    total_pixels=np.count_nonzero(~np.isinf(thresh))
    Dp=np.where(thresh < (1.25**k), 1,0)
    return (np.sum(Dp)) / total_pixels

def get_bumpiness(gt, algo_result, mask, clip=0.05, factor=100):
    # https://github.com/albert100121/AiFDepthNet/blob/master/test.py
    if type(gt) == torch.Tensor:
        gt = gt.cpu().numpy()[0, 0]
    if type(algo_result) == torch.Tensor:
        algo_result = algo_result.cpu().numpy()[0, 0]
    if type(mask) == torch.Tensor:
        mask = mask.cpu().numpy()[0, 0]
    
    # Frobenius norm of the Hesse matrix
    diff = np.asarray(algo_result - gt, dtype='float64')
    dx = skf.scharr_v(diff)
    dy = skf.scharr_h(diff)
    dxx = skf.scharr_v(dx)
    dxy = skf.scharr_h(dx)
    dyy = skf.scharr_h(dy)
    dyx = skf.scharr_v(dy)
    bumpiness = np.sqrt(np.square(dxx) + np.square(dxy) + np.square(dyy) + np.square(dyx))
    bumpiness = np.clip(bumpiness, 0, clip)
    return np.mean(bumpiness[mask]) * factor

def get_bumpiness_non_mask(gt, algo_result, clip=0.05, factor=100):
    if type(gt) == torch.Tensor:
        gt = gt.cpu().numpy()[0, 0]
    if type(algo_result) == torch.Tensor:
        algo_result = algo_result.cpu().numpy()[0, 0]
    # Frobenius norm of the Hesse matrix
    diff = np.asarray(algo_result - gt, dtype='float64')
    dx = skf.scharr_v(diff)
    dy = skf.scharr_h(diff)
    dxx = skf.scharr_v(dx)
    dxy = skf.scharr_h(dx)
    dyy = skf.scharr_h(dy)
    dyx = skf.scharr_v(dy)
    bumpiness = np.sqrt(np.square(dxx) + np.square(dxy) + np.square(dyy) + np.square(dyx))
    bumpiness = np.clip(bumpiness, 0, clip)
    return np.mean(bumpiness) * factor

def AIF_DepthNEt_abs_rel(est, gt, mask):
    return np.mean(np.abs(est[mask] - gt[mask]) / gt[mask])

def AIF_DepthNEt_sq_rel(est, gt, mask):
    return np.mean(((est[mask] - gt[mask])**2) / gt[mask])

def mask_abs_rel(est_depth, gt_depth, mask):
    return np.mean(np.abs(gt_depth[mask] - est_depth[mask]) / (gt_depth[mask]))

def mask_sq_rel(est_depth, gt_depth, mask):
    return np.mean(np.power((gt_depth[mask] - est_depth[mask]), 2) / (gt_depth[mask]))

def mask_mse(est_depth, gt_depth, mask):
    return np.mean(np.power((gt_depth[mask] - est_depth[mask]), 2))

def mask_mae(est_depth, gt_depth, mask):
    return np.mean(np.abs(gt_depth[mask] - est_depth[mask]))

def mask_rmse(est_depth, gt_depth, mask):
    return np.sqrt(np.mean(np.power(est_depth[mask] - gt_depth[mask], 2)))

def mask_rmse_log(est_depth, gt_depth, mask):
    gt_depth = np.log(gt_depth[mask])
    est_depth = np.log(est_depth[mask])
    out = np.power((gt_depth - est_depth),2)
    return np.sqrt(np.mean(out))

def mask_accuracy_k(est_depth, gt_depth, k, mask):
    A = est_depth[mask] / gt_depth[mask]
    B = gt_depth[mask] / est_depth[mask]
    thresh = np.maximum(A, B)
    total_pixels = np.sum(mask)
    Dp = np.where(thresh < (1.25**k), 1,0)
    return (np.sum(Dp)) / total_pixels

def mask_mse_w_conf(est_depth, gt_depth, conf, mask):
    return np.sum(conf[mask] * (np.power((gt_depth[mask] - est_depth[mask]), 2))) / np.sum(conf[mask])

def mask_mae_w_conf(est_depth, gt_depth, conf, mask):
    return np.sum(conf[mask] * (np.abs(gt_depth[mask] - est_depth[mask]))) / np.sum(conf[mask])

def mask_mse_w_conf_wo_mask(est_depth, gt_depth, conf):
    return np.sum(conf * (np.power((gt_depth - est_depth), 2))) / np.sum(conf)

def mask_mae_w_conf_wo_mask(est_depth, gt_depth, conf):
    return np.sum(conf * (np.abs(gt_depth - est_depth))) / np.sum(conf)


# ================================
# Matrics for aif image evaluation
# ================================
def batch_PSNR(img, img_clean):
    """ Compute PSNR for image batch.
    """
    Img = img.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
    Img_clean = img_clean.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Img_clean[i,:,:,:], Img[i,:,:,:])
    return round(PSNR/Img.shape[0], 4)

def batch_SSIM(img, img_clean):
    """ Compute SSIM for image batch.
    """
    Img = img.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
    Img_clean = img_clean.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
    SSIM = 0
    for i in range(Img.shape[0]):
        SSIM += compare_ssim(Img_clean[i,...], Img[i,...], channel_axis=0)
    return round(SSIM/Img.shape[0], 4)

def mask_psnr(est_aif, gt_aif):
    return batch_PSNR(est_aif, gt_aif)

def mask_ssim(est_aif, gt_aif):
    return batch_SSIM(est_aif, gt_aif)