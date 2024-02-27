""" Aberrationn-aware depth-from-focus (DFF).
    
Thin lens for training, real lens for testing: conventional DFF works can not generalize well for real camera lenses in the real world.
Real lens for training, real lens for testing: our aberration-aware method can generalize well in the real world with only synthetic data.

Use AiFNet for training and evaluation.
"""
import os
import yaml
import wandb
import time
import logging
import cv2 as cv
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.optim as optim
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from deeplens.utils import set_seed, set_logger
from deeplens.psfnet import *
from dff import *

def config():
    with open('configs/aber_aware_dff_aif.yml') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    
    # Device
    num_gpus = torch.cuda.device_count()
    args['num_gpus'] = num_gpus
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    args['device'] = device
    logging.info(f'Using {num_gpus} GPUs')

    # Result folder
    result_dir = f'./results/' + datetime.now().strftime("%m%d-%H%M%S") + '-AberAware_DFF_AiFNet'
    args['results_dir'] = result_dir
    os.makedirs(result_dir, exist_ok=True)
    logging.info(f'Result folder: {result_dir}')
    
    # Logger
    set_logger(result_dir)

    # Random seed
    set_seed(126)
    torch.set_default_dtype(torch.float32)
    
    return args

def train(args):
    device = args['device']

    # Lens
    train_lens, test_lens = get_lens(args)
    
    # Depth-from-focus network
    if args['pred_name'] == 'depth':
        aif_args = {'device':device, 'task':'D_FS', 'stack_num':args['n_stack']}
    elif args['pred_name'] == 'aif':
        aif_args = {'device':device, 'task':'A_FS', 'stack_num':args['n_stack']}
    args['aif_args'] = aif_args
    
    dff_net = AiFDepthNet(n_stack=args['n_stack'])
    dff_net = nn.DataParallel(dff_net)
    if args['train']['dffnet_pretrained']:
        dff_net.load_state_dict(torch.load(args['train']['dffnet_pretrained']))
    dff_net = dff_net.to(device)

    # Dataset
    train_set, val_set = get_dataset(args)
    train_loader = DataLoader(train_set, batch_size=args['bs'])
    val_loader = DataLoader(val_set, batch_size=1)
    print(f'Totally {len(train_set)} images for training, {len(val_set)} images for test.')

    # Optimizer
    optimizer = optim.Adam(dff_net.parameters(), lr=float(args['lr']))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs']*len(train_set), eta_min=0)

    # Training
    args['mse_min'] = 100
    args['acc1_max'] = 0.0
    for epoch in range(args['epochs'] + 1):

        # Evaluation
        if epoch % 1 == 0 and epoch > 0:
            validate(dff_net, test_lens, val_loader, epoch, len(val_set), args)

        # Training
        dff_net.train()
        for sample in tqdm(train_loader):
            # Input data
            aif, depth = sample
            aif = aif.to(device)
            depth = depth.to(device)    # real depth in [m]
            mask = (depth > 0) 

            # Render focal stack
            with torch.no_grad():
                # Select random focus distance
                avg_depth = torch.sum(depth, dim=(1,2,3)) / torch.sum(mask, dim=(1,2,3))
                if torch.sum(torch.isnan(avg_depth)):
                    continue
                focus_dists = select_focus_dist(depth, args['n_stack'], mode='linear')
                
                # Simulate focal stack
                focal_stack = []
                for i in range(args['n_stack']):
                    foc_dist = focus_dists[:, i]
                    defocus_img = train_lens.render(aif, depth=-depth*1e3, foc_dist=-foc_dist*1e3)
                    focal_stack.append(defocus_img)
                focal_stack = torch.stack(focal_stack, dim=2)   # shape of [B, C, S, H, W]
            
            torch.cuda.empty_cache()

            # Forward-backward optimization
            input_dict = {'stack_rgb_img':focal_stack, 'focus_position':focus_dists, 'depth':depth, 'AiF_img':aif}
            losses, outputs = dff_net(input_dict, aif_args)

            optimizer.zero_grad()
            loss = losses['total'].mean()
            loss.backward()            
            optimizer.step()
            scheduler.step()


@torch.no_grad()
def validate(net, test_lens, valid_dataloader, epoch, num_val, args):
    
    net.eval()
    result_img_dir = f'{args["results_dir"]}/reults/'
    os.makedirs(result_img_dir, exist_ok=True)
    device = args['device']
    aif_args = args['aif_args']
    
    # Score for depth prediction
    Avg_abs_rel = 0.0
    Avg_sq_rel = 0.0
    Avg_mse = 0.0
    Avg_mae = 0.0
    Avg_rmse = 0.0
    Avg_rmse_log = 0.0
    Avg_accuracy_1 = 0.0
    Avg_accuracy_2 = 0.0
    Avg_accuracy_3 = 0.0

    # Score for aif prediction
    Avg_psnr = 0.0
    Avg_ssim = 0.0

    val_time = 0.0
    for idx, samples in enumerate(tqdm(valid_dataloader, desc="valid")):
        
        # Generate input
        aif, gt_depth = samples
        aif = aif.to(device)
        gt_depth = gt_depth.to(device)    # depth in [m]
        test_mask = gt_depth.detach().clone() > 0
        avg_depth = torch.sum(gt_depth, dim=(1,2,3)) / torch.sum(test_mask, dim=(1,2,3))
        if torch.sum(torch.isnan(avg_depth)):
            continue
        
        # Render DoF image for input
        focal_stack = []
        focus_dists = select_focus_dist(gt_depth, args['n_stack'], mode='linear')
        for i in range(args['n_stack']):
            foc_dist = focus_dists[:, i]
        
            dof_img = test_lens.render(aif, depth = - gt_depth * 1e3, foc_dist = - foc_dist * 1e3)
            focal_stack.append(dof_img)

        torch.cuda.empty_cache()

        test_focal_stack = torch.stack(focal_stack, dim=2)  # shape of [B, C, S, H, W]
        test_focus_dists = focus_dists

        # Inference
        test_input_dict = {'stack_rgb_img': test_focal_stack, 'focus_position':test_focus_dists, 'depth':gt_depth}
        
        start = time.time()            
        test_outputs = net.module.inference(test_input_dict, aif_args)
        val_time = val_time + (time.time() - start)
        
        pred_depth = test_outputs['pred_depth']
        pred_aif = test_outputs['pred_AiF_img']
        
        # Depth score matrics
        test_mask = np.squeeze(test_mask.data.cpu().numpy())
        gt_depth = np.squeeze(gt_depth.data.cpu().numpy())
        pred_depth = np.squeeze(pred_depth.data.cpu().numpy())

        Avg_abs_rel = Avg_abs_rel + mask_abs_rel(pred_depth, gt_depth, test_mask)
        Avg_sq_rel = Avg_sq_rel + mask_sq_rel(pred_depth, gt_depth, test_mask)
        Avg_mse = Avg_mse + mask_mse(pred_depth, gt_depth, test_mask)
        Avg_mae = Avg_mae + mask_mae(pred_depth, gt_depth, test_mask)
        Avg_rmse = Avg_rmse + mask_rmse(pred_depth, gt_depth, test_mask)
        Avg_rmse_log = Avg_rmse_log + mask_rmse_log(pred_depth, gt_depth, test_mask)
        Avg_accuracy_1 = Avg_accuracy_1 + mask_accuracy_k(pred_depth, gt_depth, 1, test_mask)
        Avg_accuracy_2 = Avg_accuracy_2 + mask_accuracy_k(pred_depth, gt_depth, 2, test_mask)
        Avg_accuracy_3 = Avg_accuracy_3 + mask_accuracy_k(pred_depth, gt_depth, 3, test_mask)

        # Save depth images
        pred_depth = (pred_depth / gt_depth.max() * 255.).astype(np.uint8)
        gt_depth = (gt_depth / gt_depth.max() * 255.).astype(np.uint8)
        cv.imwrite(f'{result_img_dir}/img{idx}_pred.png', cv.applyColorMap(pred_depth, cv.COLORMAP_JET))
        cv.imwrite(f'{result_img_dir}/img{idx}_gt.png', cv.applyColorMap(gt_depth, cv.COLORMAP_JET))

        # AiF score matrics
        gt_aif = aif.detach().clone().cpu()
        pred_aif = pred_aif.detach().clone().cpu()
        Avg_psnr = Avg_psnr + mask_psnr(pred_aif, gt_aif)
        Avg_ssim = Avg_ssim + mask_ssim(pred_aif, gt_aif)

        # Save AiF images
        save_image(pred_aif, f'{result_img_dir}/img{idx}_pred_aif.png', normalize=True)
        save_image(gt_aif, f'{result_img_dir}/img{idx}_gt_aif.png', normalize=True)
        
    # Save model (last and best)
    torch.save(net.state_dict(), f'{args["results_dir"]}/depth_net_last.pkl')
    if Avg_mse / num_val < args['mse_min']:
        args['mse_min'] = Avg_mse / num_val
        torch.save(net.state_dict(), f'{args["results_dir"]}/depth_net_best.pkl')
    if Avg_accuracy_1 / num_val > args['acc1_max']:
        args['acc1_max'] = Avg_accuracy_1 / num_val
        torch.save(net.state_dict(), f'{args["results_dir"]}/depth_net_best_acc1.pkl')

    # Log scores
    logging.info(f"Avg_abs_rel({epoch}): {Avg_abs_rel / num_val}")
    logging.info(f"Avg_sq_rel({epoch}): {Avg_sq_rel / num_val}")
    logging.info(f"Avg_mse({epoch}): {Avg_mse / num_val}")
    logging.info(f"Avg_mae({epoch}): {Avg_mae / num_val}")
    logging.info(f"Avg_rmse({epoch}): {Avg_rmse / num_val}")
    logging.info(f"Avg_rmse_log({epoch}): {Avg_rmse_log / num_val}")
    logging.info(f"Avg_accuracy_1({epoch}): {Avg_accuracy_1 / num_val}")
    logging.info(f"Avg_accuracy_2({epoch}): {Avg_accuracy_2 / num_val}")
    logging.info(f"Avg_accuracy_3({epoch}): {Avg_accuracy_3 / num_val}")
    logging.info("\n")
    logging.info(f"Avg_psnr({epoch}): {Avg_psnr / num_val}")
    logging.info(f"Avg_ssim({epoch}): {Avg_ssim / num_val}")
    logging.info("\n")
    logging.info(f"AVG_time: {val_time / num_val}")
    logging.info("\n")

if __name__=='__main__':
    args = config()
    train(args)