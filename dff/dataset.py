import torch
from torch.utils.data import Dataset
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2 as cv
import numpy as np
from glob import glob
from torchvision import transforms
from skimage.morphology import disk, closing
import random
from scipy.ndimage.interpolation import rotate


# ================================
# Dataset
# ================================
class Matterport3D(Dataset):
    def __init__(self, rgb_path, depth_path, resize=None, train=True):
        super(Matterport3D, self).__init__()

        self.rgb_path = rgb_path
        self.depth_path = depth_path
        self.scenes = [scene.split('/')[-1] for scene in glob(f'{rgb_path}/*')]
        self.resize = resize
        self.train = train
        
        self.imgs = []
        self.depths = []
        for scene in self.scenes:
            imgs = sorted(glob(f'{rgb_path}/{scene}/undistorted_color_images/*.jpg'))
            depths = sorted(glob(f'{depth_path}/{scene}/render_depth/*.png'))
            self.imgs += imgs
            self.depths += depths

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, antialias=True)
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        aif_img = cv.cvtColor(cv.imread(self.imgs[idx]), cv.COLOR_BGR2RGB) / 255.
        depth = cv.imread(self.depths[idx], -1) / 4000 #* 1e3 # convert to [mm]
        if self.train:
            aif_img, depth = AutoAgument(aif_img, depth)
        
        aif_img = self.transform(aif_img.astype('float32'))
        depth = self.transform(depth.astype('float32'))

        return [aif_img, depth]


class FlyingThings3D(Dataset):
    def __init__(self, dataset_dir, resize=None, train=True, fs_num=0):
        super(FlyingThings3D, self).__init__()

        self.dataset_dir = dataset_dir
        self.scenes = [scene.split('/')[-1] for scene in glob(f'{dataset_dir}/*')]
        self.resize = resize
        self.fs_num = fs_num
        self.train = train

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, antialias=True)
        ])

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        dataset_dir = self.dataset_dir
        scene = self.scenes[index]
        DEPTH_FACTOR = 20
        resize = [self.resize[1], self.resize[0]]

        depth = cv.resize(cv.imread(f'{dataset_dir}/{scene}/disp.exr', cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH) / DEPTH_FACTOR, resize)
        
        if self.fs_num > 0:
            focused_imgs = []
            focal_dists = []
            full_focal_stack = sorted(glob(f'{dataset_dir}/{scene}/*.png'))[:-1]
            selected_imgs = random.sample(full_focal_stack, self.fs_num)
            for img_name in selected_imgs:
                focal_dists.append(float(img_name.split('/')[-1][:-4]) / DEPTH_FACTOR)
                focused_img = cv.resize(cv.imread(img_name).astype(np.float32)/255., resize)
                focused_imgs.append(focused_img)
            
            focal_stack = np.stack(focused_imgs, axis=-1)
            
            if self.train:
                focal_stack, depth = AutoAgument(focal_stack, depth)
                
            focal_stack = np.transpose(focal_stack, (3, 2, 0, 1))   # shape of (S, C, H, W)
            focal_stack = torch.from_numpy(focal_stack.astype('float32'))
            depth = torch.from_numpy(depth.astype('float32')).unsqueeze(0)
            focal_dists = torch.from_numpy(np.stack(focal_dists, axis=-1))    
            return [focal_stack, depth, focal_dists]

        else:
            aif_img = cv.cvtColor(cv.imread(f'{dataset_dir}/{scene}/AiF.png'), cv.COLOR_BGR2RGB) / 255.
            
            if self.train:
                aif_img, depth = AutoAgument(aif_img, depth)
            
            aif_img = self.transform(aif_img.astype('float32'))
            depth = self.transform(depth.astype('float32'))
            return [aif_img, depth]

# class Middlebury_FS(Dataset):
#     def __init__(self, dataset_dir, resize=None, train=False, fs_num=0):
#         super(Middlebury_FS).__init__()

#         self.dataset_dir = dataset_dir
#         self.scenes = [scene.split('/')[-1] for scene in glob(f'{dataset_dir}/*')]
#         self.resize = resize
#         self.fs_num = fs_num
#         self.train = train

#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Resize(resize, antialias=True)
#         ])

#     def __len__(self):
#         return len(self.scenes)

#     def __getitem__(self, index):
#         dataset_dir = self.dataset_dir
#         scene = self.scenes[index]
#         DEPTH_FACTOR = 10   # convert disparity to depth
#         resize = [self.resize[1], self.resize[0]]

#         depth = cv.resize(cv.imread(f'{dataset_dir}/{scene}/disp.exr', cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH) / DEPTH_FACTOR, resize)

#         if self.fs_num > 0:
#             raise Exception('Untested.')
#             focused_imgs = []
#             focal_dists = []
#             full_focal_stack = sorted(glob(f'{dataset_dir}/{scene}/*.png'))[:-1]
#             for _ in range(self.fs_num):
#                 focused_img = random.choice(full_focal_stack)
#                 focal_dists.append(float(focused_img.split('/')[-1][:-4]) / DEPTH_FACTOR)
#                 focused_img = cv.resize(cv.imread(focused_img).astype(np.float32)/255., resize)
#                 focused_imgs.append(focused_img)
            
#             focal_stack = np.stack(focused_imgs, axis=-1)
            
#             if self.train:
#                 focal_stack, depth = AutoAgument(focal_stack, depth)
                
#             focal_stack = np.transpose(focal_stack, (3, 2, 0, 1))
#             focal_stack = torch.from_numpy(focal_stack.astype('float32'))
#             depth = torch.from_numpy(depth.astype('float32')).unsqueeze(0)
#             focal_dists = torch.from_numpy(np.stack(focal_dists, axis=-1))
            
#             return [focal_stack, depth, focal_dists]

#         else:
#             aif_img = cv.cvtColor(cv.imread(f'{dataset_dir}/{scene}/AiF.png'), cv.COLOR_BGR2RGB) / 255.
            
#             if self.train:
#                 aif_img, depth = AutoAgument(aif_img, depth)
            
#             aif_img = self.transform(aif_img.astype('float32'))
#             depth = self.transform(depth.astype('float32'))

#             return [aif_img, depth]


class Middlebury(Dataset):
    def __init__(self, dataset_dir, resize=None, train=False):
        super(Middlebury).__init__()

        self.dataset_dir = dataset_dir
        self.scenes = sorted([scene.split('/')[-1] for scene in glob(f'{dataset_dir}/*')])
        self.resize = resize
        self.train = train

        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, antialias=True)
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        dataset_dir = self.dataset_dir
        scene = self.scenes[index]
        resize = [self.resize[1], self.resize[0]]

        aif_img = cv.cvtColor(cv.imread(f'{dataset_dir}/{scene}/im0.png'), cv.COLOR_BGR2RGB) / 255.
        depth = cv.resize(cv.imread(f'{dataset_dir}/{scene}/depth.png', -1) / 1000, resize)

        aif_img = self.train_transform(aif_img.astype('float32'))
        depth = self.train_transform(depth.astype('float32'))

        return [aif_img, depth]


class RealWorld(Dataset):
    def __init__(self, dataset_dir, resize=None, depth=False):
        super(RealWorld).__init__()

        self.dataset_dir = dataset_dir
        self.scenes = sorted([scene.split('/')[-1] for scene in glob(f'{dataset_dir}/*')])
        self.resize = resize
        self.depth = depth

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        dataset_dir = self.dataset_dir
        scene = self.scenes[index]
        resize = [self.resize[1], self.resize[0]]

        focused_imgs = []
        focal_dists = []
        full_focal_stack = sorted(glob(f'{dataset_dir}/{scene}/align/*.png')) + sorted(glob(f'{dataset_dir}/{scene}/*.JPG')) + sorted(glob(f'{dataset_dir}/{scene}/*.png'))
        for img_name in full_focal_stack:
            focal_dists.append(float(img_name.split('/')[-1].split('_')[1][4:]) / 1000)
            focused_img = cv.resize(cv.imread(img_name).astype(np.float32)/255., resize)
            focused_imgs.append(focused_img)
        
        focal_stack = np.stack(focused_imgs, axis=-1)
        focal_stack = np.transpose(focal_stack, (3, 2, 0, 1))
        focal_stack = torch.from_numpy(focal_stack.astype('float32'))
        
        focal_dists = torch.from_numpy(np.stack(focal_dists, axis=-1))

        if self.depth:
            depth = cv.resize(cv.imread(f'{dataset_dir}/{scene}/depth/depth.png', -1), resize)
            depth = (depth / 65535 * 3000 + 500) / 1000    # depth map is generated by Blender and stored in 16-bit .png file. The start is 600mm and the depth is 6000mm.
            depth = torch.from_numpy(depth.astype('float32')).unsqueeze(0)
        else:
            depth = torch.zeros_like(focal_stack[0,0,:,:].unsqueeze(0))
        
        return [focal_stack, depth, focal_dists]


# ================================
# Data augmentation
# ================================
def AutoAgument(img, depth):
    """ Automatic data augmentation.

    Args:
        img: [H, W, 3] ndarray
        depth: [H, W] ndarray 
    """
    # Color jitter
    if np.random.rand() > 0.5:
        contrast = np.random.rand()
        brightness = np.random.rand()
        img = np.clip((0.5 + contrast * (img - 0.5)) + brightness, 0.0, 1.0)
    
    # Flip
    if np.random.rand() > 0.5:
        img = np.flip(img, 1)
        depth = np.flip(depth, 1)

    # Flip
    if np.random.rand() > 0.5:
        img = np.flip(img, 0)
        depth = np.flip(depth, 0)

    # Rotate
    if np.random.rand() > 0.5:
        degree = np.random.randint(0, 180)
        if len(img.shape) == 4:
            for i in range(img.shape[-1]):
                img[...,i] = rotate(img[..., i], degree, reshape=False)
        else:
            img = rotate(img, degree, reshape=False)
        depth = rotate(depth, degree, reshape=False)
        depth[depth<0] = 0

    return img, depth