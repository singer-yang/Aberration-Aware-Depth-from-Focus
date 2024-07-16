import torch
import cv2 as cv
from deeplens.psfnet import PSFNet
from torchvision.utils import save_image
from pfmreader import read_and_clean_pfm

if __name__ == "__main__":
    # Load lens and PSFNet
    psfnet = PSFNet(filename='./lenses/rf50mm/lens.json', sensor_res=(480, 640), kernel_size=11)
    psfnet.load_net('./ckpt/rf50mm/PSFNet480x640_ks11.pkl')
    psfnet.analysis()

    # Read image with using disp.pfm
    img = cv.resize(cv.cvtColor(cv.imread('./dataset/Middlebury2014/Adirondack-perfect/im0.png'), cv.COLOR_BGR2RGB),
                    (640, 480))
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255
    disp = read_and_clean_pfm(r'./dataset/Middlebury2014/Adirondack-perfect/disp0.pfm')

    disp = cv.resize(disp, (640, 480))
    disp = torch.tensor(disp).unsqueeze(0).unsqueeze(0).float()
    print("disp in pixel:",disp)

    # important:To convert from the floating-point disparity value d [pixels] in the .pfm file to depth Z [mm] the
    # following equation can be used: Z = f * baseline / (d + doffs)
    # these datas can be found in dataset/Middlebury2014/Adirondack-perfect/calib.txt

    base_line = 176.252
    doffs = 209.059
    fx = 4161.221
    depth_map = fx * base_line / (disp + doffs)

    depth_map = -depth_map

    # Output the depth value
    print("Calculated depth in mm:", depth_map)

    focus_dist = torch.tensor([-2000.])  # unit [mm]

    # Render an image
    defocused_img = psfnet.render(img.to(psfnet.device), depth_map.to(psfnet.device), focus_dist.to(psfnet.device))
    save_image(defocused_img, 'results/aberrated_defocused_img_2000.png')
    save_image(img, 'results/all_in_focus_img.png')
