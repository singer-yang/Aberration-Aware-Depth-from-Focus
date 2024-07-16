import torch
import cv2 as cv
from deeplens.psfnet import PSFNet
from torchvision.utils import save_image

if __name__ == "__main__":

    # Load lens and PSFNet
    psfnet = PSFNet(filename='./lenses/rf50mm/lens.json', sensor_res=(480, 640), kernel_size=11)
    psfnet.load_net('./ckpt/rf50mm/PSFNet480x640_ks11.pkl')
    psfnet.analysis()

    # Read image
    img = cv.resize(cv.cvtColor(cv.imread('./datasets/Middlebury2014/Adirondack-perfect/im0.png'), cv.COLOR_BGR2RGB), (640, 480))
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255
    depth = cv.resize(cv.imread(f'./datasets/Middlebury2014/Adirondack-perfect/depth.png', -1) / 1000., (640, 480))
    depth = torch.tensor(depth).unsqueeze(0).unsqueeze(0).float()
    
    # Render an image
    depth = - depth * 1e3 # unit [mm]
    focus_dist = torch.tensor([-2400.]) # unit [mm]
    defocused_img = psfnet.render(img.to(psfnet.device), depth.to(psfnet.device), focus_dist.to(psfnet.device))
    save_image(defocused_img, './aberrated_defocused_img.png')
    save_image(img, './all_in_focus_img.png')
