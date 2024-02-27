""" 
Implicate representation of PSF.

Input [x, y, z, foc_dist]. Output [ks, ks] PSF kernel.
"""
import os
from datetime import datetime
from deeplens.psfnet import PSFNet
from deeplens.utils import set_logger, set_seed

result_dir = f'./results/' + datetime.now().strftime("%m%d-%H%M%S") + '-psfnet'
os.makedirs(result_dir, exist_ok=True)
set_logger(result_dir)
set_seed(0)

if __name__ == "__main__":

    psfnet = PSFNet(filename='./lenses/rf50mm/lens.json', sensor_res=(480, 640), kernel_size=11)
    psfnet.analysis(save_name=f'{result_dir}/lens')
    psfnet.write_lens_json(f'{result_dir}/lens.json')

    psfnet.load_net('./ckpt/rf50mm/PSFNet480x640_ks11.pkl')
    psfnet.train_psfnet(iters=100000, bs=128, lr=1e-4, spp=4096, evaluate_every=100, result_dir=result_dir)
    psfnet.evaluate_psf(result_dir=result_dir)
    
    print('Finish PSF net fitting.')