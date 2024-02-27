from deeplens.psfnet import *
from dff.dataset import *

def get_lens(args):
    ks = args['ks']
    sensor_res = args['res']
    device = args['device']

    train_lens_name = args['train']['lens']
    if train_lens_name == 'thinlens':
        foc_len = args['train']['foc_len']
        fnum = args['train']['fnum']
        sensor_size = [float(i) for i in args['train']['sensor_size']]
        train_lens = ThinLens(foc_len=foc_len, fnum=fnum, kernel_size=ks, sensor_size=sensor_size, sensor_res=sensor_res)
        train_lens = train_lens.to(device)
    else:
        train_lens = PSFNet(filename=train_lens_name, sensor_res=sensor_res, kernel_size=ks,device=device)
        train_lens.load_net(args['train']['psfnet_path'])

    test_lens_name = args['test']['lens']
    if test_lens_name == 'thinlens':
        foc_len = args['test']['foc_len']
        fnum = args['test']['fnum']
        sensor_size = [float(i) for i in args['test']['sensor_size']]
        test_lens = ThinLens(foc_len=foc_len, fnum=fnum, kernel_size=ks, sensor_size=sensor_size, sensor_res=sensor_res)
        test_lens = test_lens.to(device)
    else:
        test_lens = PSFNet(filename=test_lens_name, sensor_res=sensor_res, kernel_size=ks, device=device)
        test_lens.load_net(args['test']['psfnet_path'])

    return train_lens, test_lens

def get_dataset(args):
    train_dataset_name = args['train']['dataset']
    if train_dataset_name == 'Matterport3D':
        train_set = Matterport3D(args['train_aif_dir'], args['train_depth_dir'], resize=args['res'])
    elif train_dataset_name == 'FlyingThings3D':
        train_set = FlyingThings3D(args['FlyingThings3D_train'], resize=args['res'])
    else:
        raise NotImplementedError

    test_dataset_name = args['test']['dataset']
    if test_dataset_name == 'Middlebury2014':
        test_set = Middlebury(args['Middlebury2014_val'], resize=args['res'], train=False)
    elif test_dataset_name == 'Middlebury2021':
        test_set = Middlebury(args['Middlebury2021_val'], resize=args['res'], train=False)
    elif test_dataset_name == 'RealWorld':
        test_set = RealWorld(args['RealWorld_val'], resize=args['res'], depth=False)
    else:
        raise NotImplementedError
    
    return train_set, test_set