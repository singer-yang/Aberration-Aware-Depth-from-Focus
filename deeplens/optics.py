""" Lensgroup class. 

Use geometric ray tracing to optical computation.
"""
import torch
import random
import json
import cv2 as cv
from tqdm import tqdm
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid

from .surfaces import *
from .utils import *
from .monte_carlo import *
from .render_psf import *
from .basics import GEO_SPP, EPSILON, WAVE_SPEC

class Lensgroup(DeepObj):
    """
    The Lensgroup (consisted of multiple optical surfaces) is mounted on a rod, whose
    origin is `origin`. The Lensgroup has full degree-of-freedom to rotate around the
    x/y axes, with the rotation angles defined as `theta_x`, `theta_y`, and `theta_z` (in degree).

    In Lensgroup's coordinate (i.e. object frame coordinate), surfaces are allocated
    starting from `z = 0`. There is an additional, comparatively small 3D origin shift
    (`shift`) between the surface center (0,0,0) and the origin of the mount, i.e.
    shift + origin = lensgroup_origin.
    
    There are two configurations of ray tracing: forward and backward. In forward mode,
    rays start from `d = 0` surface and propagate along the +z axis; In backward mode,
    rays start from `d = d_max` surface and propagate along the -z axis.

    Elements:
    ``` == Varaiables                               Requires gradient
        origin [Tensor]:                            ?
        shift [Tensor]:                             ?
        theta_x [Tensor]:                           ?
        theta_y [Tensor]:                           ?
        theta_z [Tensor]:                           ?
        to_world [Tranformation]:                   ?
        to_object [Transformation]:                 ?
        surfaces (list):                            True
            r                                       float
            d                                       tensor
            c                                       tensor
            k                                       tensor
            ai                                      tensor list
        materials (list):                           

        == Float/Int:
        aper_idx: aperture index.                   False
        foclen                                      False
        fnum                                        False
        fov(half diagonal fov)                      False
        imgh(diagonal sensor distance)              False
        sensor:
            sensor_size
            sensor_res
            pixel_size
            r_last(half diagonal distance)
            d_sensor
            focz

        == String/Boolean:
        lens_name [string]:                         False
        device [string]: 'cpu' or 'cuda:0'          False
        # mts_prepared [Bool]:                      False
        sensor_prepared [Bool]:                     False
        
    ```

    Methods (divided into some groups):
    ```
        Init
        Ray sampling
        Ray tracing
        Ray-tracing-based rendering
        PSF
        Geometrical optics (calculation)
        Lens operation
        Visualization
        Calibration and distortion (will be removed later?? to move into geometrical optics)
        Mask (for automated lens design, will be removed later, or move to optimazation)
        Loss function
        Optimization
        Lens field IO
        Others
    ```

    """
    def __init__(self, filename=None, sensor_res=(1024, 1024), use_roc=False, post_computation=True, device=DEVICE):
        """ Initialize Lensgroup.

        Args:
            filename (string): lens file.
            device ('cpu' or 'cuda'): We need to spercify device here, because `sample_ray` needs it.
            sensor_res: (H, W)
        """
        super(Lensgroup, self).__init__()
        self.device = device

        # Load lens file.
        if filename is not None:
            self.lens_name = filename
            self.load_file(filename, use_roc, sensor_res)            
        else:
            self.sensor_res = sensor_res
            self.surfaces = []
            self.materials = []
        
        self.to(device)

    def load_file(self, filename, use_roc, sensor_res):
        """ Load lens from .txt file.

        Args:
            filename (string): lens file.
            use_roc (bool): use radius of curvature (roc) or not. In the old code, we store lens data in roc rather than curvature.
            post_computation (bool): compute fnum, fov, foclen or not.
            sensor_res (list): sensor resolution.
        """
        if filename[-4:] == '.txt':
            self.surfaces, self.materials, self.r_last, d_last = self.read_lensfile(filename, use_roc)
            self.d_sensor = d_last + self.surfaces[-1].d.item()
            self.focz = self.d_sensor
        
        elif filename[-5:] == '.json':
            self.read_lens_json(filename)

        else:
            raise Exception("File format not supported.")

        # Lens calculation
        self.find_aperture()
        self.prepare_sensor(sensor_res)
        self.diff_surf_range = self.find_diff_surf()
        self.post_computation()


    def load_external(self, surfaces, materials, r_last, d_sensor):
        """ Load lens from extrenal surface/material list.
        """
        self.surfaces = surfaces
        self.materials = materials
        self.r_last = r_last
        self.d_sensor = d_sensor
        

    def prepare_sensor(self, sensor_res=[512, 512], sensor_size=None):
        """ Create sensor. 

            reference values:
                Nikon z35 f1.8: diameter = 1.912 [cm] ==> But can we just use [mm] in our code?
                Congli's caustic example: diameter = 12.7 [mm]
        Args:
            sensor_res (list): Resolution, pixel number.
            pixel_size (float): Pixel size in [mm].

            sensor_res: (H, W)
        """
        sensor_res = [sensor_res, sensor_res] if isinstance(sensor_res, int) else sensor_res
        self.sensor_res = sensor_res
        H, W = sensor_res
        if sensor_size is None:
            self.sensor_size = [2 * self.r_last * H / np.sqrt(H**2 + W**2), 2 * self.r_last * W / np.sqrt(H**2 + W**2)]
        else:
            self.sensor_size = sensor_size
            self.r_last = np.sqrt(sensor_size[0]**2 + sensor_size[1]**2) / 2

        assert self.sensor_size[0] / self.sensor_size[1] == H / W, "Pixel is not square."
        self.pixel_size = self.sensor_size[0] / sensor_res[0]


    def post_computation(self):
        """ After loading lens, compute foclen, fov and fnum.
        """
        self.find_aperture()
        self.hfov = self.calc_fov()
        self.foclen = self.calc_efl()
        
        # if self.aper_idx is not None:
        avg_pupilz, avg_pupilx = self.entrance_pupil()
        self.fnum = self.foclen / avg_pupilx / 2


    def find_aperture(self):
        """ Find aperture by last and next material.
        """
        self.aper_idx = None
        for i in range(len(self.surfaces)-1):
            # if self.materials[i].A < 1.0003 and self.materials[i+1].A < 1.0003: # AIR or OCCLUDER
            if self.surfaces[i].mat1.A < 1.0003 and self.surfaces[i].mat2.A < 1.0003:
                self.aper_idx = i
                return

    def find_diff_surf(self):
        """ Get surface indices without aperture.
        """
        if self.aper_idx is None:
            diff_surf_range = range(len(self.surfaces))
        else:
            diff_surf_range = list(range(0, self.aper_idx)) + list(range(self.aper_idx+1, len(self.surfaces)))
        return diff_surf_range


    # ====================================================================================
    # Ray Sampling
    # ====================================================================================
    @torch.no_grad()
    def sample_parallel_2D(self, R=None, wvln=DEFAULT_WAVE, z=None, view=0.0, M=15, forward=True, entrance_pupil=False):
        """ Sample 2D parallel rays. Rays have shape [M, 3].
        
            Used for (1) drawing lens setup, (2) paraxial optics calculation, for example, refocusing to infinity

        Args:
            R (float, optional): sampling radius. Defaults to None.
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.
            z (float, optional): sampling depth. Defaults to None.
            view (float, optional): incident angle (in degree). Defaults to 0.0.
            M (int, optional): ray number. Defaults to 15.
            forward (bool, optional): forward or backward rays. Defaults to True.
            entrance_pupil (bool, optional): whether to use entrance pupil. Defaults to False.
        """
        if entrance_pupil:
            # Sample 2nd points on the pupil
            pupilz, pupilx = self.entrance_pupil()

            x2 = torch.linspace(- pupilx, pupilx, M) * 0.99
            y2 = torch.zeros_like(x2)
            z2 = torch.full_like(x2, pupilz)
            o2 = torch.stack((x2,y2,z2), axis=-1)   # shape [M, 3]
            
            dx = torch.full_like(x2, np.sin(view / 57.3))
            dy = torch.zeros_like(x2)
            dz = torch.full_like(x2, np.cos(view / 57.3))
            d = torch.stack((dx,dy,dz), axis=-1)

            # Move ray origins to z = -0.1 for tracing
            if pupilz > 0:
                o = o2 - d * ((z2 + 0.1) / dz).unsqueeze(-1)
            else:
                o = o2

            return Ray(o, d, wvln, device=self.device)
        
        else:
            # Sample points on z = 0 or z = d_sensor
            x = torch.linspace(-R, R, M)
            y = torch.zeros_like(x)
            if z is None:
                z = 0 if forward else self.d_sensor
            z = torch.full_like(x, z)
            o = torch.stack((x, y, z), axis=-1)
            
            # Calculate ray directions
            if forward:
                dx = torch.full_like(x, np.sin(view / 57.3))
                dy = torch.zeros_like(x)
                dz = torch.full_like(x, np.cos(view / 57.3))
            else:
                dx = torch.full_like(x, np.sin(view / 57.3))
                dy = torch.zeros_like(x)
                dz = torch.full_like(x, -np.cos(view / 57.3))

            d = torch.stack((dx,dy,dz), axis=-1)

            return Ray(o, d, wvln, device=self.device)


    @torch.no_grad()
    def sample_parallel(self,fov=0.0, R=None, z=None, M=15,  wvln=DEFAULT_WAVE, sampling='grid', forward=True, entrance_pupil=False):
        """ Sample parallel rays from plane (-R:R, -R:R, z). Rays have shape [spp, M, M, 3]
        
            Used for (1) in-focus loss, (2) RMS spot radius calculation (but not implemented)

        Args:
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.
            fov (float, optional): incident angle (in degree). Defaults to 0.0.
            R (float, optional): sampling radius. Defaults to None.
            z (float, optional): sampling depth. Defaults to 0..
            M (int, optional): ray number. Defaults to 15.
            sampling (str, optional): sampling method. Defaults to 'grid'.
            forward (bool, optional): forward or backward rays. Defaults to True.
            entrance_pupil (bool, optional): whether to use entrance pupil. Defaults to False.

        Returns:
            ray (Ray object): Ray object. Shape [spp, M, M]
        """
        if z is None:
            z = self.surfaces[0].d
        fov = np.radians(np.asarray(fov))   # convert degree to radian
        
        # Sample ray origins
        if entrance_pupil:
            pupilz, pupilr = self.entrance_pupil()
            if sampling == 'grid': # sample a square 
                x, y = torch.meshgrid(
                    torch.linspace(-pupilr, pupilr, M),
                    torch.linspace(pupilr, -pupilr, M),
                    indexing='xy'
                )
            elif sampling == 'radial':
                r2 = torch.rand((M, M)) * pupilr**2
                theta = torch.rand((M, M)) * 2 * np.pi
                x = torch.sqrt(r2) * torch.cos(theta)
                y = torch.sqrt(r2) * torch.sin(theta)
            else:
                raise Exception('Sampling method not implemented!')

        else:
            if R is None:
                # We want to sample at a depth, so radius of the cone need to be computed.
                sag = self.surfaces[0].surface(self.surfaces[0].r, 0.0).item() # sag is a float
                R = np.tan(fov) * sag + self.surfaces[0].r 

            if sampling == 'grid': # sample a square 
                x, y = torch.meshgrid(
                    torch.linspace(-R, R, M),
                    torch.linspace(R, -R, M),
                    indexing='xy'
                )
            elif sampling == 'radial':
                r2 = torch.rand((M, M)) * R**2
                theta = torch.rand((M, M)) * 2 * np.pi
                x = torch.sqrt(r2) * torch.cos(theta)
                y = torch.sqrt(r2) * torch.sin(theta)
            else:
                raise Exception('Sampling method not implemented!')

        # Generate rays
        if isinstance(fov, float):
            o = torch.stack((x, y, torch.full_like(x, pupilz)), axis=2)
            d = torch.zeros_like(o)
            if forward:
                d[...,2] = torch.full_like(x, np.cos(fov))
                d[...,0] = torch.full_like(x, np.sin(fov))
            else:
                d[...,2] = torch.full_like(x, -np.cos(fov))
                d[...,0] = torch.full_like(x, -np.sin(fov))
        else:
            spp = len(fov)
            o = torch.stack((x, y, torch.full_like(x, pupilz)), axis=2).unsqueeze(0).repeat(spp, 1, 1, 1)
            d = torch.zeros_like(o)
            for i in range(spp):
                if forward:
                    d[i, :, :, 2] = torch.full_like(x, np.cos(fov[i]))
                    d[i, :, :, 0] = torch.full_like(x, np.sin(fov[i]))
                else:
                    d[i, :, :, 2] = torch.full_like(x, -np.cos(fov[i]))
                    d[i, :, :, 0] = torch.full_like(x, -np.sin(fov[i]))


        rays = Ray(o, d, wvln, device=self.device)
        rays.propagate_to(z)
        return rays


    @torch.no_grad()
    def sample_point_source_2D(self, depth=-1000, view=0, M=9, entrance_pupil=False, wvln=DEFAULT_WAVE):
        """ Sample point source 2D rays. Rays hape shape of [M, 3].

            Used for (1) drawing lens setup, (2) paraxial optics calculation, for example, refocusing to given depth

        Args:
            depth (float, optional): sampling depth. Defaults to -1000.
            view (float, optional): incident angle (in degree). Defaults to 0.
            M (int, optional): ray number. Defaults to 9.
            entrance_pupil (bool, optional): whether to use entrance pupil. Defaults to False.
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.
        """
        if entrance_pupil:
            pupilz, pupilx = self.entrance_pupil()
        else:
            pupilz, pupilx = 0, self.surfaces[0].r

        # Second point on the pupil or first surface
        x2 = torch.linspace(-pupilx, pupilx, M, device=self.device) * 0.99
        y2 = torch.zeros_like(x2)
        z2 = torch.full_like(x2, pupilz)
        o2 = torch.stack((x2,y2,z2), axis=1)

        # First point is the point source
        o1 = torch.zeros_like(o2)
        o1[:, 2] = depth
        o1[:, 0] = depth * np.tan(view / 57.3)

        # Form the rays and propagate to z = 0
        d = o2 - o1
        ray = Ray(o1, d, wvln=wvln, device=self.device)
        ray.propagate_to(z=self.surfaces[0].d - 0.1)    # ray starts from z = - 0.1

        return ray


    @torch.no_grad()
    def sample_point_source(self, R=None, depth=-10.0, M=11, spp=16, fov=10.0, forward=True, pupil=True, wvln=DEFAULT_WAVE, importance_sampling=False):
        """ Sample forward point-grid rays. Rays have shape [spp, M, M, 3]
        
            Rays come from a 2D square array (-R~R, -Rw~Rw, depth), and fall into a cone spercified by fov or pupil. 
            
            Equivalent to self.point_source_grid() + self.sample_from_points()
            
            Used for (1) spot/rms/magnification calculation, (2) distortion/sensor sampling

        Args:
            R (float, optional): sample plane half side length. Defaults to None.
            depth (float, optional): sample plane z position. Defaults to -10.0.
            spp (int, optional): sample per pixel. Defaults to 16.
            fov (float, optional): cone angle. Defaults to 10.0.
            M (int, optional): sample plane resolution. Defaults to 11.
            forward (bool, optional): forward or backward rays. Defaults to True.
            pupil (bool, optional): whether to use pupil. Defaults to False.
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.
        """
        if R is None:
            R = self.surfaces[0].r
        Rw = R * self.sensor_res[1] / self.sensor_res[0] # half height

        # sample o
        x, y = torch.meshgrid(
            torch.linspace(-1, 1, M),
            torch.linspace(1, -1, M),
            indexing='xy'
            )

        if importance_sampling:
            x = torch.sqrt(x.abs()) * x.sign()
            y = torch.sqrt(y.abs()) * y.sign()

        x = x * Rw
        y = y * R

        # x, y = torch.t(x), torch.t(y)
        z = torch.full_like(x, depth)
        o = torch.stack((x,y,z), -1).to(self.device)
        o = o.unsqueeze(0).repeat(spp, 1, 1, 1)
        
        # sample d
        if pupil:
            o2 = self.sample_pupil(res=(M,M), spp=spp)
            d = o2 - o
            d = d / torch.linalg.vector_norm(d, ord=2, dim=-1, keepdim=True)

        else:
            raise Exception('Cone sampling specified by fov has been abandoned. Use pupil sampling instead.')

        # generate ray
        ray = Ray(o, d, wvln, device=self.device)
        return ray


    @torch.no_grad()
    def sample_from_points(self, o=[[0, 0, -10000]], spp=256, wvln=DEFAULT_WAVE, shrink_pupil=False, normalized=False):
        """ Sample forward rays from given point source (un-normalized positions). Rays have shape [spp, N, 3]

            Used for (1) PSF calculation, (2) chief ray calculation.

        Args:
            o (list): ray origin. Defaults to [[0, 0, -10000]].
            spp (int): sample per pixel. Defaults to 8.
            forward (bool): forward or backward rays. Defaults to True.
            pupil (bool): whether to use pupil. Defaults to True.
            fov (float): cone angle. Defaults to 10.
            wvln (float): ray wvln. Defaults to DEFAULT_WAVE.

        Returns:
            ray: Ray object. Shape [spp, N, 3]
        """
        # Compute o, shape [spp, N, 3]
        if not torch.is_tensor(o):
            o = torch.tensor(o)
        o = o.unsqueeze(0).repeat(spp, 1, 1)
        
        # Sample pupil and compute d
        pupilz, pupilr = self.entrance_pupil(shrink_pupil=shrink_pupil)
        theta = torch.rand(spp) * 2 * np.pi
        r = torch.sqrt(torch.rand(spp)*pupilr**2)
        x2 = r * torch.cos(theta)
        y2 = r * torch.sin(theta)
        z2 = torch.full_like(x2, pupilz)
        o2 = torch.stack((x2,y2,z2), 1)
            
        d = o2.unsqueeze(1) - o
        
        # Calculate rays
        ray = Ray(o, d, wvln=wvln, device=self.device)
        return ray

    @torch.no_grad()
    def sample_sensor(self, spp=64, pupil=True, wvln=DEFAULT_WAVE, sub_pixel=False):
        """ Sample rays from sensor pixels. Rays have shape of [spp, H, W, 3].

        Args:
            sensor_scale (int, optional): number of pixels remain the same, but only sample rays on part of sensor. Defaults to 1.
            spp (int, optional): sample per pixel. Defaults to 1.
            vpp (int, optional): sample per pixel on pupil. Defaults to 64.
            high_spp (bool, optional): whether to use high spp. Defaults to False.
            pupil (bool, optional): whether to use pupil. Defaults to True.
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.
        """
        # ===> sample o1 on sensor plane
        # In 'render_compute_img' func, we use top-left point as reference in rendering, so here we should sample bottom-right point
        x1, y1 = torch.meshgrid(
            torch.linspace(-self.sensor_size[1]/2, self.sensor_size[1]/2, self.sensor_res[1]+1, device=self.device)[1:],
            torch.linspace(self.sensor_size[0]/2, -self.sensor_size[0]/2, self.sensor_res[0]+1, device=self.device)[1:],
            indexing='xy'
        )
        z1 = torch.full_like(x1, self.d_sensor, device= self.device)

        # ==> Sample o2 on the second plane and compute rays
        if pupil is True:
            pupilz, pupilr = self.exit_pupil()
        else:
            raise Exception("This feature has been abandoned.")
            pupilz, pupilr = self.surfaces[-1].d.item(), self.surfaces[-1].r

        if sub_pixel:
            # For more realistic rendering, we can sample multiple points inside the pixel
            raise Warning("This feature is not finished yet.")

        else:
            # Use bottom-right corner to represent each pixel
            # sample o2, method 2, o2 shape [spp, res, res, 3]
            o2 = self.sample_pupil(self.sensor_res, spp, pupilr=pupilr, pupilz=pupilz)

            o = torch.stack((x1, y1, z1), 2)
            o = torch.broadcast_to(o, o2.shape)
            d = o2 - o    # broadcast to [spp, H, W, 3]
            
        ray = Ray(o, d, wvln=wvln, device=self.device)
        return ray


    @torch.no_grad()
    def sample_pupil(self, res=(512,512), spp=16, num_angle=8, pupilr=None, pupilz=None):
        """ Sample points (not rays) on the pupil plane with rings. Points have shape [spp, res, res].

            2*pi is devided into [num_angle] sectors.
            Circle is devided into [spp//num_angle] rings.

        Args:
            res (tuple): pupil plane resolution. Defaults to (512,512).
            spp (int): sample per pixel. Defaults to 16.
            num_angle (int): number of sectors. Defaults to 8.
            pupilr (float): pupil radius. Defaults to None.
            pupilz (float): pupil z position. Defaults to None.
            multiplexing (bool): whether to use multiplexing. Defaults to False.
        """
        H, W = res
        if pupilr is None or pupilz is None:
            pupilz, pupilr = self.entrance_pupil()

        # => Naive implementation
        if spp % num_angle != 0 or spp >= 10000:
            theta = torch.rand((spp, H, W), device=self.device) * 2 * np.pi
            r2 = torch.rand((spp, H, W), device=self.device) * pupilr**2
            r = torch.sqrt(r2)

            x = r * torch.cos(theta)
            y = r * torch.sin(theta)
            z = torch.full_like(x, pupilz)
            o = torch.stack((x,y,z), -1)

        # => Sample more uniformly when spp is not large
        else:
            num_r2 = spp // num_angle
            
            # ==> For each pixel, sample different points on the pupil
            x, y = [], []
            for i in range(num_angle):
                for j in range(spp//num_angle):
                    delta_theta = torch.rand((1, *res), device=self.device) * 2 * np.pi / num_angle # sample delta_theta from [0, pi/4)
                    theta = delta_theta + i * 2 * np.pi / num_angle 

                    delta_r2 = torch.rand((1, *res), device=self.device) * pupilr**2 / spp * num_angle
                    r2 = delta_r2 + j * pupilr**2 / spp * num_angle
                    r = torch.sqrt(r2)

                    x.append(r * torch.cos(theta))
                    y.append(r * torch.sin(theta))
            
            x = torch.cat(x, dim=0)
            y = torch.cat(y, dim=0)
            z = torch.full_like(x, pupilz)
            o = torch.stack((x,y,z), -1)

        return o



    # ====================================================================================
    # Ray Tracing functions
    # ====================================================================================
    def trace(self, ray, lens_range=None, record=False):
        """ General ray tracing function. Ray in and ray out.

            Transform between local and world coordinates and do ray tracing under local coordinates. 

            Forward or backward ray tracing is automatically determined by ray directions.

        Args:
            ray ([type]): [description]
            stop_ind ([int]): Early stop index.
            record: Only when we want to plot ray path, set `record` to True.

        Returns:
            ray_final (Ray object): ray after optical system.
            valid (boolean matrix): mask denoting valid rays.
            oss (): position of ray on the sensor plane.
        """
        is_forward = (ray.d.reshape(-1,3)[0,2] > 0)
        if lens_range is None:
            lens_range = range(0, len(self.surfaces))
        
        if is_forward:
            valid, ray_out, oss = self._forward_tracing(ray, lens_range, record=record)
        else:
            valid, ray_out, oss = self._backward_tracing(ray, lens_range, record=record)

        return ray_out, valid, oss


    def trace2obj(self, ray, depth=DEPTH):
        """ Trace rays through the lens and reach the sensor plane.
        """
        ray, _, _, = self.trace(ray)
        ray = ray.propagate_to(depth)
        return ray

    
    def trace2sensor(self, ray, record=False, ignore_invalid=False):
        """ Trace optical rays to sensor plane.
        """
        if record:
            ray_out, valid, oss = self.trace(ray, record=record)
            ray_out = ray_out.propagate_to(self.d_sensor)
            valid = (ray_out.ra == 1)
            p = ray.o
            for os, v, pp in zip(oss, valid.cpu().detach().numpy(), p.cpu().detach().numpy()):
                if v.any():
                    os.append(pp)

            if ignore_invalid:
                p = p[valid]
            else:
                assert len(p.shape) >= 2, 'This function is not tested.'
                p = torch.reshape(p, (np.prod(p.shape[:-1]), 3))

            for v, os, pp in zip(valid, oss, p):
                if v:
                    os.append(pp.cpu().detach().numpy())
            return p, oss

        else:
            ray, _, _, = self.trace(ray)
            ray = ray.propagate_to(self.d_sensor)
            return ray

    def _forward_tracing(self, ray, lens_range, record):
        """ Trace rays from object space to sensor plane.
        """
        wvln = ray.wvln
        dim = ray.o[..., 2].shape # What does this mean: how many rays do we have? here 31*31

        if record:
            oss = []    # oss records all points of intersection. ray.o shape of [N, 3]
            for i in range(dim[0]):
                oss.append([ray.o[i,:].cpu().detach().numpy()])
        else:
            oss = None

        for i in lens_range:
            ray = self.surfaces[i].ray_reaction(ray)
            
            valid = (ray.ra == 1)
            if record: 
                p = ray.o
                for os, v, pp in zip(oss, valid.cpu().detach().numpy(), p.cpu().detach().numpy()):
                    if v.any():
                        os.append(pp)
        
        return valid, ray, oss


    def _backward_tracing(self, ray, lens_range, record):
        """ Trace rays from sensor plane to object space.
        """
        wvln = ray.wvln 
        dim = ray.o[..., 2].shape
        valid = (ray.ra == 1)
        
        if record:
            oss = []    # oss records all points of intersection
            for i in range(dim[0]):
                oss.append([ray.o[i,:].cpu().detach().numpy()])
        else:
            oss = None

        for i in np.flip(lens_range):
            ray = self.surfaces[i].ray_reaction(ray)

            valid = (ray.ra > 0)
            if record: 
                p = ray.o
                for os, v, pp in zip(oss, valid.cpu().detach().numpy(), p.cpu().detach().numpy()):
                    if v.any():
                        os.append(pp)

        valid = (ray.ra == 1)
        return valid, ray, oss


        
    # ====================================================================================
    # Ray-tracing based rendering
    # ====================================================================================
    @torch.no_grad()
    def render_single_img(self, img_org, depth=DEPTH, spp=64, unwarp=False, save_name=None, return_tensor=False, noise=0, method='raytracing'):
        """ Render a single image for visualization and debugging.

            This function is designed non-differentiable. If want to use differentiable rendering, call self.render() function.

        Args:
            img_org (ndarray): ndarray read by opencv.
            render_unwarp (bool, optional): _description_. Defaults to False.
            depth (float, optional): _description_. Defaults to DEPTH.
            save_name (string, optional): _description_. Defaults to None.

        Returns:
            ing_render (ndarray): rendered image. uint8 dtype and ndarray.
        """
        if not isinstance(img_org, np.ndarray):
            raise Exception('This function only supports ndarray input. If you want to render an image batch, use `render` function.')

        # ==> Prepare sensor to match the image resolution
        sensor_res = self.sensor_res
        if len(img_org.shape) == 2:
            rgb = False
            H, W = img_org.shape
            raise Exception('Monochrome image is not tested yet.')
        elif len(img_org.shape) == 3:
            rgb = True 
            H, W, C = img_org.shape
            assert C == 3, 'Only support RGB image, dtype should be ndarray.'
        self.prepare_sensor(sensor_res=[H, W])

        img = torch.tensor((img_org/255.).astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(self.device)

        if method == 'raytracing':
            # ==> Render object image by ray-tracing
            scale = self.calc_scale_ray(depth=depth)
            img = torch.flip(img, [-2, -1])
            if rgb:
                img_render = torch.zeros_like(img)
                # Normal rendering
                if spp <= 64:
                    for i in range(3):
                        ray = self.render_sample_ray(spp=spp, wvln=WAVE_RGB[i])
                        ray, _, _ = self.trace(ray) 
                        img_render[:,i,:,:] = self.render_compute_image(img[:,i,:,:], depth, scale, ray)
                # High-spp rendering
                else:
                    iter_num = int(spp // 64)
                    for ii in range(iter_num):
                        for i in range(3):
                            ray = self.render_sample_ray(spp=64, wvln=WAVE_RGB[i])
                            ray, _, _ = self.trace(ray) 
                            img_render[:,i,:,:] += self.render_compute_image(img[:,i,:,:], depth, scale, ray, train=False)
                    img_render /= iter_num
            else:
                ray = self.render_sample_ray(spp=spp, wvln=DEFAULT_WAVE)
                ray, _, _ = self.trace(ray)
                img_render = self.render_compute_image(img, depth, scale, ray)
        
        elif method == 'psf':
            psf_grid = 7
            psf_ks = 21
            psf_map = self.psf_map(grid=psf_grid, ks=psf_ks, depth=depth)
            img_render = render_psf_map(img, psf_map, grid=psf_grid)
        
        # ==> Unwarp to correct geometry distortion
        if unwarp:
            img_render = self.unwarp(img_render, depth)

        # ==> Add noise
        if noise > 0:
            img_render = img_render + torch.randn_like(img_render) * noise
            img_render = torch.clamp(img_render, 0, 1)
        

        if save_name is not None:
            save_image(img_render, f'{save_name}.png')

        # ==> Change the sensor resolution back
        self.prepare_sensor(sensor_res=sensor_res)

        if return_tensor:
            return img_render
        else:
            # ==> Convert to uint8
            img_render = img_render[0,...].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            return img_render

    # ====================================================================================
    # PSF and spot diagram
    #   1. Incoherent functions
    #   2. Coherent functions 
    # ====================================================================================
    def point_source_grid(self, depth, grid=9, normalized=True, quater=False, center=False):
        """ Compute point grid [-1: 1] * [-1: 1] in the object space to compute PSF grid.

        Args:
            depth (float): Depth of the point source plane.
            grid (int): Grid size. Defaults to 9.
            normalized (bool): Whether to use normalized x, y corrdinates [-1, 1]. Defaults to True.
            quater (bool): Whether to use quater of the grid. Defaults to False.
            center (bool): Whether to use center of each patch. Defaults to False.

        Returns:
            point_source: Shape of [grid, grid, 3].
        """
        if grid == 1:
            x, y = torch.tensor([[0.]]), torch.tensor([[0.]])
            assert not quater, 'Quater should be False when grid is 1.'
        else:
            # ==> Use center of each patch
            if center:
                half_bin_size = 1 / 2 / (grid - 1)
                x, y = torch.meshgrid(
                    torch.linspace(-1 + half_bin_size, 1 - half_bin_size, grid), 
                    torch.linspace(1 - half_bin_size, -1 + half_bin_size, grid),
                    indexing='xy')
            # ==> Use corner
            else:   
                x, y = torch.meshgrid(
                    torch.linspace(-0.98, 0.98, grid), 
                    torch.linspace(0.98, -0.98, grid),
                    indexing='xy')
        
        z = torch.full((grid, grid), depth)
        point_source = torch.stack([x, y, z], dim=-1)
        
        # ==> Use quater of the sensor plane to save memory
        if quater:
            z = torch.full((grid, grid), depth)
            point_source = torch.stack([x, y, z], dim=-1)
            bound_i = grid // 2 if grid % 2 == 0 else grid // 2 + 1
            bound_j = grid // 2
            point_source = point_source[0:bound_i, bound_j:, :]

        if not normalized:
            scale = self.calc_scale_pinhole(depth)
            point_source[..., 0] *= scale * self.sensor_size[0] / 2
            point_source[..., 1] *= scale * self.sensor_size[1] / 2

        return point_source

    
    def point_source_radial(self, depth, grid=9, center=False):
        """ Compute point radial [0, 1] in the object space to compute PSF grid.

        Args:
            grid (int, optional): Grid size. Defaults to 9.

        Returns:
            point_source: Shape of [grid, 3].
        """
        if grid == 1:
            x = torch.tensor([0.])
        else:
            # Select center of bin to calculate PSF
            if center:
                half_bin_size = 1 / 2 / (grid - 1)
                x = torch.linspace(0, 1 - half_bin_size, grid)
            else:   
                x = torch.linspace(0, 0.98, grid)
        
        z = torch.full_like(x, depth)
        point_source = torch.stack([x, x, z], dim=-1)
        return point_source

    
    @torch.no_grad()
    def psf_center(self, point, method='chief_ray'):
        """ Compute reference PSF center (flipped, green light) for given point source.

        Args:
            point: [N, 3] un-normalized point is in object plane.

        Returns:
            psf_center: [N, 2] un-normalized psf center in sensor plane.
        """
        if method == 'chief_ray':
            # Shrink the pupil and calculate centroid ray as the chief ray. Distortion is allowed.
            ray = self.sample_from_points(point, spp=GEO_SPP, shrink_pupil=True)
            ray = self.trace2sensor(ray)
            assert (ray.ra == 1).any(), 'No sampled rays is valid.'
            psf_center = (ray.o * ray.ra.unsqueeze(-1)).sum(0) / ray.ra.unsqueeze(-1).sum(0).add(EPSILON) # shape [N, 3]
            psf_center = - psf_center[..., :2]   # shape [N, 2]

        elif method == 'pinhole':
            # Pinhole camera perspective projection. This doesnot allow distortion.
            scale = self.calc_scale_pinhole(point[..., 2])
            psf_center = - point[..., :2] / scale
        
        else:
            raise Exception('Unsupported method.')

        return psf_center

    def psf(self, points, ks=31, wvln=DEFAULT_WAVE, spp=GEO_SPP, center=True):
        """ Single wvln incoherent PSF calculation. 
        
            Given point sources with shape [N, 3], compute [N, ks, ks] PSF kernels.

        Args:
            points (Tensor): Shape of [N, 3], point is in object space, normalized.
            kernel_size (int, optional): Output kernel size. Defaults to 7.
            wvln (float, optional): wvln. Defaults to DEFAULT_WAVE.
            center (bool, optional): Use spot center as PSF center.

        Returns:
            kernel: Shape of [N, ks, ks].
        """
        psf = self.psf_diff(points=points, wvln=wvln, ks=ks, spp=spp, center=center)
        return psf

    
    def psf_diff(self, points, wvln=DEFAULT_WAVE, ks=31, spp=GEO_SPP, center=True):
        """ Single wvln incoherent PSF calculation.

        Args:
            points (Tnesor): Normalized point source position. Shape of [N, 3], x, y in range [-1, 1], z in range [-Inf, 0].
            kernel_size (int, optional): Output kernel size. Defaults to 7.
            spp (int, optional): Sample per pixel. For diff ray tracing, usually kernel_size^2. Defaults to 2048.
            center (bool, optional): Use spot center as PSF center.

        Returns:
            kernel: Shape of [N, ks, ks] or [ks, ks].
        """
        # Points shape of [N, 3]
        if not torch.is_tensor(points):
            points = torch.tensor(points)
        if len(points.shape) == 1:
            single_point = True
            points = points.unsqueeze(0)
        else:
            single_point = False

        # Ray position in the object space by perspective projection, because points are normalized
        depth = points[:, 2]
        scale = self.calc_scale_pinhole(depth)
        point_obj = points.clone()
        point_obj[..., 0] = points[..., 0] * scale * self.sensor_size[1] / 2   # x coordinate
        point_obj[..., 1] = points[..., 1] * scale * self.sensor_size[0] / 2   # y coordinate
        
        # Trace rays to sensor plane
        ray = self.sample_from_points(o=point_obj, spp=spp, wvln=wvln)
        ray = self.trace2sensor(ray)

        # Calculate PSF
        if center:
            # PSF center on the sensor plane by chief ray
            pointc_chief_ray = self.psf_center(point_obj)   # shape [N, 2]
            psf = forward_integral(ray, ps=self.pixel_size, ks=ks, pointc_ref=pointc_chief_ray)
        else:
            # PSF center on the sensor plane by pespective
            pointc_ideal = points.clone()[:,:2]
            pointc_ideal[:, 0] *= self.sensor_size[1] / 2
            pointc_ideal[:, 1] *= self.sensor_size[0] / 2
            psf = forward_integral(ray, ps=self.pixel_size, ks=ks, pointc_ref=pointc_ideal)
        
        # Normalize to 1
        psf = psf / psf.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1)
        
        if single_point:
            psf = psf.squeeze(0)

        return psf
    

    def psf_rgb(self, points, ks=31, spp=GEO_SPP, center=True):
        """ Compute RGB point PSF. This function is differentiable.
        
        Args:
            point (Tensor): Shape of [N, 3], point is in object space, normalized.
            ks (int, optional): Output kernel size. Defaults to 7.
            spp (int, optional): Sample per pixel. Defaults to 2048.
            center (bool, optional): Use spot center as PSF center.

        Returns:
            psf: Shape of [N, 3, ks, ks] or [3, ks, ks].
        """
        psfs = []
        for wvln in WAVE_RGB:
            psfs.append(self.psf_diff(points=points, wvln=wvln, ks=ks, spp=spp, center=center))
        
        psf = torch.stack(psfs, dim = -3)   # shape [3, ks, ks] or [N, 3, ks, ks]
        return psf
    

    def psf_map(self, depth=DEPTH, grid=7, ks=51, spp=GEO_SPP, center=True):
        """ Compute RGB PSF map at a given depth.

            Now used for (1) rendering, (2) draw PSF map

        Args:
            grid (int, optional): Grid size. Defaults to 7.
            ks (int, optional): Kernel size. Defaults to 51.
            depth (float, optional): Depth of the point source plane. Defaults to DEPTH.
            center (bool, optional): Use spot center as PSF center. Defaults to True.
            spp (int, optional): Sample per pixel. Defaults to None.

        Returns:
            psf_map: Shape of [3, grid*ks, grid*ks].
        """
        points = self.point_source_grid(depth=depth, grid=grid, quater=False)
        points = points.reshape(-1, 3)
        psfs = self.psf_rgb(points=points, ks=ks, center=center, spp=spp) # shape [grid**2, 3, ks, ks]

        psf_map = make_grid(psfs, nrow=grid, padding=0, pad_value=0.0)
        return psf_map

    def psf2mtf(self, psf, diag=False):
        """ Convert 2D PSF kernel to MTF curve by FFT.

        Args:
            psf (tensor): 2D PSF tensor.

        Returns:
            freq (ndarray): Frequency axis.
            tangential_mtf (ndarray): Tangential MTF.
            sagittal_mtf (ndarray): Sagittal MTF.
        """
        psf = psf.cpu().numpy()
        x = np.linspace(-1, 1, psf.shape[1]) * self.pixel_size * psf.shape[1] / 2
        y = np.linspace(-1, 1, psf.shape[0]) * self.pixel_size * psf.shape[0] / 2

        # Extract 1D PSFs along the sagittal and tangential directions
        center_x = psf.shape[1] // 2
        center_y = psf.shape[0] // 2
        sagittal_psf = psf[center_y, :]
        tangential_psf = psf[:, center_x]

        # Fourier Transform to get the MTFs
        sagittal_mtf = np.abs(np.fft.fft(sagittal_psf))
        tangential_mtf = np.abs(np.fft.fft(tangential_psf))

        # Normalize the MTFs
        sagittal_mtf /= sagittal_mtf.max()
        tangential_mtf /= tangential_mtf.max()

        delta_x = self.pixel_size #/ 2

        # Create frequency axis in cycles/mm
        freq = np.fft.fftfreq(psf.shape[0], delta_x)

        # Only keep the positive frequencies
        positive_freq_idx = freq > 0

        return freq[positive_freq_idx], tangential_mtf[positive_freq_idx], sagittal_mtf[positive_freq_idx]



    # ====================================================================================
    # Geometrical optics 
    #   1. Focus-related functions
    #   2. FoV-related functions
    #   3. Pupil-related functions
    # ====================================================================================

    # ---------------------------
    # 1. Focus-related functions
    # ---------------------------
    def calc_foclen(self):
        """ Calculate the focus length.
        """
        # Cellphone lens, we usually use EFL to describe the lens.
        if self.r_last < 8:
            return self.calc_efl()
        
        # Camera lens, we use the to describe the lens.
        else:
            return self.calc_bfl()

    def calc_bfl(self, wvln=DEFAULT_WAVE):
        """ Compute back focal length (BFL). 

            BFL: Distance from the second principal point to sensor plane.
        """
        return self.d_sensor - self.calc_principal(wvln=wvln)[1]

    def calc_efl(self):
        """ Compute effective focal length (EFL). Effctive focal length is also commonly used to compute F/#.

            EFL: Defined by FoV and sensor radius.
        """
        return self.r_last / np.tan(self.hfov)

    def calc_eqfl(self):
        """ 35mm equivalent focal length. For cellphone lens, we usually use EFL to describe the lens.

            35mm sensor: 36mm * 24mm
        """
        return 21.63 / np.tan(self.hfov)

    @torch.no_grad()
    def calc_foc_dist(self, wvln=DEFAULT_WAVE):
        """ Compute the focus distance (object space) of the lens.

            Rays start from sensor and trace to the object space, the focus distance is negative.
        """
        # => Sample point source rays from sensor center
        o1 = torch.tensor([0, 0, self.d_sensor], device=self.device).repeat(GEO_SPP, 1)
        o2 = self.surfaces[0].surface_sample(GEO_SPP)   # A simple method is to sample from the first surface.
        o2 *= 0.2   # Shrink sample region
        d = o2 - o1
        ray = Ray(o1, d, wvln=wvln)

        # => Trace rays to the object space and compute focus distance
        ray, _, _ = self.trace(ray)
        t = (ray.d[...,0]*ray.o[...,0] + ray.d[...,1]*ray.o[...,1]) / (ray.d[...,0]**2 + ray.d[...,1]**2) # The solution for the nearest distance.
        focus_p = (ray.o[...,2] - ray.d[...,2] * t)[ray.ra > 0].cpu().numpy()
        focus_p = focus_p[~np.isnan(focus_p) & (focus_p < 0)]
        focus_dist = np.mean(focus_p)

        return focus_dist

    @torch.no_grad()
    def refocus_inf(self):
        """ Shift sensor to get the best center focusing.
        """
        # Trace rays and compute in-focus sensor position
        ray = self.sample_parallel_2D(R=self.surfaces[0].r * 0.5, M=GEO_SPP, wvln=DEFAULT_WAVE)
        ray, _, _ = self.trace(ray)
        t = (ray.d[...,0]*ray.o[...,0] + ray.d[...,1]*ray.o[...,1]) / (ray.d[...,0]**2 + ray.d[...,1]**2)
        focus_p = (ray.o[...,2] - ray.d[...,2] * t).cpu().numpy()
        focus_p = focus_p[ray.ra.cpu() > 0]
        focus_p = focus_p[~np.isnan(focus_p) & (focus_p>0)]
        d_sensor_new = float(np.mean(focus_p))
        
        # Update sensor position
        assert d_sensor_new > 0, 'sensor position is negative.'
        self.d_sensor = d_sensor_new

        # FoV will be slightly changed
        self.post_computation()


    @torch.no_grad()
    def refocus(self, depth=DEPTH):
        """ Refocus the lens to a depth distance by changing sensor position.

            In DSLR, phase detection autofocus (PDAF) is a popular and efficient method. But here we simplify the problem by calculating the in-focus position of green light.
        """
        # Trace green light
        o = self.surfaces[0].surface_sample(GEO_SPP)
        #o *= 0.2  # shrink sample region
        d = o - torch.tensor([0, 0, depth], dtype=torch.float32).to(self.device)
        ray = Ray(o, d, wvln=DEFAULT_WAVE, device=self.device)
        ray, _, _ = self.trace(ray)

        # Calculate in-focus sensor position of green light (use least-squares solution)
        t = (ray.d[...,0]*ray.o[...,0] + ray.d[...,1]*ray.o[...,1]) / (ray.d[...,0]**2 + ray.d[...,1]**2)
        t = t * ray.ra
        focus_d = (ray.o[...,2] - ray.d[...,2] * t).cpu().numpy()
        focus_d = focus_d[ray.ra.cpu() > 0]
        focus_d = focus_d[~np.isnan(focus_d) & (focus_d>0)]
        d_sensor_new = float(np.mean(focus_d))
        
        # Update sensor position
        assert d_sensor_new > 0, 'sensor position is negative.'
        self.d_sensor = d_sensor_new # d_sensor should be a float value, not a tensor

        # FoV will be slightly changed
        self.post_computation()


    # ---------------------------
    # 2. FoV-related functions
    # ---------------------------
    @torch.no_grad()
    def calc_fov(self):
        """ Compute half diagonal fov.

            Shot rays from edge of sensor, trace them to the object space and compute
            angel, output rays should be parallel and the angle is half of fov.
        """
        # Sample rays going out from edge of sensor, shape [M, 3] 
        M = 100
        pupilz, pupilx = self.exit_pupil(shrink_pupil=True)
        o1 = torch.zeros([M, 3])
        o1 = torch.tensor([self.r_last, 0, self.d_sensor]).repeat(M, 1).to(torch.float32)

        x2 = torch.linspace(- pupilx, pupilx, M)
        y2 = torch.full_like(x2, 0)
        z2 = torch.full_like(x2, pupilz)
        o2 = torch.stack((x2, y2, z2), axis=-1)

        ray = Ray(o1, o2 - o1, device=self.device)
        ray, _, _ = self.trace(ray)

        # compute fov
        tan_fov = ray.d[...,0] / ray.d[...,2]
        fov = torch.atan(torch.sum(tan_fov * ray.ra) / torch.sum(ray.ra))

        if torch.isnan(fov):
            print('computed fov is NaN, use 0.5 rad instead.')
            fov = 0.5
        else:
            fov = fov.item()
        
        return fov


    @torch.no_grad()
    def calc_magnification3(self, depth):
        """ Use mapping relationship (ray tracing) to compute magnification. The computed magnification is very accurate.

            Advatages: can use many data points to reduce error.
            Disadvantages: due to distortion, some data points contain error
        """
        M = 21
        spp = 512
        # sample rays [spp, W, H]
        # sample on a far object plane, but shrink to avoid distortion at the edge
        ray = self.sample_point_source(M=M, spp=spp, depth=depth, R=-depth*np.tan(self.hfov)*0.5, pupil=True)
        
        # map r1 from object space to sensor space, ground-truth
        o1 = ray.o.detach()[..., :2]
        o1 = torch.flip(o1, [1, 2])
        
        ray, _, _ = self.trace(ray)
        o2 = ray.project_to(self.d_sensor)

        # use 1/4 part of regions to compute magnification, also to avoid zero values on the axis
        x1 = o1[0,:,:,0]
        y1 = o1[0,:,:,1]
        x2 = torch.sum(o2[...,0] * ray.ra, axis=0)/ torch.sum(ray.ra, axis=0).add(EPSILON)
        y2 = torch.sum(o2[...,1] * ray.ra, axis=0)/ torch.sum(ray.ra, axis=0).add(EPSILON)

        mag_x = x1 / x2
        # mag = 1 / torch.mean(mag_x[:M//2,:M//2]).item()
        tmp = mag_x[:M//2,:M//2]
        mag = 1 / torch.mean(tmp[~tmp.isnan()]).item()

        if mag == 0:
            scale = - depth * np.tan(self.hfov) / self.r_last
            return 1 / scale

        return mag


    @torch.no_grad()
    def calc_principal(self, wvln=DEFAULT_WAVE):
        """ Compute principal (front and back) planes.
        """
        M = 32

        # Backward ray tracing for the first principal point
        ray = self.sample_parallel_2D(R=self.surfaces[0].r, M=M, forward=False, wvln=wvln)
        inc_ray = ray.clone()
        out_ray, _, _ = self.trace(ray)

        t = (out_ray.o[..., 0] - inc_ray.o[..., 0]) / out_ray.d[..., 0]
        z = out_ray.o[...,2] - out_ray.d[...,2] * t
        front_principal = np.nanmean(z[ray.ra > 0].cpu().numpy())

        # Forward ray tracing fir the second principal point
        ray = self.sample_parallel_2D(R=self.surfaces[0].r, M=M, forward=True, wvln=wvln)
        inc_ray = ray.clone()
        out_ray, _, _ = self.trace(ray)

        t = (out_ray.o[..., 0] - inc_ray.o[..., 0]) / out_ray.d[..., 0]
        z = out_ray.o[..., 2] - out_ray.d[..., 2] * t
        back_principal = np.nanmean(z[ray.ra > 0].cpu().numpy())

        return front_principal, back_principal


    @torch.no_grad()
    def calc_scale_pinhole(self, depth):
        """ Assume the first principle point is at (0, 0, 0), use pinhole camera to calculate the scale factor.
        """
        scale = - depth * np.tan(self.hfov) / self.r_last
        return scale
    

    @torch.no_grad()
    def calc_scale_ray(self, depth):
        """ Use ray tracing to compute scale factor.
        """
        if isinstance(depth, torch.Tensor) and len(depth.shape) == 1:
            scale = []
            for d in depth:
                scale.append(1 / self.calc_magnification3(d))     
            scale = torch.tensor(scale)
        else:
            scale = 1 / self.calc_magnification3(depth)

        return scale


    # ---------------------------
    # 3. Pupil-related functions
    # ---------------------------
    @torch.no_grad()
    def exit_pupil(self, shrink_pupil=False):
        """ Sample **forward** rays to compute z coordinate and radius of exit pupil. 
            Exit pupil: ray comes from sensor to object space. 
        """
        return self.entrance_pupil(entrance=False, shrink_pupil=shrink_pupil)


    @torch.no_grad()
    def entrance_pupil(self, M=32, entrance=True, shrink_pupil=False):
        """ We sample **backward** rays, return z coordinate and radius of entrance pupil. 
            Entrance pupil: how many rays can come from object space to sensor. 

            When we only consider rays from very far points.

            M should not be too large.
        """
        if self.aper_idx is None:
            if entrance:
                return self.surfaces[0].d.item(), self.surfaces[0].r
            else:
                return self.surfaces[-1].d.item(), self.surfaces[-1].r

        # sample M forward rays from edge of aperture to last surface.
        aper_idx = self.aper_idx
        aper_z = self.surfaces[aper_idx].d.item()
        aper_r = self.surfaces[aper_idx].r
        ray_o = torch.tensor([[aper_r, 0, aper_z]]).repeat(M, 1).to(torch.float32)

        # phi ranges from [-0.5rad, 0.5rad]
        phi = torch.arange(-0.5, 0.5, 1.0/M)
        # phi, _ = torch.sort(torch.rand(M)-0.5, 0)
        if entrance:
            d = torch.stack((
                torch.sin(phi),
                torch.zeros_like(phi),
                -torch.cos(phi)
            ), axis=-1)
        else:
            d = torch.stack((
                torch.sin(phi),
                torch.zeros_like(phi),
                torch.cos(phi)
            ), axis=-1)

        ray = Ray(ray_o, d, device=self.device)

        # ray tracing
        if entrance:
            lens_range = range(0, self.aper_idx)
            ray,_,_ = self.trace(ray, lens_range=lens_range)
        else:
            lens_range = range(self.aper_idx+1, len(self.surfaces))
            ray,_,_ = self.trace(ray, lens_range=lens_range)

        # This step is very slow!
        # compute intersection. o1+d1*t1 = o2+d2*t2
        pupilx = []
        pupilz = []
        for i in range(M):
            for j in range(i+1, M):
                if ray.ra[i] !=0 and ray.ra[j]!=0:
                    d1x, d1z, d2x, d2z = ray.d[i,0], ray.d[i,2], ray.d[j,0], ray.d[j,2]
                    o1x, o1z, o2x, o2z = ray.o[i,0], ray.o[i,2], ray.o[j,0], ray.o[j,2]
                    
                    # # Method 1: solve by torch.linalg.solve
                    # A = torch.tensor([[d1x, -d1z],[d2x, -d2z]])
                    # B = torch.tensor([[-d1z*o1x+d1x*o1z],[-d2z*o2x+d2x*o2z]])
                    # oz, ox = torch.linalg.solve(A,B)
                    
                    # Method 2: manually solve
                    Adet = - d1x * d2z + d2x * d1z
                    B1 = -d1z*o1x+d1x*o1z
                    B2 = -d2z*o2x+d2x*o2z
                    oz = (- B1 * d2z + B2 * d1z) / Adet
                    ox = (B2 * d1x - B1 * d2x) / Adet
                    
                    pupilx.append(ox.item())
                    pupilz.append(oz.item())
        
        if len(pupilx) == 0:
            avg_pupilx = aper_r
            avg_pupilz = 0
        else:
            avg_pupilx = stats.trim_mean(pupilx, 0.1)
            avg_pupilz = stats.trim_mean(pupilz, 0.1)
            if np.abs(avg_pupilz) < EPSILON:
                avg_pupilz = 0

        if shrink_pupil:
            avg_pupilx *= 0.5
        
        return avg_pupilz, avg_pupilx
    

    # ====================================================================================
    # Lens operation 
    #   1. Set lens parameters
    #   2. Lens operation (init, reverse, spherize), will be abandoned
    #   3. Lens pruning
    # ====================================================================================

    # ---------------------------
    # 1. Set lens parameters
    # ---------------------------
    def set_aperture(self, fnum=None, foclen=None, aper_r=None):
        """ Change aperture radius.
        """
        if aper_r is None:
            if foclen is None:
                foclen = self.calc_efl()
            aper_r = foclen / fnum / 2
            self.surfaces[self.aper_idx].r = aper_r
        else:
            self.surfaces[self.aper_idx].r = aper_r
        
        self.fnum = self.foclen / aper_r / 2


    # ---------------------------
    # 2. Lens operation
    # ---------------------------
    def pertub(self):
        """ Randomly perturb all surface parameters to simulate manufacturing errors. This function should only be called in the final image simulation stage. 
        """
        for i in range(len(self.surfaces)):
            self.surfaces[i].perturb()


    # ---------------------------
    # 3. Lens pruning
    # ---------------------------
    @torch.no_grad()
    def prune_surf(self, outer=None):
        """ Prune surfaces to the minimum height that allows all valid rays to go through.

        Args:
            outer (float): extra height to reserve. 
                For cellphone lens, we usually use 0.1mm or 0.05 * r_last. 
                For camera lens, we usually use 0.5mm or 0.1 * r_last.
        """
        outer = self.r_last * 0.05 if outer is None else outer
        self.pruning_v2(outer=outer)


    @torch.no_grad()
    def pruning_v2(self, outer=None, surface_range=None):
        """ Prune surfaces to the minimum height that allows all valid rays to go through.

        Args:
            surface_range ([type], optional): [description]. Defaults to None.
        """
        if outer is None:
            outer = self.r_last * 0.05
        
        if surface_range is None:
            surface_range = self.find_diff_surf()

        # ==> 1. Reset lens to maximum height(sensor radius)
        for i in surface_range:
            self.surfaces[i].r = self.r_last

        # ==> 2. Prune to reserve valid surface height
        # sample maximum fov rays to compute valid surface height
        view = self.hfov if self.hfov is not None else np.arctan(self.r_last/self.d_sensor)
        ray = self.sample_parallel_2D(view=np.rad2deg(view), M=21, entrance_pupil=True)

        ps, oss = self.trace2sensor(ray=ray, record=True)
        for i in surface_range:
            height = []
            for os in oss:  # iterate all rays
                try:
                    # because oss records the starting point at position 0, we need to ignore this.
                    height.append(np.abs(os[i+1][0]))   # the second index 0 means x coordinate
                except:
                    continue

            try:
                self.surfaces[i].r = max(height) + outer
            except:
                continue
        
        # ==> 3. Front surface should be smaller than back surface. This does not apply to fisheye lens.
        for i in surface_range[:-1]:
            if self.materials[i].A < self.materials[i+1].A:
                self.surfaces[i].r = min(self.surfaces[i].r, self.surfaces[i+1].r)

        # ==> 4. Remove nan part, also the maximum height should not exceed sensor radius
        for i in surface_range:
            max_height = min(self.surfaces[i].max_height(), self.r_last)
            self.surfaces[i].r = min(self.surfaces[i].r, max_height)


    @torch.no_grad()
    def correct_shape(self):
        """ Correct wrong lens shape during the training.
        """
        aper_idx = self.aper_idx
        diff_surf_range = self.find_diff_surf()
        shape_changed = False

        # ==> Rule 1: Move the first surface to z = 0
        move_dist = self.surfaces[0].d.item()
        for surf in self.surfaces:
            surf.d -= move_dist
        self.d_sensor -= move_dist

        # ==> Rule 2: Move lens group to get a fixed aperture distance. Only for aperture at the first surface.
        if aper_idx == 0:
            d_aper = 0.1

            # If the first surface is concave, use the maximum negative sag. 
            aper_r = self.surfaces[aper_idx].r
            sag1 = - self.surfaces[aper_idx+1].surface(aper_r, 0).item()
            if sag1 > 0:
                d_aper += sag1

            # Update position of all surfaces.
            delta_aper = self.surfaces[1].d.item() - d_aper
            for i in diff_surf_range:
                self.surfaces[i].d -= delta_aper

        
        # ==> Rule 3: If two surfaces overlap (at center), seperate them by a small distance
        for i in diff_surf_range[:-1]:
            if self.surfaces[i].d > self.surfaces[i+1].d:
                self.surfaces[i+1].d += 0.2
                shape_changed = True

        # ==> Rule 4: Prune all surfaces
        self.prune_surf()

        if shape_changed:
            print('Surface shape corrected.')
        return shape_changed


    # ====================================================================================
    # Visualization.
    # ====================================================================================
    @torch.no_grad()
    def analysis(self, save_name='./test', render=False, multi_plot=False, plot_invalid=True, zmx_format=False, depth=DEPTH, render_unwarp=False, lens_title=None):
        """ Analyze the optical lens.
        """
        # Draw lens geometry and ray path
        self.plot_setup2D_with_trace(filename=save_name, multi_plot=multi_plot, entrance_pupil=True, plot_invalid=plot_invalid, zmx_format=zmx_format, lens_title=lens_title, depth=depth)

        # Draw spot diagram and PSF map
        self.draw_psf_map(save_name=save_name, ks=51)

        # Calculate RMS error
        rms_avg, rms_radius_on_axis, rms_radius_off_axis = self.analysis_rms()
        print(f'On-axis RMS radius: {round(rms_radius_on_axis.item()*1000,3)}um, Off-axis RMS radius: {round(rms_radius_off_axis.item()*1000,3)}um, Avg RMS spot size (radius): {round(rms_avg.item()*1000,3)}um.')

        # Render an image, compute PSNR and SSIM
        if render:
            img_org = cv.cvtColor(cv.imread(f'./datasets/resolution_chart1.png'), cv.COLOR_BGR2RGB)
            img_render = self.render_single_img(img_org, depth=depth, spp=128, unwarp=render_unwarp, save_name=f'{save_name}_render', noise=0.01)

            render_psnr = round(compare_psnr(img_org, img_render, data_range=255), 4)
            render_ssim = round(compare_ssim(img_org, img_render, channel_axis=2, data_range=255), 4)
            print(f'Rendered image: PSNR={render_psnr}, SSIM={render_ssim}')


    @torch.no_grad()      
    def plot_setup2D_with_trace(self, filename, views=[0], M=7, depth=None, entrance_pupil=True, zmx_format=False, plot_invalid=True, multi_plot=False, lens_title=None):
        """ Plot lens setup with rays.
        """
        # ==> Title
        if lens_title is None:
            if self.aper_idx is not None:
                lens_title = f'FoV{round(2*self.hfov*57.3, 1)}({int(self.calc_eqfl())}mm EFL)_F/{round(self.fnum,2)}_DIAG{round(self.r_last*2, 2)}mm_FocLen{round(self.foclen,2)}mm'
            else:
                lens_title = f'FoV{round(2*self.hfov*57.3, 1)}({int(self.calc_eqfl())}mm EFL)_DIAG{round(self.r_last*2, 2)}mm_FocLen{round(self.foclen,2)}mm'
        

        # ==> Plot RGB seperately
        if multi_plot:
            R = self.surfaces[0].r
            views = np.linspace(0, np.rad2deg(self.hfov)*0.99, num=7)
            colors_list = 'rgb'
            fig, axs = plt.subplots(1, 3, figsize=(24, 6))
            fig.suptitle(lens_title)

            for i, wvln in enumerate(WAVE_RGB):
                ax = axs[i]
                ax, fig = self.plot_setup2D(ax=ax, fig=fig, zmx_format=zmx_format)

                for view in views:
                    if depth is None:
                        ray = self.sample_parallel_2D(R, wvln, view=view, M=M, entrance_pupil=entrance_pupil)
                    else:
                        ray = self.sample_point_source_2D(depth=depth, view=view, M=M, entrance_pupil=entrance_pupil, wvln=wvln)
                    ps, oss = self.trace2sensor(ray=ray, record=True)
                    ax, fig = self.plot_raytraces(oss, ax=ax, fig=fig, color=colors_list[i], plot_invalid=plot_invalid, ra=ray.ra)
                    ax.axis('off')

            fig.savefig(f"{filename}.svg", bbox_inches='tight', format='svg', dpi=600)
            fig.savefig(f"{filename}.png", bbox_inches='tight', format='png', dpi=300)
            plt.close()
        

        # ==> Plot RGB in one figure
        else:
            R = self.surfaces[0].r
            colors_list = 'bgr'
            views = [0, np.rad2deg(self.hfov)*0.707, np.rad2deg(self.hfov)*0.99]
            aspect = self.sensor_res[1] / self.sensor_res[0]
            ax, fig = self.plot_setup2D(zmx_format=zmx_format)
            
            for i, view in enumerate(views):
                if depth is None:
                    ray = self.sample_parallel_2D(R, WAVE_RGB[2-i], view=view, M=M, entrance_pupil=entrance_pupil)
                else:
                    ray = self.sample_point_source_2D(depth=depth, view=view, M=M, entrance_pupil=entrance_pupil, wvln=WAVE_RGB[2-i])
                        
                ps, oss = self.trace2sensor(ray=ray, record=True)
                ax, fig = self.plot_raytraces(oss, ax=ax, fig=fig, color=colors_list[i], plot_invalid=plot_invalid, ra=ray.ra)

            ax.axis('off')
            ax.set_title(lens_title)
            fig.savefig(f"{filename}.png", bbox_inches='tight', format='png', dpi=600)
            plt.close()

    
    def plot_back_ray_trace(self, filename='debug_backward_rays', spp=5, vpp=5, pupil=True):
        ax, fig = self.plot_setup2D()

        ray = self.sample_sensor_2D(pupil=pupil, spp=spp, vpp=vpp)
        _, _, oss = self.trace(ray=ray, record=True)
        ax, fig = self.plot_raytraces(oss, ax=ax, fig=fig, color='b')

        ax.axis('off')
        fig.savefig(f"{filename}.png", bbox_inches='tight')


    def plot_raytraces(self, oss, ax=None, fig=None, color='b-', show=True, p=None, valid_p=None, plot_invalid=True, ra=None):
        """ Plot ray paths.
        """
        if ax is None and fig is None:
            ax, fig = self.plot_setup2D()
        else:
            show = False

        for i, os in enumerate(oss):
            o = torch.Tensor(np.array(os)).to(self.device)
            x = o[...,0]
            z = o[...,2]

            o = o.cpu().detach().numpy()
            z = o[...,2].flatten()
            x = o[...,0].flatten()

            if p is not None and valid_p is not None:
                if valid_p[i]:
                    x = np.append(x, p[i,0])
                    z = np.append(z, p[i,2])

            if plot_invalid:
                ax.plot(z, x, color, linewidth=0.8)
            elif ra[i]>0:
                ax.plot(z, x, color, linewidth=0.8)

        if show: 
            plt.show()
        else: 
            plt.close()

        return ax, fig


    def plot_setup2D(self, ax=None, fig=None, color='k', with_sensor=True, zmx_format=False, fix_bound=False):
        """ Draw experiment setup.

        """
        def plot(ax, z, x, color):
            p = torch.stack((x, torch.zeros_like(x, device=self.device), z), axis=-1)
            p = p.cpu().detach().numpy()
            ax.plot(p[...,2], p[...,0], color)

        def draw_aperture(ax, surface, color):
            N = 3
            d = surface.d
            R = surface.r
            APERTURE_WEDGE_LENGTH = 0.05 * R # [mm]
            APERTURE_WEDGE_HEIGHT = 0.15 * R # [mm]

            # wedge length
            z = torch.linspace(d.item() - APERTURE_WEDGE_LENGTH, d.item() + APERTURE_WEDGE_LENGTH, N, device=self.device)
            x = -R * torch.ones(N, device=self.device)
            plot(ax, z, x, color)
            x = R * torch.ones(N, device=self.device)
            plot(ax, z, x, color)
            
            # wedge height
            z = d * torch.ones(N, device=self.device)
            x = torch.linspace(R, R+APERTURE_WEDGE_HEIGHT, N, device=self.device)
            plot(ax, z, x, color)
            x = torch.linspace(-R-APERTURE_WEDGE_HEIGHT, -R, N, device=self.device)
            plot(ax, z, x, color)        

        # If no ax is given, generate a new one.
        if ax is None and fig is None:
            fig, ax = plt.subplots(figsize=(5,5))
        else:
            show=False

        if len(self.surfaces) == 1: # if there is only one surface, then it should be aperture
            draw_aperture(ax, self.surfaces[0], color='orange')
        else:
            # ==> Draw surface
            for i, s in enumerate(self.surfaces):
                # Draw aperture
                if self.materials[i].A < 1.0003 and self.materials[i+1].A < 1.0003: # both are AIR
                    draw_aperture(ax, s, color='orange')

                else:
                    # => Draw spherical/aspherical surface
                    r = torch.linspace(-s.r, s.r, s.APERTURE_SAMPLING, device=self.device) # aperture sampling
                    z = s.surface_with_offset(r, torch.zeros(len(r), device=self.device))   # graw surface
                    plot(ax, z, r, color)

            
            # ==> Draw boundary, connect two surfaces
            s_prev = []
            for i, s in enumerate(self.surfaces):
                if self.materials[i].A < 1.0003: # AIR
                    s_prev = s
                else:
                    r_prev = s_prev.r
                    r = s.r
                    sag_prev = s_prev.surface_with_offset(r_prev, 0.0)
                    sag      = s.surface_with_offset(r, 0.0)

                    if zmx_format:
                        z = torch.stack((sag_prev, sag_prev, sag))
                        x = torch.Tensor(np.array([[r_prev], [r], [r]])).to(self.device)
                    else:
                        z = torch.stack((sag_prev, sag))
                        x = torch.Tensor(np.array([[r_prev], [r]])).to(self.device)

                    plot(ax, z, x, color)
                    plot(ax, z,-x, color)
                    s_prev = s


            # Draw sensor
            if with_sensor:
                ax.plot([self.d_sensor, self.d_sensor], [-self.r_last, self.r_last], color)
        
        plt.xlabel('z [mm]')
        plt.ylabel('r [mm]')
        ax.set_aspect('equal', adjustable='datalim', anchor='C') 
        ax.minorticks_on() 
        ax.set_xlim(-0.5, 7.5) 
        ax.set_ylim(-4, 4)
        ax.autoscale()
        
        return ax, fig


    @torch.no_grad()
    def draw_psf_map(self, grid=7, depth=DEPTH, ks=51, log_scale=False, quater=False, save_name=None):
        """ Draw RGB PSF map at a certain depth. Will draw M x M PSFs, each of size ks x ks.
        """
        # Calculate PSF map
        psf_map = self.psf_map(depth=depth, grid=grid, ks=ks, spp=GEO_SPP, center=True)
        
        # Normalize for each field
        for i in range(0, psf_map.shape[-2], ks):
            for j in range(0, psf_map.shape[-1], ks):
                psf_map[:,i:i+ks,j:j+ks] /= psf_map[:,i:i+ks,j:j+ks].max()

        # Los scale the PSF for better visualization
        if log_scale:
            psf_map = torch.log(psf_map + 1e-3)   # 1e-3 is an empirical value

        # Save figure using matplotlib
        plt.figure(figsize=(10, 10))
        psf_map = psf_map.permute(1, 2, 0).cpu().numpy()
        plt.imshow(psf_map)

        H, W = psf_map.shape[:2]
        ruler_len = 100 # Ruler: 100um
        arrow_end = ruler_len / (self.pixel_size * 1e3)   # plot a scale ruler
        plt.annotate('', xy=(0, H - 10), xytext=(arrow_end, H - 10), arrowprops=dict(arrowstyle='<->', color='white'))
        plt.text(arrow_end + 10, H - 10, f'{ruler_len} um', color='white', fontsize=12, ha='left')
        
        plt.axis('off')
        plt.tight_layout(pad=0)  # Removes padding
        save_name = f'./psf{-depth}mm.png' if save_name is None else f'{save_name}_psf{-depth}mm.png'
        plt.savefig(save_name, dpi=300)
        plt.close()


    @torch.no_grad()
    def draw_psf_radial(self, M=3, depth=DEPTH, ks=51, log_scale=False, save_name='./psf_radial.png'):
        """ Draw radial PSF (45 deg). Will draw M PSFs, each of size ks x ks.  
        """
        x = torch.linspace(0, 1, M)
        y = torch.linspace(0, 1, M)
        z = torch.full_like(x, depth)
        points = torch.stack((x, y, z), dim=-1)
        
        psfs = []
        for i in range(M):
            # Scale PSF for a better visualization
            psf = self.psf_rgb(points=points[i], ks=ks, center=True, spp=4096)
            psf /= psf.max()

            if log_scale:
                psf = torch.log(psf + EPSILON)
                psf = (psf - psf.min()) / (psf.max() - psf.min())
            
            psfs.append(psf)

        psf_grid = make_grid(psfs, nrow=M, padding=1, pad_value=0.0)
        save_image(psf_grid, save_name, normalize=True)


    @torch.no_grad()
    def draw_spot_diagram(self, M=7, depth=DEPTH, wvln=DEFAULT_WAVE, save_name=None):
        """ Draw spot diagram of the lens. Shot rays from grid points in object space, trace to sensor and visualize.
        """
        # Sample and trace rays from grid points
        mag = self.calc_magnification3(depth)
        ray = self.sample_point_source(M=M, R=self.sensor_size[0]/2/mag, depth=depth, wvln=wvln, spp=1024, pupil=True)
        ray = self.trace2sensor(ray)
        o2 = - ray.o.clone().cpu().numpy()
        ra = ray.ra.clone().cpu().numpy()

        # Plot multiple spot diagrams in one figure
        fig, axs = plt.subplots(M, M, figsize=(30,30))
        for i in range(M):
            for j in range(M):
                ra_ = ra[:,i,j]
                x, y = o2[:,i,j,0], o2[:,i,j,1]
                x, y = x[ra_>0], y[ra_>0]
                xc, yc = x.sum()/ra_.sum(), y.sum()/ra_.sum()

                # scatter plot
                axs[i, j].scatter(x, y, 1, 'black')
                axs[i, j].scatter([xc], [yc], None, 'r', 'x')
                axs[i, j].set_aspect('equal', adjustable='datalim')
        
        if save_name is None:
            plt.savefig(f'./spot{-depth}mm.png', bbox_inches='tight', format='png', dpi=300)
        else:
            plt.savefig(f'{save_name}_spot{-depth}mm.png', bbox_inches='tight', format='png', dpi=300)

        plt.close()


    @torch.no_grad()
    def draw_spot_radial(self, M=3, depth=DEPTH, save_name=None):
        """ Draw radial spot diagram of the lens.

        Args:
            M (int, optional): field number. Defaults to 3.
            depth (float, optional): depth of the point source. Defaults to DEPTH.
            save_name (string, optional): filename to save. Defaults to None.
        """
        # Sample and trace rays
        mag = self.calc_magnification3(depth)
        ray = self.sample_point_source(M=M*2-1, R=self.sensor_size[0]/2/mag, depth=depth, spp=1024, pupil=True, wvln=589.3)
        ray, _, _ = self.trace(ray)
        ray.propagate_to(self.d_sensor)
        o2 = torch.flip(ray.o.clone(), [1, 2]).cpu().numpy()
        ra = torch.flip(ray.ra.clone(), [1, 2]).cpu().numpy()

        # Plot multiple spot diagrams in one figure
        fig, axs = plt.subplots(1, M, figsize=(M*12,10))
        for i in range(M):
            i_bias = i + M - 1

            # calculate center of mass
            ra_ = ra[:,i_bias,i_bias]
            x, y = o2[:,i_bias,i_bias,0], o2[:,i_bias,i_bias,1]
            x, y = x[ra_>0], y[ra_>0]
            xc, yc = x.sum()/ra_.sum(), y.sum()/ra_.sum()

            # scatter plot
            axs[i].scatter(x, y, 12, 'black')
            axs[i].scatter([xc], [yc], 400, 'r', 'x')
            
            # visualization
            axs[i].set_aspect('equal', adjustable='datalim')
            axs[i].tick_params(axis='both', which='major', labelsize=18)
            axs[i].spines['top'].set_linewidth(4)
            axs[i].spines['bottom'].set_linewidth(4)
            axs[i].spines['left'].set_linewidth(4)
            axs[i].spines['right'].set_linewidth(4)

        # Save figure
        if save_name is None:
            plt.savefig(f'./spot{-depth}mm_radial.svg', bbox_inches='tight', format='svg', dpi=1200)
        else:
            plt.savefig(f'{save_name}_spot{-depth}mm_radial.svg', bbox_inches='tight', format='svg', dpi=1200)

        plt.close()


    @torch.no_grad()
    def draw_mtf(self, relative_fov=[0.0, 0.7, 1.0], save_name='./mtf.png', wvlns=DEFAULT_WAVE, depth=DEPTH):
        """ Draw MTF curve of the lens. 
        """
        if save_name[-4:] != '.png':
            save_name += '.png'

        relative_fov = [relative_fov] if isinstance(relative_fov, float) else relative_fov
        wvlns = [wvlns] if isinstance(wvlns, float) else wvlns
        color_list = 'rgb'

        plt.figure(figsize=(6,6))
        for wvln_idx, wvln in enumerate(wvlns):
            for fov_idx, fov in enumerate(relative_fov):
                point = torch.Tensor([fov, fov, depth])
                psf = self.psf_diff(points=point, wvln=wvln, ks=256)
                freq, mtf_tan, mtf_sag = self.psf2mtf(psf)

                fov_deg = round(fov * self.hfov * 57.3, 1)
                plt.plot(freq, mtf_tan, color_list[fov_idx], label=f'{fov_deg}(deg)-Tangential')
                plt.plot(freq, mtf_sag, color_list[fov_idx], label=f'{fov_deg}(deg)-Sagittal', linestyle='--')

        plt.legend()
        plt.xlabel('Spatial Frequency [cycles/mm]')
        plt.ylabel('MTF')

        # Save figure
        plt.savefig(f'{save_name}', bbox_inches='tight', format='png', dpi=300)
        plt.close()


    def draw_distortion(self, depth=DEPTH, save_name=None):
        """ Draw distortion.
        """
        # Ray tracing to calculate distortion map
        M = 16
        scale = self.calc_scale_pinhole(depth)
        ray = self.sample_point_source(M=M, spp=GEO_SPP, depth=depth, R=self.sensor_size[0]/2*scale, pupil=True)
        o1 = ray.o.detach().cpu()
        x1 = o1[0,:,:,0] / scale 
        y1 = o1[0,:,:,1] / scale 

        ray, _, _ = self.trace(ray)
        o2 = ray.project_to(self.d_sensor)
        o2 = o2.clone().cpu()
        x2 = torch.sum(o2[...,0] * ray.ra.cpu(), axis=0)/ torch.sum(ray.ra.cpu(), axis=0)
        y2 = torch.sum(o2[...,1] * ray.ra.cpu(), axis=0)/ torch.sum(ray.ra.cpu(), axis=0)

        # Draw image
        fig, ax = plt.subplots()
        ax.set_title('Lens distortion')
        ax.scatter(x1, y1, s=2)
        ax.scatter(x2, y2, s=2)
        ax.legend(['ref', 'distortion'])
        ax.axis('scaled')

        if save_name is None:
            plt.savefig(f'./distortion{-depth}mm.png', bbox_inches='tight', format='png', dpi=300)
        else:
            plt.savefig(f'{save_name}_distortion{-depth}mm.png', bbox_inches='tight', format='png', dpi=300)


    def analysis_rms(self, depth=DEPTH, ref=True):
        """ Compute RMS-based error. Contain both RMS errors and RMS radius.

            Reference: green ray center. In ZEMAX, chief ray is used as reference, so our result is slightly different from ZEMAX.
        """
        H = 31
        scale = self.calc_scale_ray(depth)

        # ==> Use green light for reference
        if ref:
            ray = self.sample_point_source(M=H, spp=GEO_SPP, depth=depth, R=self.sensor_size[0]/2*scale, pupil=True, wvln=DEFAULT_WAVE)
            ray, _, _ = self.trace(ray)
            p_green = ray.project_to(self.d_sensor)
            p_center_ref = (p_green * ray.ra.unsqueeze(-1)).sum(0) / ray.ra.sum(0).add(0.0001).unsqueeze(-1)
    
        # ==> Calculate RMS errors
        rms = []
        rms_on_axis = []
        rms_off_axis = []
        for wvln in WAVE_RGB:
            ray = self.sample_point_source(M=H, spp=GEO_SPP, depth=depth, R=self.sensor_size[0]/2*scale, pupil=True, wvln=wvln)
            ray, _, _ = self.trace(ray)
            o2 = ray.project_to(self.d_sensor)
            o2_center = (o2*ray.ra.unsqueeze(-1)).sum(0)/ray.ra.sum(0).add(0.0001).unsqueeze(-1)
            
            if ref:
                o2_norm = (o2 - p_center_ref) * ray.ra.unsqueeze(-1)
            else:
                o2_norm = (o2 - o2_center) * ray.ra.unsqueeze(-1)   # normalized to center (0, 0)

            rms.append(torch.sqrt(torch.sum(o2_norm**2 * ray.ra.unsqueeze(-1)) / torch.sum(ray.ra)))
            rms_on_axis.append(torch.sqrt(torch.sum(o2_norm[:, H//2+1, H//2+1, :]**2 * ray.ra[:, H//2+1, H//2+1].unsqueeze(-1)) / torch.sum(ray.ra[:, H//2, H//2])))
            rms_off_axis.append(torch.sqrt(torch.sum(o2_norm[:, 0, 0, :]**2 * ray.ra[:, 0, 0].unsqueeze(-1)) / torch.sum(ray.ra[:, 0, 0])))

        rms_radius = sum(rms) / len(rms)
        rms_radius_on_axis = sum(rms_on_axis) / len(rms_on_axis)
        rms_radius_off_axis = sum(rms_off_axis) / len(rms_off_axis)
        return rms_radius, rms_radius_on_axis, rms_radius_off_axis

    # ====================================================================================
    # Lesn file IO
    # ====================================================================================
    def write_lens_json(self, filename='./test.json'):
        """ Write the lens into .json file.
        """
        data = {}
        data['foclen'] = self.foclen
        data['fnum'] = self.fnum
        data['r_last'] = self.r_last
        data['d_sensor'] = self.d_sensor
        data['sensor_size'] = self.sensor_size
        data['surfaces'] = []
        for i, s in enumerate(self.surfaces):
            surf_dict = s.surf_dict()
            
            if i < len(self.surfaces) - 1:
                surf_dict['d_next'] = self.surfaces[i+1].d.item() - self.surfaces[i].d.item()
            else:
                surf_dict['d_next'] = self.d_sensor - self.surfaces[i].d.item()

            if surf_dict.get('mat1') is None:
                surf_dict['mat1'] = self.materials[i].name
                surf_dict['mat2'] = self.materials[i+1].name
            
            data['surfaces'].append(surf_dict)

        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)


    def read_lens_json(self, filename='./test.json'):
        """ Read the lens from .json file.
        """
        self.surfaces = []
        self.materials = []
        with open(filename, 'r') as f:
            data = json.load(f)
            for surf_dict in data['surfaces']:
                if surf_dict['type'] == 'Aspheric':
                    s = Aspheric(r=surf_dict['r'], d=surf_dict['d'], c=surf_dict['c'], k=surf_dict['k'], ai=surf_dict['ai'], mat1=surf_dict['mat1'], mat2=surf_dict['mat2'], device=self.device)
                
                elif surf_dict['type'] == 'Stop':
                    s = Aspheric(r=surf_dict['r'], d=surf_dict['d'], c=surf_dict['c'], mat1=surf_dict['mat1'], mat2=surf_dict['mat2'], device=self.device)
                
                elif surf_dict['type'] == 'Spheric':
                    s = Aspheric(r=surf_dict['r'], d=surf_dict['d'], c=surf_dict['c'], mat1=surf_dict['mat1'], mat2=surf_dict['mat2'], device=self.device)
                
                else:
                    raise Exception('Surface type not implemented.')
                
                self.surfaces.append(s)
                self.materials.append(Material(surf_dict['mat1']))

        self.materials.append(Material(surf_dict['mat2']))
        self.r_last = data['r_last']
        self.d_sensor = data['d_sensor']

