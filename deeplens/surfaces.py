import torch
import numpy as np
import torch.nn.functional as nnF

from .basics import *
from .utils import *

class Surface(DeepObj):
    def __init__(self, r, d, mat1, mat2, is_square=False, device=DEVICE):
        super(Surface, self).__init__()
        if torch.is_tensor(d):
            self.d = d.to(device)
        else:
            self.d = torch.tensor([d]).float().to(device)

        self.r = float(r)   # r is not differentiable
        self.is_square = is_square
        if is_square:
            self.h = r * np.sqrt(2)
            self.w = r * np.sqrt(2)

        self.mat1 = Material(mat1)
        self.mat2 = Material(mat2)        
        self.to(device)

        self.NEWTONS_MAXITER = 10
        self.NEWTONS_TOLERANCE_TIGHT = 10e-6 # in [mm], i.e. 10 [nm] here (up to <10 [nm])
        self.NEWTONS_TOLERANCE_LOOSE = 50e-6 # in [mm], i.e. 100 [nm] here (up to <10 [nm])
        self.NEWTONS_STEP_BOUND = 5 # [mm], maximum time step in Newton's iteration
        self.APERTURE_SAMPLING = 257
        

    # ==============================
    # Intersection and Refraction
    # ==============================
    def ray_reaction(self, ray):
        """ Compute output ray after intersection and refraction.
        """
        # Intersection
        ray = self._newtons_method(ray)
        
        # Determine ray direction and refractive index
        wvln = ray.wvln
        forward = (ray.d * ray.ra.unsqueeze(-1))[...,2].sum() > 0
        if forward:
            eta = self.mat1.ior(wvln) / self.mat2.ior(wvln)
            n = self.mat1.ior(wvln)
        else:
            eta = self.mat2.ior(wvln) / self.mat1.ior(wvln)
            n = self.mat2.ior(wvln)

        # Refraction
        ray = self._refract(ray, eta)

        return ray


    def _newtons_method(self, ray, maxt=MAXT):
        # 1. inital guess of t
        t0 = (self.d - ray.o[...,2]) / ray.d[...,2]   # if the shape of aspheric surface is strange, will hit the back surface region instead 

        # 2. use newton's method to update t to approach final results
        with torch.no_grad():
            it = 0
            t = t0  # initial guess of t
            ft = maxt * torch.ones_like(ray.o[...,2])
            while (torch.abs(ft) > self.NEWTONS_TOLERANCE_LOOSE).any() and (it < self.NEWTONS_MAXITER):
                it += 1

                new_o = ray.o + ray.d * t.unsqueeze(-1)
                new_x, new_y = new_o[...,0], new_o[...,1]
                valid = self._valid(new_x, new_y) & (ray.ra>0)
                
                ft = self.g(new_x, new_y, valid) + self.d - new_o[...,2]
                dxdt, dydt, dzdt = ray.d[...,0], ray.d[...,1], ray.d[...,2]
                dfdx, dfdy, dfdz = self._dfdxyz(new_x, new_y)
                dfdt = dfdx*dxdt + dfdy*dydt + dfdz*dzdt
                t = t - torch.clamp((ft+1e-9)/(dfdt+1e-6), -self.NEWTONS_STEP_BOUND, self.NEWTONS_STEP_BOUND)

            t1 = t - t0

        # 3. do one more iteration to re-gain gradient
        t = t0 + t1

        new_o = ray.o + ray.d * t.unsqueeze(-1)
        new_x, new_y = new_o[...,0], new_o[...,1]
        valid = self._valid(new_x, new_y) & (ray.ra>0)
        
        ft = self.g(new_x, new_y, valid) + self.d - new_o[...,2]
        dxdt, dydt, dzdt = ray.d[...,0], ray.d[...,1], ray.d[...,2]
        dfdx, dfdy, dfdz = self._dfdxyz(new_x, new_y)
        dfdt = dfdx*dxdt + dfdy*dydt + dfdz*dzdt
        t = t - torch.clamp((ft+1e-9)/(dfdt+1e-6), -self.NEWTONS_STEP_BOUND, self.NEWTONS_STEP_BOUND)
        
        # 4. update rays
        new_o = ray.o + ray.d * t.unsqueeze(-1)
        
        # determine valid rays
        with torch.no_grad():
            new_x, new_y = new_o[...,0], new_o[...,1]
            valid = self._valid(new_x, new_y) & (ray.ra>0)
            ft = self.g(new_x, new_y, valid) + self.d - new_o[...,2]
            valid = valid & (torch.abs(ft.detach()) < self.NEWTONS_TOLERANCE_TIGHT) & (t > 0)   # points valid & points accurate & donot go back

        new_o[~valid] = ray.o[~valid]
        ray.o = new_o
        ray.ra = ray.ra * valid

        return ray


    def _refract(self, ray, eta, approx=False):
        """ Snell's law (surface normal n defined along the positive z axis)
            https://physics.stackexchange.com/a/436252/104805
            https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/reflection-refraction-fresnel

            We follow the first link and normal vector should have the same direction with incident ray(veci), but by default it
            points to left. We use the second link to check.

            veci: incident ray
            vect: refractive ray
            eta: relevant refraction coefficient, eta = eta_i/eta_t
        """
        # 1, process eta
        if type(eta) is float:
            eta = eta
        else:
            if np.prod(eta.shape) > 1:
                eta = eta[..., None]

        # 2, compute normal
        n = self._normal(ray)
        forward = (ray.d*ray.ra.unsqueeze(-1))[...,2].sum()>0
        if forward:
            n = -n

        # 3, compute refraction
        cosi = torch.sum(ray.d * n, axis=-1)   # n * i

        # we set a boundary condition: 1 - eta**2 * (1 - cosi**2) > 0.04
        valid = (eta**2*(1-cosi**2) < 1) & (ray.ra>0)

        # FIXED: an error occurs here due to cost2 * valid, which results in divided by 0 during computing derivative
        sr = torch.sqrt(1 - eta**2 * (1 - cosi.unsqueeze(-1)**2) * valid.unsqueeze(-1))  # square root term

        # first term: vertical, second term: parallel. already normalized if n and ray.d is normalized. 
        new_d = sr * n + eta * (ray.d - cosi.unsqueeze(-1) * n)
        new_d[~valid] = ray.d[~valid]
        
        # 4, update ray directions
        ray.d = new_d
        ray.ra = ray.ra * valid

        return ray


    def _normal(self, ray):
        """ Calculate normal vector of the surface at intersection point.
        """
        x, y, z = ray.o[...,0], ray.o[...,1], ray.o[...,2]
        nx, ny, nz = self._dfdxyz(x, y)
        n = torch.stack((nx, ny, nz), axis=-1)
        n = nnF.normalize(n, p=2, dim=-1)

        return n
    

    # =========================================
    # Common methods (must not be overridden)
    # =========================================

    def surface_with_offset(self, x, y):
        """ Calculate z coordinate of the surface at (x, y) with offset.
        """
        return self.surface(x, y) + self.d
    
    def normal(self, x, y):
        ds_dxyz = self.surface_derivatives(x, y)
        return nnF.normalize(torch.stack(ds_dxyz, axis=-1))

    def is_valid(self, p):
        return (self.sdf_approx(p) < 0.0).bool()

    def _valid(self, x, y):
        valid = ((x**2 + y**2) < self.r**2)
        return valid

    def surface_sample(self, N=1000):
        """ Sample uniform points on the surface.
        """
        r_max = self.r
        theta = torch.rand(N)*2*np.pi
        r = torch.sqrt(torch.rand(N)*r_max**2)
        x2 = r * torch.cos(theta)
        y2 = r * torch.sin(theta)
        z2 = torch.full_like(x2, self.d.item())
        o2 = torch.stack((x2,y2,z2), 1).to(self.device)

        return o2
    
    def sag(self, x, y):
        return self.g(x, y)

    def g(self, x, y):
        raise NotImplementedError()
    
    def dgd(self, x, y):
        """
        Derivatives of g: (g'x, g'y).
        """
        raise NotImplementedError()

    def h(self, z):
        raise NotImplementedError()

    def dhd(self, z):
        """
        Derivative of h.
        """
        raise NotImplementedError()

    def surface(self, x, y):
        """
        Solve z from h(z) = -g(x,y).
        """
        raise NotImplementedError()

    def reverse(self):
        raise NotImplementedError()

    # === Default methods (better be overridden)
    def surface_derivatives(self, x, y):
        """
        Returns \nabla f = \nabla (g(x,y) + h(z)) = (dg/dx, dg/dy, dh/dz).
        (Note: this default implementation is not efficient)
        """
        gx, gy = self.dgd(x, y)
        z = self.surface(x, y)
        return gx, gy, self.dhd(z)
        
    def surface_and_derivatives_dot_D(self, t, dx, dy, dz, ox, oy, z):
        """
        Returns g(x,y)+h(z) and dot((g'x,g'y,h'), (dx,dy,dz)).
        (Note: this default implementation is not efficient)
        """
        x = ox + t * dx
        y = oy + t * dy
        s = self.g(x,y) + self.h(z)
        sx, sy = self.dgd(x, y)
        sz = self.dhd(z)
        return s, sx*dx + sy*dy + sz*dz
    

    # =========================================
    # Optimization-related methods
    # =========================================

    def activate_grad(self, activate=True):
        raise NotImplementedError()
    
    def get_optimizer_params(self, lr):
        raise NotImplementedError()

    def get_optimizer(self, lr):
        params = self.get_optimizer_params(lr)
        return torch.optim.Adam(params)

    def surf_dict(self):
        surf_dict = {
            'type': self.__class__.__name__,
            'r': self.r,
            'd': self.d,
            'is_square': self.is_square,
            'mat1': self.mat1.name,
            'mat2': self.mat2.name,
        }

        return surf_dict


class Aspheric(Surface):
    """ This class can represent plane, spheric and aspheric surfaces.

        Aspheric surface: https://en.wikipedia.org/wiki/Aspheric_lens.

        Three kinds of surfaces:
            1. flat: always use round 
            2. spheric: 
            3. aspheric: 
    """
    def __init__(self, r, d, c=0., k=0., ai=None, mat1=None, mat2=None, is_square=False, device=DEVICE, diff=False, square=False):
        """ Initialize an aspherical surface.

        Args:
            r (float): radius/height
            d (float): distance/z coordinate
            c (float): 1/roc
            k (float): conic parameter(k in the wiki) of aspherical surface. 0.0 for spherical lens.
            ai ([type], optional): aspherical parameters.
            is_square (bool, optional): [description]. Defaults to False.
            device: Either call `to(device)` after initialization, or move to device during initialization.
        """
        Surface.__init__(self, r, d, mat1, mat2, is_square, device)
        self.c = torch.Tensor([c])
        self.k = torch.Tensor([k])
        if ai is not None:
            self.ai = torch.Tensor(np.array(ai))
            self.ai_degree = len(ai)
            if self.ai_degree == 4:
                self.ai2 = torch.Tensor([ai[0]])
                self.ai4 = torch.Tensor([ai[1]])
                self.ai6 = torch.Tensor([ai[2]])
                self.ai8 = torch.Tensor([ai[2]])
            elif self.ai_degree == 5:
                self.ai2 = torch.Tensor([ai[0]])
                self.ai4 = torch.Tensor([ai[1]])
                self.ai6 = torch.Tensor([ai[2]])
                self.ai8 = torch.Tensor([ai[3]])
                self.ai10 = torch.Tensor([ai[4]])
            elif self.ai_degree == 6:
                for i, a in enumerate(ai):
                    exec(f'self.ai{2*i+2} = torch.Tensor([{a}])')
            else:
                for i, a in enumerate(ai):
                    exec(f'self.ai{2*i+2} = torch.Tensor([{a}])')
        else:
            self.ai = None
            self.ai_degree = 0
        
        self.is_square = square
        self.to(device)


    def init(self, ai_degree=6):
        """ Initialize all parameters.
        """
        self.init_c()
        self.init_k()
        self.init_ai(ai_degree=ai_degree)
        self.init_d()


    def init_c(self, c_bound=0.0002):
        """ Initialize lens surface c parameters by small values between [-0.05, 0.05], 
            which means roc should be (-inf, 20) or (20, inf)
        """
        self.c = c_bound * (torch.rand(1) - 0.5).to(self.device)

    def init_ai(self, ai_degree=3, bound=0.0001):
        """ If ai is None, set to random value.
            For different length, create a new initilized value and set original ai.
        """
        old_ai_degree = self.ai_degree
        self.ai_degree = ai_degree
        if old_ai_degree == 0:
            if ai_degree == 4:
                self.ai2 = (torch.rand(1, device=self.device)-0.5) * bound * 10
                self.ai4 = (torch.rand(1, device=self.device)-0.5) * bound
                self.ai6 = (torch.rand(1, device=self.device)-0.5) * bound * 0.1
                self.ai8 = (torch.rand(1, device=self.device)-0.5) * bound * 0.01
            elif ai_degree == 5:
                self.ai2 = (torch.rand(1, device=self.device)-0.5) * bound * 10
                self.ai4 = (torch.rand(1, device=self.device)-0.5) * bound
                self.ai6 = (torch.rand(1, device=self.device)-0.5) * bound * 0.1
                self.ai8 = (torch.rand(1, device=self.device)-0.5) * bound * 0.01
                self.ai10 = (torch.rand(1, device=self.device)-0.5) * bound* 0.001
            elif ai_degree == 6:
                for i in range(1, self.ai_degree+1):
                    exec(f'self.ai{2*i} = (torch.rand(1, device=self.device)-0.5) * bound * 0.1 ** {i-2}')
            else:
                raise Exception('Wrong ai degree')
        else:
            for i in range(old_ai_degree+1, self.ai_degree+1):
                exec(f'self.ai{2*i} = (torch.rand(1, device=self.device)-0.5) * bound * 0.1 ** {i-2}')

    
    def init_k(self, bound=1):
        """ When k is 0, set to a random value.
        """
        if self.k == 0:
            k = torch.rand(1)* bound
            self.k = k.to(self.device) 


    def init_d(self, bound=0.1):
        return

    # ------------------------------------------------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------------------------------------------------
    def ray_reaction(self, ray):
        """ Compute output ray after intersection and refraction.

            In each step, first get a guess of new o and d, then compute valid and only update valid rays. 
        """
        wvln = ray.wvln
        k = 2 * np.pi / (wvln * 1e-3) # wvln in [um], k in [mm]-1

        forward = (ray.d * ray.ra.unsqueeze(-1))[...,2].sum()>0
        if forward:
            eta = self.mat1.ior(wvln) / self.mat2.ior(wvln)
            n = self.mat1.ior(wvln)
        else:
            eta = self.mat2.ior(wvln) / self.mat1.ior(wvln)
            n = self.mat2.ior(wvln)


        # Aperture and filter
        if self.c.item() == 0:
            # -------------------------------------
            # Intersection
            # ------------------------------------- 
            t = (self.d - ray.o[...,2]) / ray.d[...,2]
            o0 = ray.o.clone()
            new_o = ray.o + t.unsqueeze(-1) * ray.d
            if self.is_square:
                height = self.r
                width = self.r
                valid = (torch.abs(new_o[...,0])<=height) & (torch.abs(new_o[...,1])<=width) & (ray.ra>0)
            else:
                valid = (torch.sqrt(new_o[...,0]**2 + new_o[...,1]**2) <= self.r) & (ray.ra>0)

            # => Update position
            new_o[~valid] = ray.o[~valid]
            ray.o = new_o

            if ray.coherent:
                # => Update phase
                if t.min() > 100:
                    # High-precision calculation
                    opd = - (ray.o * o0).sum(-1) / (o0 * o0).sum(-1).sqrt() 
                    ray.opl = ray.opl + opd

                else:
                    new_opl = ray.opl + n * t
                    new_opl[~valid] = ray.opl[~valid]
                    ray.opl = new_opl

                    new_phi = ray.phi + torch.fmod(n * k * (t - t.min(0).values), 2 * np.pi)
                    new_phi[~valid] = ray.phi[~valid]
                    ray.phi = new_phi


            # => Update validity
            ray.ra = ray.ra * valid

            # -------------------------------------
            # Refraction
            # ------------------------------------- 
            if eta != 1:
                ray = self._refract(ray, eta)

            return ray

        # Spheric surface
        elif self.ai is None and self.k == 0:
            # -------------------------------------
            # Intersection
            # -------------------------------------     
            valid, t = self._newtons_method(ray, MAXT)

            # => Update position
            new_o = ray.o + t.unsqueeze(-1) * ray.d
            valid = (new_o[...,0]**2 + new_o[...,1]**2 <= self.r**2) & (t>=0) & (ray.ra>0)
            new_o[~valid] = ray.o[~valid]
            ray.o = new_o

            if ray.coherent:
                # => Update opl (phase)
                new_opl = ray.opl + n * t
                new_opl[~valid] = ray.opl[~valid]
                ray.opl = new_opl

                new_phi = ray.phi + torch.fmod(n * k * (t - t.min(0).values), 2 * np.pi)
                new_phi[~valid] = ray.phi[~valid]
                ray.phi = new_phi

            # => Update validity
            ray.ra = ray.ra * valid
            

            # -------------------------------------
            # Refraction
            # ------------------------------------- 
            ray = self._refract(ray, eta)
            
            return ray


        # Aspheric surface
        else:
            # -------------------------------------
            # Intersection
            # ------------------------------------- 
            valid, t = self._newtons_method(ray, MAXT)

            # => Update position
            new_o = ray.o + t.unsqueeze(-1) * ray.d
            new_o[~valid] = ray.o[~valid]   # invalid rays will stay in the original positions
            ray.o = new_o
            
            if ray.coherent:
                # => Update phase
                new_opl = ray.opl + n * t
                new_opl[~valid] = ray.opl[~valid]
                ray.opl = new_opl

                new_phi = ray.phi + torch.fmod(n * k * (t - t.min(0).values), 2 * np.pi)
                new_phi[~valid] = ray.phi[~valid]
                ray.phi = new_phi

            # => Update validity
            ray.ra = ray.ra * valid
            
            # -------------------------------------
            # Refraction
            # -------------------------------------
            ray = self._refract(ray, eta)

            return ray


    def _newtons_method(self, ray, maxt):
        """ Use Newton's method to compute intersection.
            https://en.wikipedia.org/wiki/Newton%27s_method
            https://en.wikipedia.org/wiki/Aspheric_lens

            We want to solve:
                o' = o + d * t              ---eq.1              
                o'(z) = f(o'(x), o'(y))     ---eq.2

            which can be written as:
                f(t) = 0.

            Newton's method:
                t0 = t0
                t(n+1) = t(n) - f(t)/f'(t)
        """
        # 1. inital guess of t
        t0 = (self.d - ray.o[...,2]) / ray.d[...,2]

        # 2. use newton's method to update t to approach final results
        with torch.no_grad():
            it = 0
            t = t0
            ft = maxt * torch.ones_like(ray.o[...,2])
            while (torch.abs(ft) > self.NEWTONS_TOLERANCE_LOOSE).any() and (it < self.NEWTONS_MAXITER):
                it += 1
                
                new_o = ray.o + ray.d * t.unsqueeze(-1)
                new_x, new_y = new_o[...,0], new_o[...,1]
                valid = self._valid_loose(new_x, new_y) & (ray.ra>0)
                
                ft = self.g(new_x, new_y, valid) + self.d - new_o[...,2]
                if torch.sum(torch.isnan(ft))>0:
                    print('found nan in ft in non-diff newton method.')
                    print(f'self.c = {self.c}')
                    exit(0)
                dr2dt = 2*((ray.d[...,0]**2+ray.d[...,1]**2)*t + (ray.d[...,0]*ray.o[...,0] + ray.d[...,1]*ray.o[...,1]))
                dfdt = self._dsdr2(new_x, new_y, valid) * dr2dt  - ray.d[...,2]
                t = t - torch.clamp(ft/(dfdt + EPSILON), -self.NEWTONS_STEP_BOUND, self.NEWTONS_STEP_BOUND)

            t1 = t - t0

        # 3. do one more iteration to re-gain gradient
        # If we uncommand this line, t0 will have gradient. Surface global move.
        t = t0 + t1

        new_o = ray.o + ray.d * t.unsqueeze(-1)
        new_x, new_y = new_o[...,0], new_o[...,1]
        valid = self._valid(new_x, new_y) & (ray.ra>0)
        
        ft = self.g(new_x, new_y, valid) + self.d - new_o[...,2]
        dr2dt = 2*((ray.d[...,0]**2+ray.d[...,1]**2)*t + (ray.d[...,0]*ray.o[...,0] + ray.d[...,1]*ray.o[...,1]))
        dfdt = self._dsdr2(new_x, new_y, valid) * dr2dt  - ray.d[...,2]

        # t = t0 + t1 - ft/dfdt, t0 and ft/dfdt require gradient, t1 doesnot require gradient.
        t = t - torch.clamp(ft/(dfdt+EPSILON), -self.NEWTONS_STEP_BOUND, self.NEWTONS_STEP_BOUND)
        
        # 4. determine valid rays
        with torch.no_grad():
            new_o = ray.o + ray.d * t.unsqueeze(-1)
            new_x, new_y = new_o[...,0], new_o[...,1]
            valid = self._valid(new_x, new_y) & (torch.abs(ft.detach()) < self.NEWTONS_TOLERANCE_TIGHT) & (ray.ra>0) & (t>0) # t>0 prevents rays to go back

        return valid, t

    
    def _normal(self, ray):
        """ Compute nabla f(x, y, z).
            Normal should be (deltax, deltay, deltaz), deltax = df/dx
            https://mathworld.wolfram.com/NormalVector.html

            Normal vector points to left by default.
        """
        x, y, z = ray.o[...,0], ray.o[...,1], ray.o[...,2]
        
        # ==> Flat plane, square or rectangle
        if self.c == 0:
            deltax = torch.zeros_like(x)
            deltay = torch.zeros_like(y)
            deltaz = torch.full_like(z, -1)

        # ==> Spheric
        # We can either use the formula in the Aspheric situation, or use this simplified formula
        elif self.ai is None and self.k==0:
            R = 1/self.c
            if self.c > 0:
                deltax = 2 * x
                deltay = 2 * y
                deltaz = 2 * z - 2 * (self.d + R)
            else:
                deltax = -2 * x
                deltay = -2 * y
                deltaz = -2 * z + 2 * (self.d + R)


        # ==> Aspheric
        else:
            # https://en.wikipedia.org/wiki/Aspheric_lens
            # here sign of R can control deltax direction, normal vector always points to left.
            valid = ray.ra > 0
            deltax, deltay = self.dgd(x, y, valid)
            deltaz = torch.full_like(deltax, -1)

        # normalize
        n = torch.stack((deltax, deltay, deltaz), axis=-1)
        n = nnF.normalize(n, p=2, dim=-1)

        return n


    def _refract(self, ray, eta, approx=False):
        """ Snell's law (surface normal n defined along the positive z axis)
            https://physics.stackexchange.com/a/436252/104805
            https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/reflection-refraction-fresnel

            We follow the first link and normal vector should have the same direction with incident ray(veci), but by default it
            points to left. We use the second link to check.

            veci: incident ray
            vect: refractive ray
            eta: relevant refraction coefficient, eta = eta_i/eta_t
        """
        # 1, process eta
        if type(eta) is float:
            eta = eta
        else:
            if np.prod(eta.shape) > 1:
                eta = eta[..., None]

        # 2, compute normal
        n = self._normal(ray)
        forward = (ray.d * ray.ra.unsqueeze(-1))[...,2].sum()>0
        if forward:
            n = -n

        # 3, compute refraction
        cosi = torch.sum(ray.d * n, axis=-1)   # n * i

        # we set a boundary condition: 1 - eta**2 * (1 - cosi**2) > 0.04
        # set a deep ray condition, cosi^2 > 0.1
        valid = (cosi**2>0.1) & (eta**2*(1-cosi**2) < 1) & (ray.ra>0)

        # FIXED: an error occurs here due to cost2 * valid, which results in divided by 0 during computing derivative
        sr = torch.sqrt(1 - eta**2 * (1 - cosi.unsqueeze(-1)**2) * valid.unsqueeze(-1))  # square root term

        # first term: vertical, second term: parallel. already normalized if n and ray.d is normalized. 
        new_d = sr * n + eta * (ray.d - cosi.unsqueeze(-1) * n)
        new_d[~valid] = ray.d[~valid]
        
        # 4, update ray directions.
        old_d = ray.d.detach().clone()
        ray.obliq *= torch.sum(new_d * old_d, dim=-1)
        
        ray.d = new_d
        ray.ra = ray.ra * valid
        
        return ray


    # -----------------------------------------------------------------
    # Intermediate variables
    # -----------------------------------------------------------------
    def sag(self, x, y):
        return self.g(x, y)
    
    def g(self, x, y, valid=None):
        """ Compute surface height.
        """
        if valid is None:
            valid = torch.full_like(x, 1)

        x = x * valid
        y = y * valid
        return self._g(x**2 + y**2)


    def dgd(self, x, y, valid):
        """ Compute surface height derivatives to x and y.

            Invalid rays will be replaced by 0
        """
        x = x * valid
        y = y * valid
        r2 = x**2 + y**2
        dsdr2 = self._dgd(r2)
        return dsdr2*2*x, dsdr2*2*y


    def _dsdr2(self, x, y, valid):
        """ Compute surface height derivative to r2.

            If we want to compute derivative to x, we should multiply dr2dx which is 2x.
            If we want to compute derivative to t, we should multiply dr2dt.
        """
        x = x * valid
        y = y * valid
        r2 = x**2 + y**2
        dsdr2 = self._dgd(r2)
        return dsdr2


    def _valid(self, x, y):
        """ Invalid when shape is non-defined and rays exceed boundary.
        """
        if self.k > -1:
            valid = ((x**2 + y**2) < self.r**2) & ((x**2 + y**2) < (1-EPSILON)/self.c**2 / (1+self.k))
        else:
            valid = ((x**2 + y**2) < self.r**2)

        return valid


    def _valid_loose(self, x, y):
        """ Invalid only when shape is non-defined.
        """
        if self.k > -1:
            valid = ((x**2 + y**2) < (1-EPSILON) / self.c**2 / (1+self.k))
        else:
            valid = (x**2 + y**2) > 0

        return valid


    def max_height(self):
        """ Maximum valid height.
        """
        if self.k > -1:
            max_height= torch.sqrt(1/(self.k+1)/(self.c**2)).item() - 0.01
        else:
            # always valid, we can set it to a large value.
            max_height = 100

        return max_height


    # --------------------------------------------------------------------

    def h(self, z):
        return -z

    def dhd(self, z):
        return -torch.ones_like(z)

    def surface(self, x, y):
        """ Use surface funciton to compute z/sag coordinate.
        """
        valid = self._valid_loose(x, y)
        x, y = x*valid, y*valid
        return self._g(x**2 + y**2)

    def surface_derivatives(self, x, y):
        dsdr2 = self._dgd(x**2 + y**2)
        return dsdr2*2*x, dsdr2*2*y, -torch.ones_like(x)

    def surface_and_derivatives_dot_D(self, t, dx, dy, dz, ox, oy, z):
        """ Surface height and derivative to t.
        """
        r2 = (ox + dx * t)**2 + (oy + dy * t)**2
        residual = self._g(r2) - z
        residual_derivative = self._dgd(r2) * 2 * ((dx**2+dy**2)*t**2 + (ox+oy)*t) - dz

        return residual, residual_derivative


    def _g(self, r2):
        """ Compute z(r) according to aspherical function.
        """
        total_surface = r2 * self.c / (1 + torch.sqrt(1 - (1 + self.k) * r2 * self.c**2))

        # after testing, there is no difference between two different expressions
        if self.ai_degree > 0:
            if self.ai_degree == 4:
                total_surface = total_surface + self.ai2 * r2 + self.ai4 * r2 ** 2 + self.ai6 * r2 ** 3 + self.ai8 * r2 ** 4
            elif self.ai_degree == 5:
                total_surface = total_surface + self.ai2 * r2 + self.ai4 * r2 ** 2 + self.ai6 * r2 ** 3 + self.ai8 * r2 ** 4 + self.ai10 * r2 ** 5
            elif self.ai_degree == 6:
                total_surface = total_surface + self.ai2 * r2 + self.ai4 * r2 ** 2 + self.ai6 * r2 ** 3 + self.ai8 * r2 ** 4 + self.ai10 * r2 ** 5 + self.ai12 * r2 ** 6
            elif self.ai_degree == 7:
                total_surface = total_surface + (self.ai2 + (self.ai4 + (self.ai6 + (self.ai8 + (self.ai10 + (self.ai12 + self.ai14 * r2) * r2) * r2) * r2) * r2) * r2) * r2
            elif self.ai_degree == 8:
                total_surface = total_surface + (self.ai2 + (self.ai4 + (self.ai6 + (self.ai8 + (self.ai10 + (self.ai12 + (self.ai14 + self.ai16 * r2)* r2) * r2) * r2) * r2) * r2) * r2) * r2
            else:
                for i in range(1, self.ai_degree+1):
                    exec(f'total_surface += self.ai{2*i} * r2 ** {i}')
                
        return total_surface

    
    def _dgd(self, r2):
        """ Compute d z(r^2)/d r^2
        """
        sf = torch.sqrt(1-(1+self.k)*r2*self.c**2)
        dsdr2 = (1+sf+(1+self.k)*r2*self.c**2/2/sf) *self.c/(1+sf)**2

        if self.ai_degree > 0:
            if self.ai_degree == 4:
                dsdr2 = dsdr2 + self.ai2 + 2 * self.ai4 * r2 + 3 * self.ai6 * r2 ** 2 + 4 * self.ai8 * r2 ** 3
            elif self.ai_degree == 5:
                dsdr2 = dsdr2 + self.ai2 + 2 * self.ai4 * r2 + 3 * self.ai6 * r2 ** 2 + 4 * self.ai8 * r2 ** 3 + 5 * self.ai10 * r2 ** 4
            elif self.ai_degree == 6:
                dsdr2 = dsdr2 + self.ai2 + 2 * self.ai4 * r2 + 3 * self.ai6 * r2 ** 2 + 4 * self.ai8 * r2 ** 3 + 5 * self.ai10 * r2 ** 4 + 6 * self.ai12 * r2 ** 5
            elif self.ai_degree == 8: # debug 
                dsdr2 = dsdr2 + self.ai2 + (2 * self.ai4 + (3 * self.ai6 + (4 * self.ai8 + (5 * self.ai10 + (6 * self.ai12 + (7 * self.ai14 + 8 * self.ai16 * r2)* r2)* r2)* r2) * r2) * r2) * r2 
            else:
                for i in range(1, self.ai_degree+1):
                    exec(f'dsdr2 += {i} * self.ai{2*i} * r2 ** {i-1}')

        return dsdr2


    # -----------------------------------------------------------------
    # Other utils
    # -----------------------------------------------------------------

    def activate_grad(self, activate=True, term=None):
        """ Activate/deactivate greadients.
        """
        self.c.requires_grad_(activate)
        self.d.requires_grad_(activate)

        if self.k != 0:
            self.k.requires_grad_(activate)

        if self.ai_degree > 0:
            if self.ai_degree == 4:
                self.ai2.requires_grad_(activate)
                self.ai4.requires_grad_(activate)
                self.ai6.requires_grad_(activate)
                self.ai8.requires_grad_(activate)
            elif self.ai_degree == 5:
                self.ai2.requires_grad_(activate)
                self.ai4.requires_grad_(activate)
                self.ai6.requires_grad_(activate)
                self.ai8.requires_grad_(activate)
                self.ai10.requires_grad_(activate)
            elif self.ai_degree == 6:
                for i in range(1, self.ai_degree+1):
                    exec(f'self.ai{2*i}.requires_grad_({activate})')
            else:
                raise Exception('Wrong ai degree')


    def get_optimizer_params(self, lr=[1e-4, 1e-4, 1e-1, 1e-4], decay=0.1):
        """ Get optimizer data (parameters and lr).

        Args:
            lr (list, optional): learning rates for c, d, k, ai. Defaults to [1e-4, 1e-4, 1e-1, 1e-4].
            decay (float, optional): decay rate for ai. Defaults to 0.1.
        """
        if isinstance(lr, float):
            lr = [lr, lr, lr*1e3, lr]

        params = []
        if lr[0] > 0 and self.c != 0:
            self.c.requires_grad_(True)
            params.append({'params': [self.c], 'lr': lr[0]})
        if lr[1] > 0:
            self.d.requires_grad_(True)
            params.append({'params': [self.d], 'lr': lr[1]})
        if lr[2] > 0 and self.k != 0:
            params.append({'params': [self.k], 'lr': lr[2]})
        if lr[3] > 0:
            if self.ai_degree == 4:
                # self.ai2.requires_grad_(True)
                self.ai4.requires_grad_(True)
                self.ai6.requires_grad_(True)
                self.ai8.requires_grad_(True)
                # params.append({'params': [self.ai2], 'lr': lr[3]/decay})
                params.append({'params': [self.ai4], 'lr': lr[3]})
                params.append({'params': [self.ai6], 'lr': lr[3]*decay})
                params.append({'params': [self.ai8], 'lr': lr[3]*decay**2})
            if self.ai_degree == 5:
                # self.ai2.requires_grad_(True)
                self.ai4.requires_grad_(True)
                self.ai6.requires_grad_(True)
                self.ai8.requires_grad_(True)
                self.ai10.requires_grad_(True)
                # params.append({'params': [self.ai2], 'lr': lr[3]/decay})
                params.append({'params': [self.ai4], 'lr': lr[3]})
                params.append({'params': [self.ai6], 'lr': lr[3]*decay})
                params.append({'params': [self.ai8], 'lr': lr[3]*decay**2})
                params.append({'params': [self.ai10], 'lr': lr[3]*decay**3})
            if self.ai_degree == 6:
                # self.ai2.requires_grad_(True)
                self.ai4.requires_grad_(True)
                self.ai6.requires_grad_(True)
                self.ai8.requires_grad_(True)
                self.ai10.requires_grad_(True)
                self.ai12.requires_grad_(True)
                # params.append({'params': [self.ai2], 'lr': lr[3]/decay})
                params.append({'params': [self.ai4], 'lr': lr[3]})
                params.append({'params': [self.ai6], 'lr': lr[3]*decay})
                params.append({'params': [self.ai8], 'lr': lr[3]*decay**2})
                params.append({'params': [self.ai10], 'lr': lr[3]*decay**3})
                params.append({'params': [self.ai12], 'lr': lr[3]*decay**4})
            else:
                for i in range(2, self.ai_degree + 1):
                    exec(f'self.ai{2*i}.requires_grad_(True)')
                    exec(f'params.append({{\'params\': [self.ai{2*i}], \'lr\': lr[3]/decay**{i-1}}})')
        
        return params


    @torch.no_grad()
    def perturb(self, ratio=0.001, thickness_precision=0.0005, diameter_precision=0.001):
        """ Randomly perturb surface parameters to simulate manufacturing errors. This function should only be called in the final image simulation stage. 
        
        Args:
            ratio (float, optional): perturbation ratio. Defaults to 0.001.
            thickness_precision (float, optional): thickness precision. Defaults to 0.0005.
            diameter_precision (float, optional): diameter precision. Defaults to 0.001.
        """
        self.r += np.random.randn() * diameter_precision
        if self.c != 0:
            self.c *= 1 + np.random.randn() * ratio
        if self.d != 0:
            self.d += np.random.randn() * thickness_precision
        if self.k != 0:
            self.k *= 1 + np.random.randn() * ratio
        for i in range(1, self.ai_degree+1):
            exec(f'self.ai{2*i} *= 1 + np.random.randn() * ratio')


    def surf_dict(self):
        """ Return a dict of surface.
        """
        if self.c.item() == 0:
            surf_dict = {
                'type': 'Stop',
                'r': self.r,
                'c': self.c.item(),
                'd': self.d.item(),
                'mat1': self.mat1.name,
                'mat2': self.mat2.name,
                }
        elif self.ai is None and self.k == 0:
            surf_dict = {
                'type': 'Spheric',
                'r': self.r,
                'c': self.c.item(),
                'roc': 1/self.c.item(),
                'd': self.d.item(),
                'mat1': self.mat1.name,
                'mat2': self.mat2.name,
                }
        else:
            surf_dict = {
                'type': 'Aspheric',
                'r': self.r,
                'c': self.c.item(),
                'roc': 1/self.c.item(),
                'd': self.d.item(),
                'k': self.k.item(),
                'ai': [],
                'mat1': self.mat1.name,
                'mat2': self.mat2.name,
                }
            for i in range(1, self.ai_degree+1):
                exec(f'surf_dict[\'ai{2*i}\'] = self.ai{2*i}.item()')
                surf_dict['ai'].append(eval(f'self.ai{2*i}.item()'))

        return surf_dict