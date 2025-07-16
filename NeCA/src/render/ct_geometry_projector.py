import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import odl
from odl.contrib import torch as odl_torch

class Initialization_ConeBeam:
    def __init__(self, image_size, image_reso, proj_angle, proj_axis, proj_size, proj_reso, dde, dso):
        '''
        image_size: [z, x, y], assume x = y for each slice image
        proj_size: [h, w]
        '''
        self.param = {}
        
        self.image_size = image_size
        self.image_reso = image_reso
        self.proj_size = proj_size
        self.proj_reso = proj_reso

        self.num_proj = 180       
        self.proj_angle = 2 * np.pi
        self.proj_axis = proj_axis
        self.dde = dde
        self.dso = dso
        
        self.param['nx'] = image_size[1]
        self.param['ny'] = image_size[2]
        self.param['nz'] = image_size[0]
        self.param['sx'] = self.param['nx'] * self.image_reso[1]
        self.param['sy'] = self.param['ny'] * self.image_reso[2]
        self.param['sz'] = self.param['nz'] * self.image_reso[0]

        self.param['start_angle'] = 0
        self.param['end_angle'] = proj_angle
        self.param['proj_axis'] = proj_axis
        self.param['nProj'] = self.num_proj

        self.param['sh'] = proj_size[0] * proj_reso[0]
        self.param['sw'] = proj_size[1] * proj_reso[1]
        self.param['nh'] = proj_size[0]
        self.param['nw'] = proj_size[1]
        self.param['dde'] = dde
        self.param['dso'] = dso

        print("🔍 [DEBUG] Initialization_ConeBeam 参数：")
        for k, v in self.param.items():
            print(f"  {k}: {v}")
        print("----")

def build_conebeam_gemotry(param):
    print("🔍 [DEBUG] 进入 build_conebeam_gemotry()")

    # 打印 param.param 内容
    for k, v in param.param.items():
        print(f"  param[{k}]: {v}")
    print("----")

    reco_space = odl.uniform_discr(
        min_pt=[-param.param['sx'] / 2.0, -param.param['sy'] / 2.0, -param.param['sz'] / 2.0],
        max_pt=[ param.param['sx'] / 2.0,  param.param['sy'] / 2.0,  param.param['sz'] / 2.0], 
        shape=[param.param['nx'], param.param['ny'], param.param['nz']],
        dtype='float32'
    )

    print(f"🟢 reco_space shape = {reco_space.shape}")
    print(f"🟢 reco_space cell size = {reco_space.cell_sides}")
    print("----")

    angle_partition = odl.uniform_partition(
        min_pt=param.param['start_angle'], 
        max_pt=param.param['end_angle'],
        shape=param.param['nProj']
    )

    detector_partition = odl.uniform_partition(
        min_pt=[-param.param['sh'] / 2.0, -param.param['sw'] / 2.0], 
        max_pt=[ param.param['sh'] / 2.0,  param.param['sw'] / 2.0],
        shape=[param.param['nh'], param.param['nw']]
    )

    geometry = odl.tomo.ConeBeamGeometry(
        apart=angle_partition,
        dpart=detector_partition,
        src_radius=param.param['dso'],
        det_radius=param.param['dde'],
        src_to_det_init=(0, 1, 0),
        det_axes_init=[(1, 0, 0), (0, 0, 1)],
        axis=param.param['proj_axis']
    )

    ray_trafo = odl.tomo.RayTransform(
        vol_space=reco_space,
        geometry=geometry,
        impl='astra_cuda'
    )

    print("🟢 RayTransform 创建成功")
    
    FBPOper = odl.tomo.fbp_op(
        ray_trafo=ray_trafo, 
        filter_type='Ram-Lak',
        frequency_scaling=1.0
    )

    print("✅ FBP Operator 创建成功")

    return ray_trafo, FBPOper

class Projection_ConeBeam(nn.Module):
    def __init__(self, param):
        super(Projection_ConeBeam, self).__init__()
        self.param = param

        print("🧪 DEBUG: 调用 build_conebeam_gemotry()")
        ray_trafo, fbpOper = build_conebeam_gemotry(self.param)

        self.trafo = odl_torch.OperatorModule(ray_trafo)
        self.back_projector = odl_torch.OperatorModule(ray_trafo.adjoint)

    def forward(self, x):
        return self.trafo(x)
    
    def back_projection(self, x):
        return self.back_projector(x)

class FBP_ConeBeam(nn.Module):
    def __init__(self, param):
        super(FBP_ConeBeam, self).__init__()
        self.param = param

        ray_trafo, FBPOper = build_conebeam_gemotry(self.param)
        self.fbp = odl_torch.OperatorModule(FBPOper)

    def forward(self, x):
        return self.fbp(x)

    def filter_function(self, x):
        return self.filter(x)

class ConeBeam3DProjector():
    def __init__(self, image_size, image_reso, proj_angle, proj_axis, proj_size, proj_reso, dde, dso):
        self.image_size = image_size
        self.image_reso = image_reso
        self.proj_size = proj_size
        self.proj_reso = proj_reso
        self.proj_angle = proj_angle
        self.proj_axis = proj_axis
        self.dde = dde
        self.dso = dso

        geo_param = Initialization_ConeBeam(
            image_size, image_reso, proj_angle, proj_axis, proj_size, proj_reso, dde, dso
        )

        self.forward_projector = Projection_ConeBeam(geo_param)
        self.fbp = FBP_ConeBeam(geo_param)

    def forward_project(self, volume):
        return self.forward_projector(volume)

    def backward_project(self, projs):
        return self.fbp(projs)