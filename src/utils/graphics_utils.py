#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))



def compute_rays(projection_matrix, T_world_camera, image_width, image_height):
    # Unproject rays to store local rays
    umap = torch.linspace(0.5, image_width-0.5, image_width, device=projection_matrix.device)
    vmap = torch.linspace(0.5, image_height-0.5, image_height, device=projection_matrix.device)
    umap, vmap = torch.meshgrid(umap, vmap, indexing='xy')
    points_2d = torch.stack((umap, vmap, torch.ones_like(umap, device=projection_matrix.device)), -1).float()
    
    local_rays = torch.einsum("ij,mnj -> mni",projection_matrix[:3,:3].inverse(),points_2d)
    local_rays = torch.cat((local_rays,torch.ones(local_rays.shape[:-1],device=local_rays.device)[...,None]),dim=-1).float()
    world_rays = torch.einsum("ij,mnj -> mni",T_world_camera, local_rays)
    return  world_rays/world_rays[...,None].norm(dim=-1)


def rot_to_euler(R) :
    "Implementation based on https://www.eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf"
    
    tol = 1e-4  
    singular = (1-tol<abs(R[2,0])<1+tol) # If equal to +-1
 
    if  not singular :
        theta1 = -torch.asin(R[2,0])
        theta2 = math.pi - theta1
        cos_t1 = torch.cos(theta1)
        cos_t2 = torch.cos(theta2)
        psi1 = torch.atan2(R[2,1]/cos_t1, R[2,2]/cos_t1)
        psi2 = torch.atan2(R[2,1]/cos_t2, R[2,2]/cos_t2)
        phi1 = torch.atan2(R[1,0]/cos_t1, R[0,0]/cos_t1)
        phi2 = torch.atan2(R[1,0]/cos_t2, R[0,0]/cos_t2)
        angles_1 = torch.hstack([theta1, psi1, phi1])
        angles_2 = torch.hstack([theta2, psi2, phi2])
        euler_angles = angles_1 if angles_1.abs().sum()<  angles_2.abs().sum() else angles_2
    else :
        phi = torch.tensor([0], device=R.device)
        if -1+tol < R[2,0] < -1-tol :
            theta =  torch.tensor([math.pi/2], device=R.device)
            psi = phi+torch.atan2(R[0,1, R[0,2]])
        else:
            theta = torch.tensor([-math.pi/2], device=R.device)
            psi = -phi+torch.atan2(-R[0,1], -R[0,2])
        euler_angles = torch.hstack([theta, psi, phi])

    return euler_angles