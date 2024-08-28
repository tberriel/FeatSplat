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
# This file includes derivatives from the original gaussian-splatting software
# 
import torch
from scene.gaussian_model import GaussianModel
from gsplat.rendering import rasterization

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None,features_splatting = False):
    """
    """
    means = pc.get_xyz
    quats = pc.get_rotation
    scales = pc.get_scaling
    opacities = pc.get_opacity
    colors = pc.get_features
    viewmats = viewpoint_camera.world_view_transform.transpose(0,1)
    Ks = viewpoint_camera.intrinsic_matrix
    width=int(viewpoint_camera.image_width)
    height=int(viewpoint_camera.image_height)

    # render
    latent_image, alphas, meta = rasterization(
       means, quats, scales, opacities.flatten(), colors[None], viewmats[None], Ks[None], width, height
    )
    if features_splatting:
        rendered_image, segmentation_image = pc.nn_forward(latent_image[0].permute(2,0,1),viewpoint_camera.camera_center, viewpoint_camera.camera_rot, None)
    else:
        rendered_image, segmentation_image = latent_image, None

    return {"render": rendered_image,
            "segmentation": None,
            #"viewspace_points": screenspace_points,
            "visibility_filter" : meta["radii"] > 0,
            "radii": meta["radii"]}