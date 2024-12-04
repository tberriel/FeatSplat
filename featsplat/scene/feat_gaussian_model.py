#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# This file is a derivative modification from the original gaussian_model.py file
# 

import torch

import numpy as np
from torch import nn
import os
from plyfile import PlyData, PlyElement
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, get_linear_lr_func, build_rotation, knn
from utils.system_utils import mkdir_p
from utils.graphics_utils import BasicPointCloud

from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

class FeatGaussianModel(GaussianModel):

    def __init__(self, opt):
        super().__init__(0)
        self.active_sh_degree = opt.sh_degree
        self.max_sh_degree = opt.sh_degree
        self.n_latents = opt.n_latents  
        self.n_classes = opt.n_classes
        self.pixel_embedding = opt.pixel_embedding
        self.pos_embedding = opt.pos_embedding
        self.rot_embedding = opt.rot_embedding
        self.p_embedding  = None
        self._features = torch.empty(0)
        self.n_classes = opt.n_classes

        embedding_size =0
        if opt.pixel_embedding:
            embedding_size +=2
        if opt.pos_embedding:
            embedding_size +=3
        if self.rot_embedding:
            embedding_size +=3
        try:
            n_neurons =  opt.n_neurons
            h_layers = opt.h_layers
        except:
            n_neurons = 64
            h_layers = 0
        mlp = [nn.Linear(opt.n_latents+embedding_size, n_neurons),
            nn.SiLU()]
        for _ in range(h_layers):
            mlp+=[nn.Linear(n_neurons, n_neurons),
                nn.SiLU()]
        mlp+=[nn.Linear(n_neurons,3*(opt.sh_degree+1)**2+opt.n_classes)]
        self.mlp = nn.Sequential(*mlp).cuda()
        
        self.setup_functions()

    def capture(self):
        return (self.n_latents,
            self._xyz,
            self._features,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.mlp.state_dict(),
        )
    
    def restore(self, model_args, training_args):
        (self.n_latents, 
        self._xyz, 
        self._features, 
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        mlp_dict) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.mlp.load_state_dict(mlp_dict)
    
    @property
    def get_features(self):
        """Transform stred features to return SH features"""
        return self._features
    
    def nn_forward(self, projected_features, camera_pos, camera_rot, camera_rays):
        """ 
        - Input is n_latentsxHxW
        - Output is 3xHxW
        """
        _, h, w = projected_features.shape
        x = projected_features.flatten(1,2).permute(1,0)
        embeddings = []
        if self.pos_embedding:
            camera_pos = camera_pos[None,...].repeat((h*w, 1))
            embeddings.append(camera_pos)
        if self.pixel_embedding:
            if self.p_embedding is None or (self.p_embedding.shape[0] == h*w):
                umap = torch.linspace(-1, 1, w, device = projected_features.device)
                vmap = torch.linspace(-1, 1, h, device = projected_features.device)
                umap, vmap = torch.meshgrid(umap, vmap, indexing='xy')
                points_2d = torch.stack((umap, vmap), -1).float()
                self.p_embedding =  points_2d.flatten(0,1)
            embeddings.append(self.p_embedding)

        if self.rot_embedding:
            embeddings.append(camera_rot)
            camera_rot = camera_rot[None,...].repeat((h*w, 1))
            
        x = torch.cat([x]+embeddings, axis=-1 )

        rendered_image = self.mlp(x).permute(1,0)[...,None].reshape((3+self.n_classes,h,w))

        if self.n_classes > 0: 
            segmentation_image = rendered_image[3*(self.active_sh_degree+1)**2:]
            rendered_image = rendered_image[:3*(self.active_sh_degree+1)**2]
        else:
            segmentation_image = None

        if self.active_sh_degree>0:
            rendered_image = eval_sh(self.active_sh_degree,rendered_image.permute(1,2,0).unsqueeze(-1).reshape((h,w,3,(self.active_sh_degree+1)**2)), camera_rays)
            #rendered_image = torch.clamp_min(rendered_image+0.5,0.0)
            rendered_image = torch.sigmoid(rendered_image).permute(2,0,1)
        else:
            rendered_image = torch.sigmoid(rendered_image)
        return rendered_image, segmentation_image

    def oneupSHdegree(self):
        raise "FeatGaussianModel does not allow a variable number of SH degrees."

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        features = torch.randn((fused_point_cloud.shape[0],self.n_latents)) 

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        points = torch.from_numpy(np.asarray(pcd.points)).float().cuda()
        dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        scales = torch.log(torch.sqrt(dist2_avg))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features = nn.Parameter(features.contiguous().requires_grad_(True).cuda())
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features], 'lr': training_args.feature_lr_init, "name": "f_nn"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [x for x in self.mlp.parameters()], 'lr': training_args.mlp_lr_init, "name": "mlp"},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.mlp_scheduler_args = get_linear_lr_func(lr_init= training_args.mlp_lr_init,
                                                    lr_final=training_args.mlp_lr_final,
                                                    max_steps=training_args.mlp_lr_max_steps)
        self.features_scheduler_args = get_linear_lr_func(lr_init= training_args.feature_lr_init,
                                                    lr_final=training_args.feature_lr_final,
                                                    max_steps=training_args.feature_lr_max_steps)
        

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        super().update_learning_rate(iteration)
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "mlp":
                lr = self.mlp_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "f_nn":
                lr = self.features_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        
        for i in range(self._features.shape[1]):
            l.append('f_nn_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_nn = self._features.detach().contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_nn, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        mlp_path = os.path.join(os.path.split(path)[0],"mlp.ckpt")
        torch.save(self.mlp.state_dict(),mlp_path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_nn_")]
        f_names = sorted(f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(f_names)==self.n_latents
        features = np.zeros((xyz.shape[0], len(f_names)))
        for idx, attr_name in enumerate(f_names):
            features[:, idx] = np.asarray(plydata.elements[0][attr_name])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features = nn.Parameter(torch.tensor(features, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        mlp_path = os.path.join(os.path.split(path)[0],"mlp.ckpt")
        self.mlp.load_state_dict(torch.load(mlp_path, weights_only=True))

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name']=="mlp":
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features = optimizable_tensors["f_nn"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name']=="mlp":
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_nn": new_features,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features = optimizable_tensors["f_nn"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features = self._features[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features = self._features[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features, new_opacities, new_scaling, new_rotation)