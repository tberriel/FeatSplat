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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from deep_gaussian_renderer import render
from utils.seg_utils import mapClassesToRGB, loadSemanticClasses
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from deep_gaussian_model import DeepGaussianModel, GaussianModel

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, save, gaussian_splatting):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    if save:
        makedirs(render_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)

    if gaussians.n_classes>0:
        data_mapping, _ = loadSemanticClasses(n = gaussians.n_classes)    
        render_sem_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_sem")
        gts_sem_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_sem")
        makedirs(render_sem_path, exist_ok=True)
        makedirs(gts_sem_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if gaussian_splatting:
            rendering = render(view, gaussians, pipeline, background)["render"]
        else: 
            out = render(view, gaussians, pipeline, background, override_color=gaussians.get_features, features_splatting=True)
            rendering = out["render"]
            if gaussians.n_classes>0:
                sem_render, _ = mapClassesToRGB(out["segmentation"].argmax(0), data_mapping)
                sem_gt, _ = mapClassesToRGB(view.original_semantic.cuda().squeeze(0), data_mapping)
                
        if save:
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            if gaussians.n_classes>0:

                torchvision.utils.save_image(torch.from_numpy(sem_render).permute(2,0,1), os.path.join(render_sem_path, '{0:05d}'.format(idx) + ".png"))
                torchvision.utils.save_image(torch.from_numpy(sem_gt).permute((2,0,1)), os.path.join(gts_sem_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, gaussian_splatting : bool, save : bool = True):
    with torch.no_grad():
        if gaussian_splatting:
            assert dataset.n_classes == 0, "Gaussian Splatting does not predict semantics. Set n_classes to 0."
            gaussians = GaussianModel(dataset.sh_degree)
        else:
            gaussians = DeepGaussianModel(dataset)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        if gaussian_splatting:
            bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        else:
            bg_color = [0 for _ in range(gaussians.n_latents)] #if dataset.white_background else [1 for _ in range(gaussians.n_latents)]# Let's start with black background, ideally, background light could also be learnt as a latent vector
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, save, gaussian_splatting)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, save, gaussian_splatting)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save", action="store_false", default=True)
    parser.add_argument('--gaussian_splatting', action="store_true", default=False)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.gaussian_splatting, args.save)