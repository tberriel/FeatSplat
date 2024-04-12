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

import os
import torch
from random import randint
from deep_gaussian_renderer import render, network_gui
import sys
from scene import Scene
from modules.gaussian_splatting.utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import PipelineParams, OptimizationParams
from deep_gaussian_model import DeepGaussianModel
from arguments import ModelParams
from utils.seg_utils import mapClassesToRGB, loadSemanticClasses
from keyboard import wait
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import matplotlib.pyplot as plt
# Function to plot the image
def plot_seg_image(seg_image, data_mapping, fig = 0):
    plt.figure(fig)
    image, lgnd_classes = mapClassesToRGB(seg_image, data_mapping)

    # Create legend elements
    legend_handles = []
    for i in range(len(lgnd_classes["labels"])):
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=lgnd_classes["rgb"][i], label=lgnd_classes["labels"][i]))

    # Add legend to the plot

    plt.clf()
    plt.imshow(image)
    plt.legend(handles=legend_handles, title="Class Legend",bbox_to_anchor=(1.6, 1), borderaxespad=0.5,)
    plt.tight_layout()
    plt.axis('off')
    plt.draw()
    plt.pause(0.01)

def streaming(dataset, opt, pipe, checkpoint):
    gaussians = DeepGaussianModel(dataset.sh_degree, dataset.n_latents, dataset.n_classes, dataset.pixel_embedding, dataset.pos_embedding,dataset.rot_embedding)
    gaussians.training_setup(opt)
    if checkpoint and False:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    scene = Scene(dataset, gaussians, load_iteration=30000)

    bg_color = [0 for _ in range(gaussians.n_latents)] # Let's start with black background, ideally, background light could also be learnt as a latent vector
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if dataset.n_classes>0:
        data_mapping, _ = loadSemanticClasses(n = dataset.n_classes)
        plt.figure(0)
        #fig, ax = plt.subplots()
    with torch.no_grad():
        while True:     
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        out =  render(custom_cam, gaussians, pipe, background, scaling_modifer, override_color=gaussians.get_features, features_splatting=True)
                        net_image = out["render"]
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                        if dataset.n_classes>0:
                            seg_image = out["segmentation"].argmax(0)
                            plot_seg_image(seg_image, data_mapping, fig=0)
                    network_gui.send(net_image_bytes, dataset.source_path)

                except Exception as e:
                    network_gui.conn = None

def streaming_gt(dataset, opt, pipe, checkpoint):
    gaussians = DeepGaussianModel(dataset.sh_degree, dataset.n_latents, dataset.n_classes, dataset.pixel_embedding, dataset.pos_embedding,dataset.rot_embedding)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    scene = Scene(dataset, gaussians, load_iteration=30000)

    if dataset.n_classes>0:
        all_data_mapping, _ = loadSemanticClasses(n = dataset.n_classes)
        plt.figure(0)
        fig, ax = plt.subplots(figsize=(9,6))
    with torch.no_grad():
        while True:     
            train_cameras = scene.getTrainCameras().copy()
            for cam in train_cameras:

                net_image = torch.nn.functional.interpolate(cam.original_image.cuda().unsqueeze(0), (800,800)).squeeze(0)

                if dataset.n_classes>0:
                    plot_seg_image(cam.original_semantic.cuda().squeeze(0), all_data_mapping, fig=0)
                plt.figure(1)
                plt.imshow(net_image.permute(1,2,0).cpu().numpy())
                plt.waitforbuttonpress()




if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # Set up command line argument parser
    parser = ArgumentParser(description="Viewer script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    
    print("Streaming " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    streaming(lp.extract(args), op.extract(args), pp.extract(args), args.start_checkpoint)

    # All done
    print("\Streaming complete.")
