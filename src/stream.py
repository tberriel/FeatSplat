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
from utils.loss_utils import l1_loss, ssim
from deep_gaussian_renderer import render, network_gui
import sys
from scene import Scene
from modules.gaussian_splatting.utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from modules.gaussian_splatting.utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from modules.gaussian_splatting.arguments import PipelineParams, OptimizationParams
from deep_gaussian_model import DeepGaussianModel
from arguments import ModelParams
from segmentation import mapClassesToRGB, loadClassesMapping
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import matplotlib.pyplot as plt
# Function to plot the image
def plot_image(image):
  plt.clf()
  plt.imshow(image)
  plt.axis('off')
  plt.draw()
  plt.pause(0.01)

def streaming(dataset, opt, pipe, checkpoint):
    gaussians = DeepGaussianModel(dataset.sh_degree, dataset.n_latents, dataset.n_classes)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    scene = Scene(dataset, gaussians, load_iteration=30000)

    bg_color = [0 for _ in range(gaussians.n_latents)] # Let's start with black background, ideally, background light could also be learnt as a latent vector
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if dataset.n_classes>0:
        data_mapping = loadClassesMapping()
        fig, ax = plt.subplots()
    with torch.no_grad():
        while True:     
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        out =  render(custom_cam, gaussians, pipe, background, scaling_modifer, override_color=gaussians.get_features)
                        net_image = out["render"]
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                        if dataset.n_classes>0:
                            seg_image = out["segmentation"].argmax(0)
                            rgb_seg_image = mapClassesToRGB(seg_image, data_mapping)
                            plot_image(rgb_seg_image)
                    network_gui.send(net_image_bytes, dataset.source_path)

                except Exception as e:
                    network_gui.conn = None


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
