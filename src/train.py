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
from random import randint
from deep_gaussian_renderer import render, network_gui
from utils.general_utils import safe_state
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from deep_gaussian_model import DeepGaussianModel, GaussianModel
import sys
import uuid
from utils.seg_utils import mapClassesToRGB, loadSemanticClasses
from utils.loss_utils import l1_loss, ssim
from torchmetrics.classification import MulticlassJaccardIndex
from utils.image_utils import psnr
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import matplotlib.pyplot as plt
# Function to plot the image
def plot_seg_image(seg_image, data_mapping):
  
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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, stream, gaussian_splatting):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    if gaussian_splatting:
        assert dataset.n_classes == 0, "Gaussian Splatting does not predict semantics. Set n_classes to 0."
        gaussians = GaussianModel(dataset.sh_degree)
    else:
        gaussians = DeepGaussianModel(0, dataset.n_latents, dataset.n_classes, dataset.pixel_embedding, dataset.pos_embedding,dataset.rot_embedding, dataset.h_layers)

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    if gaussian_splatting:
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    else:
        bg_color = [0 for _ in range(gaussians.n_latents)] # Let's start with black background, ideally, background light could also be learnt as a latent vector
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    ce_loss = None
    mIoU = None 
    if dataset.n_classes>0:
        data_mapping, weights = loadSemanticClasses(n = dataset.n_classes)
        #fig, ax = plt.subplots()
        ce_loss = torch.nn.CrossEntropyLoss(weight=weights.cuda() if dataset.weighted_ce_loss else None)# In dataloading set 31 as no class, that number shouldn't be ignored as to give a way to the network to label thinks it does not recognizes
        mIoU = MulticlassJaccardIndex(dataset.n_classes).cuda()
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if stream:        
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
                            plot_seg_image(seg_image, data_mapping)
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if gaussian_splatting and iteration % 1000 == 0 :
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        if gaussian_splatting:
            bg = torch.rand((3), device="cuda") if opt.random_background else background

            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        else:
            bg = torch.rand((gaussians.n_latents), device="cuda") if opt.random_background else background

            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, override_color=gaussians.get_features, features_splatting=True)

        image, segmentation, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["segmentation"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        if segmentation is not None:
            gt_segmentation = viewpoint_cam.original_semantic.cuda()
            Lce = ce_loss(segmentation.permute(1,2,0).flatten(0,1), gt_segmentation.flatten().long())
        else: 
            Lce = 0.
        
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + Lce*opt.lambda_sem
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, ce_loss, mIoU, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), gaussian_splatting)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, ce_loss, mIoU, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, gaussian_splatting):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ce_test = 0.0
                #mIoU_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    out = renderFunc(viewpoint, scene.gaussians, *renderArgs,override_color=scene.gaussians.get_features,features_splatting=not gaussian_splatting)
                    image = torch.clamp(out["render"], 0.0, 1.0)
                    segmentation = out["segmentation"]
                    
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if segmentation is not None:
                        gt_segmentation = viewpoint.original_semantic.cuda()

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    if segmentation is not None:
                        ce_test += ce_loss(segmentation.permute(1,2,0).flatten(0,1), gt_segmentation.flatten().long())
                    #mIoU_test += mIoU(segmentation.permute(1,2,0).flatten(0,1), gt_segmentation.flatten().long())
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                if segmentation is not None: 
                    ce_test /= len(config['cameras']) 
                #mIoU_test /= len(config['cameras'])

                print("\n[ITER {}] Evaluating {}: L1 {:.3f} PSNR {:.3f} CE {:.3f} mIOU {:.3f}".format(iteration, config['name'], l1_test, psnr_test, ce_test, 0.0))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    if segmentation is not None:
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ce_loss', ce_test, iteration)
                    #tb_writer.add_scalar(config['name'] + '/loss_viewpoint - mIoU', mIoU_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--stream', action="store_true", default=False)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 21000, 30_000])
    parser.add_argument("--test_str", nargs="+", type=str, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 21000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--gaussian_splatting', action="store_true", default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if args.stream:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.stream, args.gaussian_splatting)

    # All done
    print("\nTraining complete.")
