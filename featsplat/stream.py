
import os
import torch
from feat_gaussian_renderer import render, network_gui
import sys
from utils.general_utils import safe_state
from argparse import ArgumentParser, Namespace
from arguments import PipelineParams, OptimizationParams, get_combined_args
from scene.feat_gaussian_model import FeatGaussianModel
from arguments import ModelParams


def streaming(dataset, opt, pipe):
    gaussians = FeatGaussianModel(dataset)
    gaussians.load_ply(os.path.join(dataset.model_path, "point_cloud", "iteration_" + str(30000),"point_cloud.ply"))

    bg_color = [0 for _ in range(gaussians.n_latents)] 
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print("Starting stream ...")
    with torch.no_grad():
        while True:     
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        out =  render(custom_cam, gaussians, features_splatting=True)
                        net_image = out["render"]
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                        
                    network_gui.send(net_image_bytes, dataset.source_path)

                except Exception as e:
                    network_gui.conn = None


if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # Set up command line argument parser
    parser = ArgumentParser(description="Viewer script parameters")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    args = get_combined_args(parser)
    
    print("Streaming " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    streaming(lp.extract(args), op.extract(args), pp.extract(args))

    # All done
    print("\Streaming complete.")