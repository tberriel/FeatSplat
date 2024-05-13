import torch
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
import json
from utils.general_utils import safe_state
from deep_gaussian_renderer import render
from deep_gaussian_model import DeepGaussianModel, GaussianModel
from scene import Scene
from plyfile import PlyData
import os


def render_set(repetitions, views, gaussians, pipeline, background, gaussian_splatting):
    fps = 0.
    for i in range(repetitions):
        torch.cuda.synchronize()
        p_bar = tqdm(views, desc="Rendering progress")
        for idx, view in enumerate(p_bar):
            if gaussian_splatting:
                rendering = render(view, gaussians, pipeline, background)["render"]
            else: 
                rendering = render(view, gaussians, pipeline, background, override_color=gaussians.get_features, features_splatting=True)["render"]
        torch.cuda.synchronize()
        t_end = p_bar.format_dict["elapsed"]
        if i >0: # Use first repetition as warm-up
            fps += p_bar.format_dict["total"]/t_end

    return fps/(repetitions-1)

def compute_fps(dataset : ModelParams, pipeline : PipelineParams, gaussian_splatting : bool, repetitions : int = 1, data_set: str ="train"):
    with torch.no_grad():
        if gaussian_splatting:
            assert dataset.n_classes == 0, "Gaussian Splatting does not predict semantics. Set n_classes to 0."
            gaussians = GaussianModel(dataset.sh_degree)
        else:
            gaussians = DeepGaussianModel(dataset.sh_degree, dataset.n_latents, dataset.n_classes, dataset.pixel_embedding, dataset.pos_embedding, dataset.rot_embedding)
        scene = Scene(dataset, gaussians, load_iteration=30000, shuffle=False)
        if gaussian_splatting:
            bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        else:
            bg_color = [0 for _ in range(gaussians.n_latents)] #if dataset.white_background else [1 for _ in range(gaussians.n_latents)]# Let's start with black background, ideally, background light could also be learnt as a latent vector
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        return render_set(repetitions, scene.getTrainCameras(), gaussians, pipeline, background, gaussian_splatting)

    
def compute_size(model_path):
    folder_path = os.path.join(model_path, 'point_cloud', 'iteration_30000')
    size_bytes = sum(os.path.getsize(os.path.join(folder_path, file)) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file)))
    size_mbytes = size_bytes / (1024 * 1024)
    return size_mbytes

def compute_n_gaussians(model_path):
    path = os.path.join(model_path, 'point_cloud', 'iteration_30000', 'point_cloud.ply')

    plydata = PlyData.read(path)
    
    return plydata.elements[0]["x"].shape[0]

def computation_metrics(common_args, fps_repetitions=2):
    parser = ArgumentParser(description="Full evaluation script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument('--gaussian_splatting', action="store_true", default=False)
    args = get_combined_args(parser, common_args)
    
    # Initialize system state (RNG)
    safe_state(True)

    fps = compute_fps(model.extract(args), pipeline.extract(args), args.gaussian_splatting, repetitions=fps_repetitions, data_set="train")

    size = compute_size(args.model_path)
    n_gaussians = compute_n_gaussians(args.model_path)
    # Read the JSON file
    with open(args.model_path + "/results.json", 'r') as fp:
        full_dict = json.load(fp)

    # Add the fields fps, size, and n_gaussians
    full_dict["ours_30000"]['fps'] = fps
    full_dict["ours_30000"]['size'] = size
    full_dict["ours_30000"]['n_gaussians'] = n_gaussians

    # Write the updated dictionary back to the JSON file
    with open(args.model_path + "/results.json", 'w') as fp:
        json.dump(full_dict, fp, indent=True)
