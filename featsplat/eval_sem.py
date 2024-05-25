#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
#

import os
import torch
from torchmetrics.classification import MulticlassJaccardIndex
from feat_gaussian_renderer import render
import sys
from scene import Scene
from utils.general_utils import safe_state
from argparse import ArgumentParser, Namespace
from scene.feat_gaussian_model import FeatGaussianModel
from arguments import ModelParams, PipelineParams
from tqdm import tqdm
def _safe_divide(num, denom):
    """Safe division, by preventing division by zero.

    Additionally casts to float if input is not already to secure backwards compatibility.

    """
    denom[denom == 0.0] = 1
    num = num if num.is_floating_point() else num.float()
    denom = denom if denom.is_floating_point() else denom.float()
    return num / denom

def jaccard_index_reduce(confmat, average=None, ignore_index=None):
    """ From https://github.com/Lightning-AI/torchmetrics/blob/v1.3.2/src/torchmetrics/functional/classification/jaccard.py
    Perform reduction of an un-normalized confusion matrix into jaccard score.

    Args:
        confmat: tensor with un-normalized confusionmatrix
        average: reduction method

            - ``'binary'``: binary reduction, expects a 2x2 matrix
            - ``'macro'``: Calculate the metric for each class separately, and average the
              metrics across classes (with equal weights for each class).
            - ``'micro'``: Calculate the metric globally, across all samples and classes.
            - ``'weighted'``: Calculate the metric for each class separately, and average the
              metrics across classes, weighting each class by its support (``tp + fn``).
            - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
              the metric for every class.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation

    """
    allowed_average = ["binary", "micro", "macro", "weighted", "none", None]
    if average not in allowed_average:
        raise ValueError(f"The `average` has to be one of {allowed_average}, got {average}.")
    confmat = confmat.float()
    if average == "binary":
        return confmat[1, 1] / (confmat[0, 1] + confmat[1, 0] + confmat[1, 1])

    ignore_index_cond = ignore_index is not None and 0 <= ignore_index < confmat.shape[0]
    multilabel = confmat.ndim == 3
    if multilabel:
        num = confmat[:, 1, 1]
        denom = confmat[:, 1, 1] + confmat[:, 0, 1] + confmat[:, 1, 0]
    else:  # multiclass
        num = torch.diag(confmat)
        denom = confmat.sum(0) + confmat.sum(1) - num

    if average == "micro":
        num = num.sum()
        denom = denom.sum() - (denom[ignore_index] if ignore_index_cond else 0.0)

    jaccard = _safe_divide(num, denom)

    if average is None or average == "none" or average == "micro":
        return jaccard
    if average == "weighted":
        weights = confmat[:, 1, 1] + confmat[:, 1, 0] if confmat.ndim == 3 else confmat.sum(1)
    else:
        weights = torch.ones_like(jaccard)
        if ignore_index_cond:
            weights[ignore_index] = 0.0
        if not multilabel:
            weights[confmat.sum(1) + confmat.sum(0) == 0] = 0.0
    return ((weights * jaccard) / weights.sum()).sum()

def avgmIoU(data_path, eval_path, n_classes):
    confMat = torch.zeros((n_classes,n_classes), dtype=torch.long, device="cuda")
    scene_list = tqdm(os.listdir(data_path))
    torch.set_printoptions(4)
    for scene in scene_list:
        model_path = os.path.join(eval_path, scene+"_sem64_0.001_0")
        if os.path.isdir(model_path):
            confMat_tmp = torch.load(os.path.join(model_path, "confMat.pt"))
            print("Scene {} mIoU w: {}".format(scene,jaccard_index_reduce(confMat_tmp, "weighted").item()))
            confMat += confMat_tmp

    mIoU_macro = jaccard_index_reduce(confMat, "macro", None)
    mIoU_micro = jaccard_index_reduce(confMat, "micro", None)
    mIoU_weighted = jaccard_index_reduce(confMat, "weighted", None)
    print("mIoU macro {:3f}; micro{:3f},;weighted {:3f}".format(mIoU_macro, mIoU_micro, mIoU_weighted))

def computeConfMat(dataset, pipe, data_path, eval_path):
    bg_color = [0 for _ in range(dataset.n_latents)] # Let's start with black background, ideally, background light could also be learnt as a latent vector
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    gaussians = FeatGaussianModel(dataset)
    scene_list = os.listdir(data_path)
    scenes = []
    for scene in scene_list:
        scene_path = os.path.join(eval_path, scene+"_sem64_0.001_0")
        if os.path.isdir(scene_path):# and not os.path.exists(os.path.join(scene_path,"confMat.pt")):
            scenes.append(scene)
    scene_list = tqdm(scenes)
    for scene in scene_list:
        dataset.source_path = os.path.join(data_path, scene)
        dataset.model_path = os.path.join(eval_path, scene+"_sem64_0.001_0")
        scene = Scene(dataset, gaussians, load_iteration=30000, load_train=False)  

        confMatPerScene(scene, render, (pipe, background), dataset.n_classes, dataset.model_path)


def confMatPerScene( scene : Scene, renderFunc, renderArgs, n_classes : int, model_path : str):
    mIoU = MulticlassJaccardIndex(n_classes, average=None).cuda()

    # Report test and samples of evaluating set
    cameras = scene.getTestCameras()
    confmat_path = os.path.join(model_path,"confMat.pt")
    for idx, viewpoint in enumerate(cameras):
        out = renderFunc(viewpoint, scene.gaussians, *renderArgs,override_color=scene.gaussians.get_features, features_splatting=True)
        
        segmentation = out["segmentation"].permute(1,2,0).flatten(0,1)
        gt_segmentation = viewpoint.original_semantic.cuda().flatten().long()

        mIoU.update(segmentation, gt_segmentation)

    torch.save(mIoU.confmat,confmat_path)


if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # Set up command line argument parser
    parser = ArgumentParser(description="evaluating script parameters")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--confmat", action="store_true")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--eval_path", type=str, required=True)
    args = parser.parse_args(sys.argv[1:])
    
    print("Evaluating " + args.model_path)

    # Initialize system state (RNG)

    with torch.no_grad():
        lp_args = lp.extract(args)
        if args.confmat:
            safe_state(True)
            computeConfMat(lp_args, pp.extract(args), args.data_path, args.eval_path)
        avgmIoU( args.data_path, args.eval_path, lp_args.n_classes)

    # All done
    print("\nEvaluation complete.")
