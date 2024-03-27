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
from argparse import ArgumentParser

mipnerf360_outdoor_scenes = []
deep_blending_scenes = ["playroom", "drjohnson"]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval/lr_search/")
parser.add_argument("--n_classes", default=0, type=int)
args, _ = parser.parse_known_args()
lr_values = [0.001,0.0005,0.0001,0.00001]
iterations = 2
all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
all_scenes.extend(deep_blending_scenes)

if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--mipnerf360', "-m360", required=True, type=str)
    parser.add_argument("--deepblending", "-db", required=True, type=str)
    args = parser.parse_args()

if not args.skip_training:
    for lr in lr_values:
        for i in range(iterations):
            common_args = f" --quiet --eval --test_iterations -1 --n_classes {args.n_classes} --mlp_lr {lr}"
    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360 + "/" + scene
        os.system("python src/train.py -s " + source + " -i images_4 -m " + args.output_path + "/" + scene+f"_sem{args.n_classes}_lr{lr}_{i}" + common_args)
    for scene in deep_blending_scenes:
        source = args.deepblending + "/" + scene
        os.system("python src/train.py -s " + source + " -m " + args.output_path + "/" + scene+f"_sem{args.n_classes}_lr{lr}_{i}" + common_args)

if not args.skip_rendering:
    all_sources = []
    for scene in mipnerf360_outdoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in deep_blending_scenes:
        all_sources.append(args.deepblending + "/" + scene)
    for lr in lr_values:
        for i in range(iterations):
            common_args = " --quiet --eval --skip_train"
            for scene, source in zip(all_scenes, all_sources):
                os.system("python src/render.py --iteration 7000 -s " + source + " -m " + args.output_path + "/" + scene+f"_sem{args.n_classes}_lr{lr}_{i}" + common_args)
                os.system("python src/render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene+f"_sem{args.n_classes}_lr{lr}_{i}" + common_args)

if not args.skip_metrics:
    scenes_string = ""
    for lr in lr_values:
        for i in range(iterations):
            for scene in all_scenes:
                scenes_string += "\"" + args.output_path + "/" + scene+f"_sem{args.n_classes}_lr{lr}_{i}" + "\" "

    os.system("python src/metrics.py -m " + scenes_string)
