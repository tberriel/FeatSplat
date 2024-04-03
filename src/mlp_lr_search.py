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

mipnerf360_outdoor_scenes = ["bycicle"]
deep_blending_scenes = []#["playroom", "drjohnson"]
tanks_and_temples_scenes = ["train"]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval/lr_search/")
parser.add_argument("--n_classes", default=0, type=int)
parser.add_argument("--iterations", default=1, type=int)
parser.add_argument("--sh_degree", default=0, type=int)
parser.add_argument("--pembedding", action="store_true")
args, _ = parser.parse_known_args()
lr_values = [0.001,0.0005,0.0001,0.00001]
iterations = 2
all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
all_scenes.extend(deep_blending_scenes)
all_scenes.extend(tanks_and_temples_scenes)

if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--mipnerf360', "-m360", required=True, type=str)
    parser.add_argument("--tanksandtemples", "-tat", required=True, type=str)
    parser.add_argument("--deepblending", "-db", required=True, type=str)
    args = parser.parse_args()

if not args.skip_training:

    for i in range(args.iterations):
        for lr in lr_values:
            common_args = f" --quiet --eval --test_iterations -1 --n_classes {args.n_classes} --mlp_lr {lr} --sh_degree {args.sh_degree}"
            if args.pembedding:
                common_args += " --pixel_embedding"
            for scene in mipnerf360_outdoor_scenes:
                source = args.mipnerf360 + "/" + scene
                os.system("python src/train.py -s " + source + " -i images_4 -m " + args.output_path + "/" + scene+f"_sem{args.n_classes}_lr{lr}_{i}" + common_args)
            for scene in deep_blending_scenes:
                source = args.deepblending + "/" + scene
                os.system("python src/train.py -s " + source + " -m " + args.output_path + "/" + scene+f"_sem{args.n_classes}_lr{lr}_{i}" + common_args)
            for scene in tanks_and_temples_scenes:
                source = args.tanksandtemples + "/" + scene
                os.system("python src/train.py -s " + source + " -m " + args.output_path + "/" + scene+f"_sem{args.n_classes}_lr{lr}_{i}" + common_args)

if not args.skip_rendering:
    all_sources = []
    for scene in mipnerf360_outdoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in deep_blending_scenes:
        all_sources.append(args.deepblending + "/" + scene)
    for scene in tanks_and_temples_scenes:
        all_sources.append(args.tanksandtemples + "/" + scene)
    for lr in lr_values:
        for i in range(args.iterations):
            common_args = f" --quiet --eval --skip_train --n_classes {args.n_classes} --sh_degree {args.sh_degree}"
            if args.pembedding:
                common_args += " --pixel_embedding"
            for scene, source in zip(all_scenes, all_sources):
                os.system("python src/render.py --iteration 7000 -s " + source + " -m " + args.output_path + "/" + scene+f"_sem{args.n_classes}_lr{lr}_{i}" + common_args)
                os.system("python src/render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene+f"_sem{args.n_classes}_lr{lr}_{i}" + common_args)

if not args.skip_metrics:
    scenes_string = ""
    for lr in lr_values:
        for i in range(args.iterations):
            for scene in all_scenes:
                scenes_string += "\"" + args.output_path + "/" + scene+f"_sem{args.n_classes}_lr{lr}_{i}" + "\" "

    os.system("python src/metrics.py -m " + scenes_string)
