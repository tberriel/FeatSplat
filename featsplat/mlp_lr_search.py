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
from argparse import ArgumentParser
import json

mipnerf360_outdoor_scenes =[]#["bicycle"]
deep_blending_scenes = []#["playroom", "drjohnson"]
tanks_and_temples_scenes = []#"train"
scannetpp_scenes = ["5d152fab1b" , "5656608266", "c0f5742640" ]#
parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval/lr_search/")
parser.add_argument("--n_classes", default=0, type=int)
parser.add_argument("--iterations", default=2, type=int)
parser.add_argument("--sh_degree", default=0, type=int)
parser.add_argument("--h_layers", default=0, type=int)
parser.add_argument("--pembedding", action="store_true")
parser.add_argument("--cam_pos", action="store_true")
parser.add_argument("--cam_rot", action="store_true")
args, _ = parser.parse_known_args()
lr_values = [0.001, 0.0001, 0.0005]
iterations = 2
all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
all_scenes.extend(deep_blending_scenes)
all_scenes.extend(tanks_and_temples_scenes)
all_scenes.extend(scannetpp_scenes)

if not args.skip_training or not args.skip_rendering:
    #parser.add_argument('--mipnerf360', "-m360", required=True, type=str)
    #parser.add_argument("--tanksandtemples", "-tat", required=True, type=str)
    #parser.add_argument("--deepblending", "-db", required=True, type=str)
    parser.add_argument("--scannetpp", required=True, type=str)
    args = parser.parse_args()

if not args.skip_training:

    for i in range(args.iterations):
        for lr in lr_values:
            common_args = f" --eval --n_classes {args.n_classes} --mlp_lr {lr} --sh_degree {args.sh_degree} --h_layers {args.h_layers}"
            if args.pembedding:
                common_args += " --pixel_embedding"
            if args.cam_pos:
                common_args += " --pos_embedding"
            if args.cam_rot:
                common_args += " --rot_embedding"
            for scene in mipnerf360_outdoor_scenes:
                source = args.mipnerf360 + "/" + scene
                os.system("python src/train.py -s " + source + " -i images_4 -m " + args.output_path + "/" + scene+f"_lr{lr}_{i}" + common_args)
            for scene in deep_blending_scenes:
                source = args.deepblending + "/" + scene
                os.system("python src/train.py -s " + source + " -m " + args.output_path + "/" + scene+f"_lr{lr}_{i}" + common_args)
            for scene in tanks_and_temples_scenes:
                source = args.tanksandtemples + "/" + scene
                os.system("python src/train.py -s " + source + " -m " + args.output_path + "/" + scene+f"_lr{lr}_{i}" + common_args)
            for scene in scannetpp_scenes:
                source = args.scannetpp + "/" + scene
                os.system("python src/train.py -s " + source + " -r 1 -m " +  os.path.join(args.output_path,scene+f"_lr{lr}_{i}") + common_args)

if not args.skip_rendering:
    all_sources = []
    for scene in mipnerf360_outdoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in deep_blending_scenes:
        all_sources.append(args.deepblending + "/" + scene)
    for scene in tanks_and_temples_scenes:
        all_sources.append(args.tanksandtemples + "/" + scene)
    for scene in scannetpp_scenes:
        all_sources.append(args.scannetpp + "/" + scene)
    for lr in lr_values:
        for i in range(args.iterations):
            common_args = f" --quiet --eval --skip_train --n_classes {args.n_classes} --sh_degree {args.sh_degree} --h_layers {args.h_layers}"
            if args.pembedding:
                common_args += " --pixel_embedding"
            if args.cam_pos:
                common_args += " --pos_embedding"
            if args.cam_rot:
                common_args += " --rot_embedding"
            for scene, source in zip(all_scenes, all_sources):
                os.system("python src/render.py --iteration 7000 -s " + source + " -m " + args.output_path + "/" + scene+f"_lr{lr}_{i}" + common_args)
                os.system("python src/render.py --iteration 14000 -s " + source + " -m " + args.output_path + "/" + scene+f"_lr{lr}_{i}" + common_args)
                os.system("python src/render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene+f"_lr{lr}_{i}" + common_args)

if not args.skip_metrics:
    scenes_string = ""
    for lr in lr_values:
        for i in range(args.iterations):
            for scene in all_scenes:
                scenes_string += "\"" + args.output_path + "/" + scene+f"_lr{lr}_{i}" + "\" "

    os.system("python src/metrics.py -m " + scenes_string)
for lr in lr_values:
    for scene in all_scenes:
        results = []
        for i in range(args.iterations):

            result_file = os.path.join(args.output_path, scene+f"_lr{lr}_{i}", "results.json")
            with open(result_file, "r") as f:
                data = json.load(f)
                ssim = data["ours_30000"]["SSIM"]
                lpips = data["ours_30000"]["LPIPS"]
                psnr = data["ours_30000"]["PSNR"]
                results.append((ssim, lpips, psnr))

        # Compute averages
        avg_ssim = sum([result[0] for result in results]) / len(results)
        avg_lpips = sum([result[1] for result in results]) / len(results)
        avg_psnr = sum([result[2] for result in results]) / len(results)
        print(f"{scene} with {lr} ")
        print("Average SSIM: {:.3f}; PSNR: {:.3f}; LPIPS: {:.3f};".format(avg_ssim, avg_psnr, avg_lpips))