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
import json
#mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
#mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
#tanks_and_temples_scenes = ["truck", "train"]
#deep_blending_scenes = ["drjohnson", "playroom"]

#scannetpp_scenes = ['0a5c013435', 'f07340dfea',  '7bc286c1b6', 'd2f44bf242',  '85251de7d1', '0e75f3c4d9', '98fe276aa8', '7e7cd69a59', 'f3685d06a9', '21d970d8de', '8b5caf3398', 'ada5304e41', '4c5c60fa76']#['a5114ca13d','108ec0b806','bb87c292ad', 'ebc200e928', 'a08d9a2476','5942004064',]#

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval")
parser.add_argument("--n_classes", default=0, type=int)
parser.add_argument("--sh_degree", default=0, type=int)
parser.add_argument("--pembedding", action="store_true")
parser.add_argument("--iterations", default=4, type=int)

parser.add_argument('--mipnerf360', "-m360", type=str)
parser.add_argument("--tanksandtemples", "-tat",type=str)
parser.add_argument("--deepblending", "-db", type=str)
parser.add_argument("--scannetpp", "-s", type=str)
args, _ = parser.parse_known_args()

parser.add_argument('--mipnerf360_outdoor_scenes', nargs="+", type=str, default=["bicycle", "flowers", "garden", "stump", "treehill"] if args.mipnerf360 is not None else [] )
parser.add_argument('--mipnerf360_indoor_scenes', nargs="+", type=str, default=["room", "counter", "kitchen", "bonsai"] if args.mipnerf360 is not None else [] )
parser.add_argument('--tanks_and_temples_scenes', nargs="+", type=str, default=["truck", "train"] if args.tanksandtemples is not None else [] )
parser.add_argument('--deep_blending_scenes', nargs="+", type=str, default=["drjohnson", "playroom"] if args.deepblending is not None else [] )
parser.add_argument('--scannetpp_scenes', nargs="+", type=str, default=['0a5c013435', 'f07340dfea',  '7bc286c1b6', 'd2f44bf242',  '85251de7d1', '0e75f3c4d9', '98fe276aa8', '7e7cd69a59', 'f3685d06a9', '21d970d8de', '8b5caf3398', 'ada5304e41', '4c5c60fa76'] if args.scannetpp is not None else [] )#['a5114ca13d','108ec0b806','bb87c292ad', 'ebc200e928', 'a08d9a2476','5942004064',])

args = parser.parse_args()


if not args.skip_training:
    common_args = f" --quiet --eval --test_iterations -1 --n_classes {args.n_classes} --sh_degree {args.sh_degree}"
    if args.pembedding:
        common_args += " --pixel_embedding "
    for scene in args.mipnerf360_outdoor_scenes:
        source = args.mipnerf360+ "/" + scene
        os.system("python src/train.py -s " + source + " -i images_4 -m " + args.output_path + "/" + scene+f"_sem{args.n_classes}" + common_args)
    for scene in args.mipnerf360_indoor_scenes:
        source = args.mipnerf360 + "/" + scene
        os.system("python src/train.py -s " + source + " -i images_2 -m " + args.output_path + "/" + scene+f"_sem{args.n_classes}" + common_args)
    for scene in args.tanks_and_temples_scenes:
        source = args.tanksandtemples + "/" + scene
        os.system("python src/train.py -s " + source + " -m " + args.output_path + "/" + scene+f"_sem{args.n_classes}" + common_args)
    for scene in args.deep_blending_scenes:
        source = args.deepblending + "/" + scene
        os.system("python src/train.py -s " + source + " -m " + args.output_path + "/" + scene+f"_sem{args.n_classes}" + common_args)
    for scene in args.scannetpp_scenes:
        for i in range(args.iterations):
            source = args.scannetpp + "/" + scene
            os.system("python src/train.py -s " + source + " -m " + args.output_path + "/" + scene+f"_sem{args.n_classes}_{i}" + common_args)

almost_all_scenes = []
almost_all_scenes.extend(args.mipnerf360_outdoor_scenes)
almost_all_scenes.extend(args.mipnerf360_indoor_scenes)
almost_all_scenes.extend(args.tanks_and_temples_scenes)
almost_all_scenes.extend(args.deep_blending_scenes)

assert len(almost_all_scenes)+len(args.scannetpp_scenes) > 0 , "At least one dataset should be selected"

if not args.skip_rendering:
    all_sources = []
    for scene in args.mipnerf360_outdoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in args.mipnerf360_indoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in args.tanks_and_temples_scenes:
        all_sources.append(args.tanksandtemples + "/" + scene)
    for scene in args.deep_blending_scenes:
        all_sources.append(args.deepblending + "/" + scene)

    common_args = f" --quiet --eval --skip_train --n_classes {args.n_classes} --sh_degree {args.sh_degree}"
    if args.pembedding:
        common_args += " --pixel_embedding "
    for scene, source in zip(almost_all_scenes, all_sources):
        os.system("python src/render.py --iteration 7000 -s " + source + " -m " + args.output_path + "/" + scene+f"_sem{args.n_classes}" + common_args)
        os.system("python src/render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene+f"_sem{args.n_classes}" + common_args)
    for scene in args.scannetpp_scenes:
        for i in range(args.iterations):
            #os.system("python src/render.py --iteration 7000 -s " + args.scannetpp + "/" + scene + " -m " + args.output_path + "/" + scene+f"_sem{args.n_classes}_{args.iterations}" + common_args)
            os.system("python src/render.py --iteration 30000 -s " + args.scannetpp + "/" + scene + " -m " + args.output_path + "/scannet/" + scene+f"_sem{args.n_classes}_{i}" + common_args)
    

if not args.skip_metrics:
    scenes_string = ""
    for scene in almost_all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene+f"_sem{args.n_classes}" + "\" "
    for scene in args.scannetpp_scenes:
        for i in range(args.iterations):
            scenes_string += "\"" + args.output_path + "/scannet/" + scene+f"_sem{args.n_classes}_{i}" + "\" "

    os.system("python src/metrics.py -m " + scenes_string)


for scene in almost_all_scenes:
    results = []

    result_file = os.path.join(args.output_path, scene+f"_sem{args.n_classes}", "results.json")
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
    print(f"{scene} ")
    print("Average SSIM: {:.3f}; PSNR: {:.3f}; LPIPS: {:.3f};".format(avg_ssim, avg_psnr, avg_lpips))

for scene in args.scannetpp_scenes:
    results = []
    for i in range(args.iterations):

        result_file = os.path.join(args.output_path, "scannet/", scene+f"_sem{args.n_classes}_{i}", "results.json")
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
    print(f"Scannet {scene} ")
    print("Average SSIM: {:.3f}; PSNR: {:.3f}; LPIPS: {:.3f};".format(avg_ssim, avg_psnr, avg_lpips))