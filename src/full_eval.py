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
from computation_metrics import computation_metrics

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--skip_comp_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval")
parser.add_argument("--n_classes", default=0, type=int)
parser.add_argument("--lambda_sem", default=0.001, type=float)
parser.add_argument("--pembedding", action="store_true")
parser.add_argument("--cam_pos", action="store_true")
parser.add_argument("--cam_rot", action="store_true")
parser.add_argument("--iterations", default=4, type=int)
parser.add_argument("--gs", action="store_true")

parser.add_argument('--mipnerf360', "-m360", type=str)
parser.add_argument("--tanksandtemples", "-tat",type=str)
parser.add_argument("--deepblending", "-db", type=str)
parser.add_argument("--scannetpp", "-s", type=str)
args, _ = parser.parse_known_args()

parser.add_argument("--sh_degree", default=0 if not args.gs else 3, type=int)
parser.add_argument('--mipnerf360_outdoor_scenes', nargs="+", type=str, default=["bicycle", "flowers", "garden", "stump", "treehill"] if args.mipnerf360 is not None else [] )
parser.add_argument('--mipnerf360_indoor_scenes', nargs="+", type=str, default=["room", "counter", "kitchen", "bonsai"] if args.mipnerf360 is not None else [] )
parser.add_argument('--tanks_and_temples_scenes', nargs="+", type=str, default=["truck", "train"] if args.tanksandtemples is not None else [] )
parser.add_argument('--deep_blending_scenes', nargs="+", type=str, default=["drjohnson", "playroom"] if args.deepblending is not None else [] )
parser.add_argument('--scannetpp_scenes', nargs="+", type=str, default=['0a5c013435', 'f07340dfea',  '7bc286c1b6', 'd2f44bf242',  '85251de7d1', '0e75f3c4d9', '98fe276aa8', '7e7cd69a59', 'f3685d06a9', '21d970d8de', '8b5caf3398', 'ada5304e41', '4c5c60fa76', 'ebc200e928', 'a5114ca13d', '5942004064'] if args.scannetpp is not None else [] )#['108ec0b806','bb87c292ad', 'a08d9a2476'])

args = parser.parse_args()

if args.n_classes > 0:
    name_suffix = f"_sem{args.n_classes}_{args.lambda_sem}"
    assert not args.gs, "Gaussian Splatting does not predict semantics. Set n_classes to 0."
else:
    name_suffix = ""

if not args.skip_training:
    common_args = f" --quiet --eval --test_iterations -1 --n_classes {args.n_classes} --sh_degree {args.sh_degree}"
    if args.pembedding:
        common_args += " --pixel_embedding "
    if args.cam_pos:
        common_args += " --pos_embedding"
    if args.cam_rot:
        common_args += " --rot_embedding"
    if args.gs:
        common_args += "--gaussian_splatting "
    for scene in args.mipnerf360_outdoor_scenes:
        source = args.mipnerf360+ "/" + scene
        os.system("python src/train.py -s " + source + " -i images_4 -m " + args.output_path + "/" + scene+ common_args)
    for scene in args.mipnerf360_indoor_scenes:
        source = args.mipnerf360 + "/" + scene
        os.system("python src/train.py -s " + source + " -i images_2 -m " + args.output_path + "/" + scene+ common_args)
    for scene in args.tanks_and_temples_scenes:
        source = args.tanksandtemples + "/" + scene
        os.system("python src/train.py -s " + source + " -m " + args.output_path + "/" + scene+ common_args)
    for scene in args.deep_blending_scenes:
        source = args.deepblending + "/" + scene
        os.system("python src/train.py -s " + source + " -m " + args.output_path + "/" + scene+ common_args)
    for scene in args.scannetpp_scenes:
        for i in range(args.iterations):
            source = args.scannetpp + "/" + scene
            os.system("python src/train.py -s " + source + " -m " + args.output_path + "/" + scene+name_suffix+f"_i" + common_args)

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
    if args.cam_pos:
        common_args += " --pos_embedding"
    if args.cam_rot:
        common_args += " --rot_embedding"
    if args.gs:
        common_args += "--gaussian_splatting "
    for scene, source in zip(almost_all_scenes, all_sources):
        os.system("python src/render.py --iteration 7000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)
        os.system("python src/render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)
    for scene in args.scannetpp_scenes:
        for i in range(args.iterations):
            os.system("python src/render.py --iteration 30000 -s " + args.scannetpp + "/" + scene + " -m " + args.output_path + "/scannet/" + scene+name_suffix+f"_{i}" + common_args)


if not args.skip_metrics:
    scenes_string = ""
    for scene in almost_all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene+f"_sem{args.n_classes}" + "\" "
    for scene in args.scannetpp_scenes:
        for i in range(args.iterations):
            scenes_string += "\"" + args.output_path + "/scannet/" + scene+name_suffix+f"_{i}" + "\" "

    os.system("python src/metrics.py -m " + scenes_string)

if not args.skip_comp_metrics:
    all_sources = []
    for scene in args.mipnerf360_outdoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in args.mipnerf360_indoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in args.tanks_and_temples_scenes:
        all_sources.append(args.tanksandtemples + "/" + scene)
    for scene in args.deep_blending_scenes:
        all_sources.append(args.deepblending + "/" + scene)

    common_args = ["--eval","--n_classes", f"{args.n_classes}", "--sh_degree", f"{args.sh_degree}"]
    if args.pembedding:
        common_args += ["--pixel_embedding"]
    if args.cam_pos:
        common_args += ["--pos_embedding"]
    if args.cam_rot:
        common_args += ["--rot_embedding"]
    if args.gs:
        common_args += ["--gaussian_splatting"]
        
    for scene, source in zip(almost_all_scenes, all_sources):
        computation_metrics(common_args + ["-s", source, "-m", args.output_path + "/" + scene])

    for scene in args.scannetpp_scenes:
        computation_metrics(common_args + ["-s",args.scannetpp + "/" + scene, "-m",args.output_path + "/scannet/" + scene+name_suffix+f"_{0}"])
