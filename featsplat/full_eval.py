#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
#
# This file includes derivative from the original gaussian-splatting software
# 

import os
from argparse import ArgumentParser
from computation_metrics import computation_metrics

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--skip_comp_metrics", action="store_true")
parser.add_argument("--test_midway", action="store_true")
parser.add_argument("--output_path", default="./data/working/eval")
parser.add_argument("--train_iterations", default=30_000, type=int)
parser.add_argument("--h_layers", default=0, type=int)
parser.add_argument("--n_neurons", default=64, type=int)
parser.add_argument("--n_classes", default=0, type=int)
parser.add_argument("--lambda_sem", default=0.001, type=float)
parser.add_argument("--pembedding", action="store_true")
parser.add_argument("--cam_pos", action="store_true")
parser.add_argument("--cam_rot", action="store_true")
parser.add_argument("--gs", action="store_true")
parser.add_argument("--data_device", default="cuda")

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
parser.add_argument('--scannetpp_scenes', nargs="+", type=str, default=['0a5c013435', 'f07340dfea',  '7bc286c1b6', 'd2f44bf242',  '85251de7d1', '0e75f3c4d9', '98fe276aa8', '7e7cd69a59', 'f3685d06a9', '21d970d8de', '8b5caf3398', 'ada5304e41', '4c5c60fa76', 'ebc200e928', 'a5114ca13d', '5942004064', '1ada7a0617','f6659a3107', '1a130d092a', '80ffca8a48',   '08bbbdcc3d'])

args = parser.parse_args()
if args.gs:
    print("Training Gaussian splatting model. The following arguments will be ignored: --n_classes, --lambda_sem, --pos_embedding, --rot_embedding, --pixel_embedding, --h_layers, --n_neurons, ")
if not args.skip_training:
    common_args = f" --quiet --eval --iterations {args.train_iterations} --sh_degree {args.sh_degree} --data_device {args.data_device}"
    
    if not args.test_midway:
        common_args += " --test_iterations -1 "

    if args.gs:
        common_args += " --gaussian_splatting "
    else:
        common_args += f" --n_classes {args.n_classes} --lambda_sem {args.lambda_sem} --h_layers {args.h_layers} --n_neurons {args.n_neurons}"
        if args.pembedding:
            common_args += " --pixel_embedding "
        if args.cam_pos:
            common_args += " --pos_embedding"
        if args.cam_rot:
            common_args += " --rot_embedding"

    for scene in args.mipnerf360_outdoor_scenes:
        source = args.mipnerf360+ "/" + scene
        os.system("python featsplat/train.py -s " + source + " -i images_4 -m " + os.path.join(args.output_path,"360_v2",scene)+ common_args)
    for scene in args.mipnerf360_indoor_scenes:
        source = args.mipnerf360 + "/" + scene
        os.system("python featsplat/train.py -s " + source + " -i images_2 -m " + os.path.join(args.output_path,"360_v2",scene)+ common_args)
    for scene in args.tanks_and_temples_scenes:
        source = args.tanksandtemples + "/" + scene
        os.system("python featsplat/train.py -s " + source + " -m " + os.path.join(args.output_path,"tandt",scene)+ common_args)
    for scene in args.deep_blending_scenes:
        source = args.deepblending + "/" + scene
        os.system("python featsplat/train.py -s " + source + " -m " + os.path.join(args.output_path,"db",scene)+ common_args)
    for scene in args.scannetpp_scenes:
        source = args.scannetpp + "/" + scene
        os.system("python featsplat/train.py -s " + source + " -r 1 -m " +  os.path.join(args.output_path,"scannetpp",scene) + common_args + " --images_extension .JPG ")

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

    common_args = f" --quiet --eval --skip_train --sh_degree {args.sh_degree} --data_device {args.data_device}"

    if args.gs:
        common_args += " --gaussian_splatting "
    else:
        common_args += f" --n_classes {args.n_classes} --h_layers {args.h_layers} --n_neurons {args.n_neurons}"
        if args.pembedding:
            common_args += " --pixel_embedding "
        if args.cam_pos:
            common_args += " --pos_embedding"
        if args.cam_rot:
            common_args += " --rot_embedding"

    for scene, source in zip(almost_all_scenes, all_sources):
        if scene in  args.mipnerf360_indoor_scenes or scene in args.mipnerf360_outdoor_scenes:
            dataset = "360_v2"
        elif scene in  args.tanks_and_temples_scenes:
            dataset = "tandt"
        elif scene in args.deep_blending_scenes:
            dataset = "db"
        os.system("python featsplat/render.py --iteration 7000 -s " + source + " -m " + os.path.join(args.output_path,dataset,scene) + common_args)
        os.system("python featsplat/render.py --iteration 30000 -s " + source + " -m " + os.path.join(args.output_path,dataset,scene) + common_args)

    for scene in args.scannetpp_scenes:
        os.system("python featsplat/render.py --iteration 7000 -s " + args.scannetpp + "/" + scene + " -m " + os.path.join(args.output_path,"scannetpp",scene) + " -r 1 "+ common_args)
        os.system("python featsplat/render.py --iteration 30000 -s " + args.scannetpp + "/" + scene + " -m " + os.path.join(args.output_path,"scannetpp",scene) + " -r 1 "+ common_args + " --images_extension .JPG ")


if not args.skip_metrics:
    scenes_string = ""
    for scene in almost_all_scenes:
        if scene in  args.mipnerf360_indoor_scenes or scene in args.mipnerf360_outdoor_scenes:
            dataset = "360_v2"
        elif scene in  args.tanks_and_temples_scenes:
            dataset = "tandt"
        elif scene in args.deep_blending_scenes:
            dataset = "db"
        scenes_string += "\"" + os.path.join(args.output_path,dataset,scene) + "\" "
    for scene in args.scannetpp_scenes:
        scenes_string += "\"" + os.path.join(args.output_path,"scannetpp",scene) + "\" "

    os.system("python featsplat/metrics.py -m " + scenes_string)

if not args.skip_comp_metrics:
    failed_scenes = []
    all_sources = []
    for scene in args.mipnerf360_outdoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in args.mipnerf360_indoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in args.tanks_and_temples_scenes:
        all_sources.append(args.tanksandtemples + "/" + scene)
    for scene in args.deep_blending_scenes:
        all_sources.append(args.deepblending + "/" + scene)

    common_args = ["--eval","--n_classes", str(args.n_classes), "--sh_degree", str(args.sh_degree), "--data_device",args.data_device, "--h_layers", str(args.h_layers)]
    if args.pembedding:
        common_args += ["--pixel_embedding"]
    if args.cam_pos:
        common_args += ["--pos_embedding"]
    if args.cam_rot:
        common_args += ["--rot_embedding"]
    if args.gs:
        common_args += ["--gaussian_splatting"]
        
    for scene, source in zip(almost_all_scenes, all_sources):
        if scene in  args.mipnerf360_indoor_scenes or scene in args.mipnerf360_outdoor_scenes:
            dataset = "360_v2"
        elif scene in  args.tanks_and_temples_scenes:
            dataset = "tandt"
        elif scene in args.deep_blending_scenes:
            dataset = "db"
        try:
            computation_metrics(common_args + ["-s", source, "-m", os.path.join(args.output_path,dataset,scene)])
        except:
            failed_scenes.append(scene)

    for scene in args.scannetpp_scenes:
        try:
            computation_metrics(common_args + ["-s",args.scannetpp + "/" + scene, "-m",os.path.join(args.output_path,"scannetpp",scene), "--images_extension", ".JPG"])

        except:
            failed_scenes.append(scene)
    print(f"Failed on scenes: {failed_scenes}")