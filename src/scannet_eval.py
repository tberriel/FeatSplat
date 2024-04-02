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
scannetpp_scenes =  ['a5114ca13d','108ec0b806','bb87c292ad', 'ebc200e928', 'a08d9a2476','5942004064']#['0a5c013435', 'f07340dfea',  '7bc286c1b6', 'd2f44bf242',  '85251de7d1', '0e75f3c4d9', '98fe276aa8', '7e7cd69a59', 'f3685d06a9', '21d970d8de', '8b5caf3398', 'ada5304e41', '4c5c60fa76']#'a5114ca13d','108ec0b806','bb87c292ad', 'ebc200e928', 'a08d9a2476','5942004064',

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./mymodels/dgx1/id_0005")
args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(scannetpp_scenes)

if not args.skip_training or not args.skip_rendering:
    parser.add_argument("--scannetpp", "-s", required=True, type=str)
    args = parser.parse_args()

if not args.skip_rendering:
    all_sources = []
    for scene in scannetpp_scenes:
        all_sources.append(args.scannetpp + "/" + scene)
 

    common_args = " --quiet --eval --skip_train"
    for i in range (4):
        for n_classes, lambda_sem in [[0,0.0],[64,0.001]]:
            for scene, source in zip(all_scenes, all_sources):
                #os.system("python src/render.py --iteration 7000 -s " + source + " -m " + args.output_path + "/" + f"scannet_sem{n_classes}_1h_{lambda_sem}_{scene}_{i} " + common_args)
                os.system("python src/render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + f"scannet_sem{n_classes}_1h_{lambda_sem}_{scene}_{i} " + common_args)

if not args.skip_metrics:
    scenes_string = ""
    for i in range(4):
        for n_classes, lambda_sem in [[0,0.0],[64,0.001]]:
            for scene in all_scenes:
                scenes_string += "\"" + args.output_path + "/" + f"scannet_sem{n_classes}_1h_{lambda_sem}_{scene}_{i} "+ "\" "

    os.system("python src/metrics.py -m " + scenes_string)

