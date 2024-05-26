import os
import numpy as np
import json

from argparse import ArgumentParser

def read_results_json(scenes, model, base_path,dataset_folder, label_padding):
    label = model.ljust(label_padding)
    results = np.ndarray((len(scenes), 3))
    comp_results = np.ndarray((len(scenes), 3))
    failed_scenes = []
    for i,scene in enumerate(scenes):

            result_file = os.path.join(base_path, model, dataset_folder, scene)
            result_file = os.path.join(result_file, "results.json")

            try:
                with open(result_file, "r") as f:
                    data = json.load(f)
                    assert not (np.isnan(data["ours_30000"]["SSIM"]) or np.isnan(data["ours_30000"]["PSNR"]) or np.isnan(data["ours_30000"]["LPIPS"]))
                    results[i,0] = data["ours_30000"]["SSIM"]
                    results[i,1]  = data["ours_30000"]["PSNR"]
                    results[i,2]  = data["ours_30000"]["LPIPS"]
                    comp_results[i,0] = data["ours_30000"]["fps"]
                    comp_results[i,1]  = data["ours_30000"]["size"]
                    comp_results[i,2]  = data["ours_30000"]["n_gaussians"]
            except:
                failed_scenes.append(f"{scene}")


    # Compute averages
    avg_results = results.mean(axis=(0))
    std_results = results.std(0)
    # Compute averages
    avg_comp_results = comp_results.mean(axis=(0))
    std_comp_results = comp_results.std(0)
    print("{} {:.3f}pm{:.3f}; {:.2f}pm{:.2f}; {:.3f}pm{:.3f}; {:.1f} pm {:.1f}; {:.0f} pm {:.0f}; {:.0f}k pm {:.0f} ".format(label,avg_results[0], std_results[0], avg_results[1], std_results[1], avg_results[2], std_results[2], avg_comp_results[0], std_comp_results[0], avg_comp_results[1], std_comp_results[1], avg_comp_results[2]/1000, std_comp_results[2]/1000))
    if len(failed_scenes)>0:
        print(f"Failed scenes: {failed_scenes}")

def read_dataset_results(scenes, models, base_path, dataset_folder):
    label_padding = max([len(x)+1 for x in models])
    for model in models:
        read_results_json(scenes, model, base_path, dataset_folder, label_padding)
         

if __name__ == "__main__":

    parser = ArgumentParser(description="Testing script parameters")

    parser.add_argument("--base_path", default="/home/tberriel/Workspaces/splatting_ws/featsplat/data/output/eval")
    args = parser.parse_args()

    models = ["gs", "c3dgs", "featsplat_16", "featsplat_32"]

    print("           SSIM         PSNR            LPIPS       FPS         Size(MB)      Nº Gaus")

    print("Mip-360 res: 1237x822 and 1557x1038")
    mipnerf360_scenes = ["bicycle", "flowers", "garden", "stump", "treehill","room", "counter", "kitchen", "bonsai"] 
    read_dataset_results(mipnerf360_scenes, models, args.base_path, "360_v2")

    print("\ntandt res: 980x545")
    tanks_and_temples_scenes= ["truck", "train"]
    read_dataset_results(tanks_and_temples_scenes, models, args.base_path, "tandt")

    print("\nDB res: 1332x876 and 1264x832")
    deep_blending_scenes = ["drjohnson", "playroom"] 
    read_dataset_results(deep_blending_scenes, models, args.base_path, "db")

    print("\n21 Scannet++ res: 1752x1168   ")
    scannetpp_scenes =['0a5c013435', 'f07340dfea',  '7bc286c1b6', 'd2f44bf242',  '85251de7d1', '0e75f3c4d9', '98fe276aa8', '7e7cd69a59', 'f3685d06a9', '21d970d8de', '8b5caf3398', 'ada5304e41', '4c5c60fa76', 'ebc200e928', 'a5114ca13d', '5942004064','1ada7a0617','f6659a3107', '1a130d092a', '80ffca8a48',   '08bbbdcc3d' ]
    read_dataset_results(scannetpp_scenes, models+["semfeatsplat_32"], args.base_path, "scannetpp/")