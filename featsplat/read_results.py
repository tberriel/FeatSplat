import os
import numpy as np
import json

def read_results_json_2(scenes, iterations, name_suffix, output_path,label, base_path="/home/tberriel/Workspaces/splatting_ws/deep_splatting/", folder="", per_view=False, add_idx=False):
    results = np.ndarray((len(scenes), iterations, 3))
    comp_results = np.ndarray((len(scenes), 3))
    failed_scenes = []
    metrics = {"ssim":[], "psnr":[], "lpips":[]}
    for i,scene in enumerate(scenes):
        for j in range(iterations):

            result_file = os.path.join(base_path, output_path, folder, scene+name_suffix)
            if iterations>1 or add_idx:
                result_file+=f"_{j}"

            try:
                if per_view:
                    with open(os.path.join(result_file, "per_view.json"), "r") as f:
                        data = json.load(f)
                        metrics["ssim"] += list(data["ours_30000"]["SSIM"].values())
                        metrics["psnr"] += list(data["ours_30000"]["PSNR"].values())
                        metrics["lpips"] += list(data["ours_30000"]["LPIPS"].values())

                with open(os.path.join(result_file, "results.json"), "r") as f:
                    data = json.load(f)
                    if not per_view:
                        assert not (np.isnan(data["ours_30000"]["SSIM"]) or np.isnan(data["ours_30000"]["PSNR"]) or np.isnan(data["ours_30000"]["LPIPS"]))
                        metrics["ssim"].append(data["ours_30000"]["SSIM"])
                        metrics["psnr"].append(data["ours_30000"]["PSNR"])
                        metrics["lpips"].append(data["ours_30000"]["LPIPS"])
                    if j == 0:
                        comp_results[i,0] = data["ours_30000"]["fps"]
                        comp_results[i,1]  = data["ours_30000"]["size"]
                        comp_results[i,2]  = data["ours_30000"]["n_gaussians"]
            except:
                if iterations>1:
                    failed_scenes.append(f"{scene}_{j}")
                else:
                    failed_scenes.append(f"{scene}")

    results = np.vstack([np.array(metrics["ssim"]),np.array(metrics["psnr"]),np.array(metrics["lpips"])])
    # Compute averages
    avg_results = results.mean(axis=(1))
    std_results = results.std(1)
    # Compute averages
    avg_comp_results = comp_results.mean(axis=(0))
    std_comp_results = comp_results.std(0)
    print("{} {:.3f}pm{:.3f}; {:.2f}pm{:.2f}; {:.3f}pm{:.3f}; {:.1f} pm {:.1f}; {:.0f} pm {:.0f}; {:.0f}k pm {:.0f} ".format(label,avg_results[0], std_results[0], avg_results[1], std_results[1], avg_results[2], std_results[2], avg_comp_results[0], std_comp_results[0], avg_comp_results[1], std_comp_results[1], avg_comp_results[2]/1000, std_comp_results[2]/1000))
    if len(failed_scenes)>0:
        print(f"Failed scenes: {failed_scenes}")
def read_results_json(scenes, iterations, name_suffix, output_path,label, base_path="/home/tberriel/Workspaces/splatting_ws/deep_splatting/", folder="", per_view=False, add_idx=False):
    results = np.ndarray((len(scenes), iterations, 3))
    comp_results = np.ndarray((len(scenes), 3))
    failed_scenes = []
    for i,scene in enumerate(scenes):
        for j in range(iterations):

            result_file = os.path.join(base_path, output_path, folder, scene+name_suffix)
            if iterations>1 or add_idx:
                result_file+=f"_{j}"
            result_file = os.path.join(result_file, "results.json")

            try:
                with open(result_file, "r") as f:
                    data = json.load(f)
                    assert not (np.isnan(data["ours_30000"]["SSIM"]) or np.isnan(data["ours_30000"]["PSNR"]) or np.isnan(data["ours_30000"]["LPIPS"]))
                    results[i,j,0] = data["ours_30000"]["SSIM"]
                    results[i,j,1]  = data["ours_30000"]["PSNR"]
                    results[i,j,2]  = data["ours_30000"]["LPIPS"]
                    if j == 0:
                        comp_results[i,0] = data["ours_30000"]["fps"]
                        comp_results[i,1]  = data["ours_30000"]["size"]
                        comp_results[i,2]  = data["ours_30000"]["n_gaussians"]
            except:
                if iterations>1:
                    failed_scenes.append(f"{scene}_{j}")
                else:
                    failed_scenes.append(f"{scene}")


    # Compute averages
    avg_results = results.mean(axis=(0,1))
    std_results = results.mean(axis=(1)).std(0)
    # Compute averages
    avg_comp_results = comp_results.mean(axis=(0))
    std_comp_results = comp_results.std(0)
    print("{} {:.3f}pm{:.3f}; {:.2f}pm{:.2f}; {:.3f}pm{:.3f}; {:.1f} pm {:.1f}; {:.0f} pm {:.0f}; {:.0f}k pm {:.0f} ".format(label,avg_results[0], std_results[0], avg_results[1], std_results[1], avg_results[2], std_results[2], avg_comp_results[0], std_comp_results[0], avg_comp_results[1], std_comp_results[1], avg_comp_results[2]/1000, std_comp_results[2]/1000))
    if len(failed_scenes)>0:
        print(f"Failed scenes: {failed_scenes}")


def bmvc_results():

    scannetpp_scenes =['0a5c013435', 'f07340dfea',  '7bc286c1b6', 'd2f44bf242',  '85251de7d1', '0e75f3c4d9', '98fe276aa8', '7e7cd69a59', 'f3685d06a9', '21d970d8de', '8b5caf3398', 'ada5304e41', '4c5c60fa76', 'ebc200e928', 'a5114ca13d', '5942004064','1ada7a0617','f6659a3107', '1a130d092a', '80ffca8a48',   '08bbbdcc3d' ]
    print("           SSIM         PSNR            LPIPS       FPS         Size(MB)      Nº Gaus")
    print("21 Scannet++ res: 1752x1168   ")
    read_results_json_2(scannetpp_scenes, 1, "","eval/gs", "GS       "   , folder="scannet_sem/", add_idx=True, per_view=False)
    read_results_json_2(scannetpp_scenes, 1, "","eval/c3dgs", "C3DGS    ", folder="scannet_sem/", add_idx=True, per_view=False)
    read_results_json_2(scannetpp_scenes, 1, "","eval/featsplat_32", "FS+pe+cp ", folder="scannet_sem/", add_idx=True, per_view=False)
    read_results_json_2(scannetpp_scenes, 1, "","eval/featsplat_16", "FS+pe+cp 16 ", folder="scannet_sem/", add_idx=True, per_view=False)
    read_results_json_2(scannetpp_scenes, 1, "_sem64_0.001","eval/featsplat_32","SFS+pe+cp", folder="scannet_sem/", add_idx=True, per_view=False)

    print("Mip-360 res: 1237x822 and 1557x1038")
    mipnerf360_scenes = ["bicycle", "flowers", "garden", "stump", "treehill","room", "counter", "kitchen", "bonsai"] 
    read_results_json(mipnerf360_scenes, 1, "","eval/gs",   "GS          ", folder="360_v2")
    read_results_json(mipnerf360_scenes, 1, "","eval/c3dgs",   "C3DGS       ", folder="360_v2")
    #read_results_json(mipnerf360_scenes, 1, "","eval/base", "FS          ", folder="360_v2")
    read_results_json(mipnerf360_scenes, 1, "","eval/featsplat_32", "FS+pe+cp    ", folder="360_v2")
    read_results_json(mipnerf360_scenes, 1, "","eval/featsplat_16", "FS+pe+cp 16  ", folder="360_v2")
    read_results_json(mipnerf360_scenes, 1, "","eval/ablations/pecpr", "FS+pe+cp+cr ", folder="360_v2")

    print("tant res: 980x545")
    tanks_and_temples_scenes= ["truck", "train"]
    read_results_json(tanks_and_temples_scenes, 1, "","eval/gs",   "GS         ", folder="tant")
    read_results_json(tanks_and_temples_scenes, 1, "","eval/c3dgs",   "C3DGS      ", folder="tant")
    read_results_json(tanks_and_temples_scenes, 1, "","eval/base", "FS         ", folder="tant")
    read_results_json(tanks_and_temples_scenes, 1, "","eval/featsplat_32", "FS+pe+cp   ", folder="tant")
    read_results_json(tanks_and_temples_scenes, 1, "","eval/featsplat_16", "FS+pe+cp 16", folder="tant")
    read_results_json(tanks_and_temples_scenes, 1, "","eval/ablations/pecpr", "FS+pe+cp+cr ", folder="tant")
    print("DB res: 1332x876 and 1264x832")
    deep_blending_scenes = ["drjohnson", "playroom"] 
    read_results_json(deep_blending_scenes, 1, "","eval/gs",   "GS          ", folder="db")
    read_results_json(deep_blending_scenes, 1, "","eval/c3dgs",   "C3DGS          ", folder="db")
    read_results_json(deep_blending_scenes, 1, "","eval/base", "FS          ", folder="db")
    read_results_json(deep_blending_scenes, 1, "","eval/featsplat_32", "FS+pe+cp    ", folder="db")
    read_results_json(deep_blending_scenes, 1, "","eval/featsplat_16", "FS+pe+cp 16  ", folder="db")
    read_results_json(deep_blending_scenes, 1, "","eval/ablations/pecpr", "FS+pe+cp+cr ", folder="db")

if __name__ == "__main__":

    dir_list =  os.listdir("/home/tberriel/Workspaces/splatting_ws/gaussian-splatting/Datasets/ScanNetpp")
    """"""
    scannetpp_rest_scenes = [x[:-2] for x in os.listdir("/home/tberriel/Workspaces/splatting_ws/deep_splatting/eval/featsplat_32/scannet_rest") if x not in ['9b74afd2d2_0', '355e5e32db_0', 'cc5237fd77_0', '5f99900f09_0', '6b40d1a939_0', '0a76e06478_0', 'e9e16b6043_0', '95d525fbfd_0', '484ad681df_0', 'fb5a96b1a2_0', '251443268c_0', '6855e1ac32_0', 'faec2f0468_0', '9071e139d9_0', 'b1d75ecd55_0', '281ba69af1_0', 'eb4bc76767_0', 'ab11145646_0', 'd6d9ddb03f_0', 'bd7375297e_0', '8a20d62ac0_0', '961911d451_0', 'a05ee63164_0', 'b0a08200c9_0', 'e7af285f7d_0', '54b6127146_0', 'c24f94007b_0', '09c1414f1b_0', '28a9ee4557_0', '13285009a4_0', 'cf1ffd871d_0', 'c545851c4f_0', 'e0de253456_0', 'bc03d88fc3_0', '5748ce6f01_0', '419cbe7c11_0', 'bf6e439e38_0', '32280ecbca_0', '7977624358_0', 'd918af9c5f_0', 'acd95847c5_0', 'e9ac2fc517_0', 'd6cbe4b28b_0', 'a003a6585e_0', '280b83fcf3_0', 'e050c15a8d_0', '89214f3ca0_0', '6ee2fc1070_0', '8e6ff28354_0', '7f4d173c9c_0', 'e3ecd49e2b_0', 'b08a908f0f_0', 'ef25276c25_0', 'c49a8c6cff_0', '410c470782_0', '303745abc7_0']+['108ec0b806_0', 'a08d9a2476_0', 'bb87c292ad_0','a4e227f506_0']]

    scannetpp_rest_scenes = [f"{x}_0" for x in scannetpp_rest_scenes]
    
    scannetpp_scenes =['0a5c013435', 'f07340dfea',  '7bc286c1b6', 'd2f44bf242',  '85251de7d1', '0e75f3c4d9', '98fe276aa8', '7e7cd69a59', 'f3685d06a9', '21d970d8de', '8b5caf3398', 'ada5304e41', '4c5c60fa76', 'ebc200e928', 'a5114ca13d', '5942004064','1ada7a0617','f6659a3107', '1a130d092a', '80ffca8a48',   '08bbbdcc3d' ]
    print("Scannet++ res: 1752x1168   ")
    read_results_json_2(scannetpp_scenes, 1, "","eval/gs", "GS       "   , folder="scannet_sem/", add_idx=True, per_view=True)
    read_results_json_2(scannetpp_scenes, 1, "","eval/c3dgs", "C3DGS    ", folder="scannet_sem/", add_idx=True, per_view=True)
    read_results_json_2(scannetpp_scenes, 1, "","eval/featsplat_32", "FS+pe+cp ", folder="scannet_sem/", add_idx=True, per_view=True)
    read_results_json_2(scannetpp_scenes, 1, "","eval/featsplat_16", "FS+pe+cp 16 ", folder="scannet_sem/", add_idx=True, per_view=True)
    read_results_json_2(scannetpp_scenes, 1, "_sem64_0.001","eval/featsplat_32","SFS+pe+cp", folder="scannet_sem/", add_idx=True, per_view=True)

    print("Scannet++ res: 1752x1168   ")
    read_results_json_2(scannetpp_rest_scenes, 1, "","eval/gs", "GS       "   , folder="scannet_rest/", add_idx=False, per_view=True)
    read_results_json_2(scannetpp_rest_scenes, 1, "","eval/featsplat_32", "FS+pe+cp ", folder="scannet_rest/", add_idx=False, per_view=True)
    