import os
import subprocess
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Viewer script parameters")
    parser.add_argument('model', type=str)
    parser.add_argument('scenes', nargs="+", type=str, default=[])
    
    args = parser.parse_args()
    assert args.model in ["gaussian-splatting", "deep_splatting"]

    base_path = "/workspace/tberriel/splatting_ws/"
    scenes = args.scenes
    if len(scenes) == 0:
        scenes_path = os.path.join(base_path, "Scannet_data")
        for folder in os.listdir(scenes_path):
            if os.path.isdir(os.path.join(scenes_path,folder)):
                scenes.append(folder)
    res = 1
    if args.model == "deep_splatting":
        chkpt_args = "_sem32_1h_0.1" 
        train_file = "src/train.py"
        sh_degree = 1
    else: 
        chkpt_args = "" 
        train_file = "train.py"
        sh_degree = 4
    model_path = os.path.join(base_path,args.model)
    os.chdir(model_path)

    for n, scene in enumerate(scenes):
        source = os.path.join(base_path, "Scannnet_data",scene)
        chkpt = os.path.join(model_path, f"mymodels/scannet{chkpt_args}_{scene}/")
            
        stdout_file = os.path.join(chkpt, "stdout.log")
        stderr_file = os.path.join(chkpt, "stderr.log")
        print("Run {}: scene {}; model {}".format(n, scene))
            
        config_args = "-s {} -m {} --eval -r 1 --sh_degree {}".format(
            source,
            chkpt,
            sh_degree,
            )
        
        command = f"python {train_file} {config_args} "
        os.mkdir(chkpt, exist_ok=True)
        with open(stdout_file,'w') as out_file, open(stderr_file,'w') as err_file:
            subprocess.run("conda run -n deep_splatting  "+command,
                            shell=True,
                            executable="/bin/bash", 
                            stdout=out_file, 
                                    stderr=err_file)        
                    