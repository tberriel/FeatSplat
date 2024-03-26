import os
import subprocess
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Viewer script parameters")
    parser.add_argument('--model', type=str)
    parser.add_argument('--scenes', nargs="+", type=str, default=[])
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--n-classes', type=int, default=64)
    parser.add_argument('--lambda-sem', type=float, default=0.05)
    parser.add_argument("--weighted-ce-loss", action="store_true", default=False)
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
        chkpt_args = f"_sem{args.n_classes}_1h_{args.lambda_sem}" 
        train_file = "src/train.py"
        sh_degree = 1
        flags = f"--n_classes {args.n_classes} --lambda_sem {args.lambda_sem}"
        if args.weighted_ce_loss:
            flags += " --weighted_ce_loss "
            chkpt_args = f"_sem{args.n_classes}_wce1h_{args.lambda_sem}" 
    else: 
        chkpt_args = "" 
        train_file = "train.py"
        sh_degree = 4
        flags = ""
    model_path = os.path.join(base_path,args.model)
    os.chdir(model_path)

    for n, scene in enumerate(scenes):
        source = os.path.join(base_path, "Scannet_data",scene)
        for i in range(args.runs):
            chkpt = os.path.join(model_path, f"mymodels/scannet{chkpt_args}_{scene}_{i}/")
                
            stdout_file = os.path.join(chkpt, "stdout.log")
            stderr_file = os.path.join(chkpt, "stderr.log")
            print("Run {}:{} scene {}".format(n,i, scene))
                
            config_args = "-s {} -m {} --eval -r 1 --sh_degree {} {}".format(
                source,
                chkpt,
                sh_degree,
                flags
                )
            
            command = f"python {train_file} {config_args} "
            os.makedirs(chkpt, exist_ok=True)
            with open(stdout_file,'w') as out_file, open(stderr_file,'w') as err_file:
                subprocess.run("conda run -n deep_splatting  "+command,
                                shell=True,
                                executable="/bin/bash", 
                                stdout=out_file, 
                                        stderr=err_file)        
                    
