import numpy as np
import argparse
import os
import json
import glob
import math
from plyfile import PlyData, PlyElement
import subprocess
import yaml


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    xyzs = None
    rgbs = None
    errors = None
    num_points = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_points += 1


    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    count = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = np.array(float(elems[7]))
                xyzs[count] = xyz
                rgbs[count] = rgb
                errors[count] = error
                count += 1

    return xyzs, rgbs, errors

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def organize_like_gs(folders, destination, base_path, segmentation):

    for folder in folders:
        folder_path = os.path.join(base_path+"/data", folder)
        new_folder_path = os.path.join(base_path+destination, folder)
        os.makedirs(new_folder_path, exist_ok=True)
        print(f"Created folder: {new_folder_path}")

        # Create symbolic links for images
        images_folder = os.path.join(folder_path, "dslr/undistorted_images")

        # Read images_folder content
        image_files = glob.glob(os.path.join(images_folder, "*"))

        # Create symbolic links for each image inside new_folder
        for image_file in image_files:
            image_link = os.path.basename(image_file)
            image_link_path = os.path.join(new_folder_path, image_link)
            try:
                os.symlink(image_file, image_link_path)
            except Exception as e:
                print(e)
                break
        print(f"Created images symbolic links")

        # Read transforms.json and split into train and val
        transforms_file = os.path.join(folder_path, "dslr/nerfstudio/transforms_undistorted.json")
        with open(transforms_file, "r") as f:
            transforms_data = json.load(f)
        
        common_keys = ["fl_x","fl_y","w","h","k1","k2","k3","k4","camera_model","has_mask","aabb_range"]
        angle_x =  focal2fov(transforms_data["fl_x"],transforms_data["w"])
        angle_y = focal2fov(transforms_data["fl_y"],transforms_data["h"])

        for frames_key, data_set in [["frames", "train"], ["test_frames", "test"]]: 
            transform = dict()
            for key in common_keys:
                transform[key] = transforms_data[key]

            transform["camera_angle_x"] = angle_x
            transform["camera_angle_y"] = angle_y

            transform["frames"] = transforms_data[frames_key]
        
            for frame in transform["frames"]:
                frame["file_path"] = frame["file_path"][:-4]# remove .JPG suffix


            file = os.path.join(new_folder_path, f"transforms_{data_set}.json")
            with open(file, "w") as f:
                json.dump(transform, f, indent=4)
            print(f"Created file: {file}")

        # read point cloud
        bin_path = os.path.join(folder_path, "dslr/colmap/points3D.txt")
        ply_path = os.path.join(new_folder_path, "points3D.ply")
        xyz, rgb, _ = read_points3D_text(bin_path)
        storePly(ply_path, xyz, rgb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create folders in a destination directory")
    parser.add_argument("--scenes_list", type = str, default="/home/tberriel/Workspaces/splatting_ws/featsplat/scripts/scannetpp_21_scenes.txt")
    parser.add_argument("--scannetpp_toolbox_path", "-stp", default="/media/tberriel/My_Book_2/ScanNetpp/scannetpp", help="Origin directory")
    parser.add_argument("--scannetpp_data_path", "-sdp", default="/media/tberriel/My_Book_2/ScanNetpp/data", help="Origin directory")
    parser.add_argument("--destination", "-d", default="scannetpp_nerfstudio", help="Destination directory")
    parser.add_argument("--undistort", action='store_true', default=False)
    parser.add_argument("--organize", action='store_true', default=True)
    args = parser.parse_args()
    
    with open(args.scenes_list) as f:
        scenes = f.read().splitlines()

    if args.undistort:            
        # Undistort RGB images and labels
        config_file = os.path.join(args.scannetpp_toolbox_path,"dslr/configs/undistort.yml")
        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)

        config_data["data_root"] = args.scannetpp_data_path
        config_data["scene_ids"] = scenes

        with open(config_file, "w") as f:
            yaml.safe_dump(config_data, f)
        
        os.chdir(args.scannetpp_toolbox_path)
        command = f"python -m dslr.undistort {config_file} "
        subprocess.run("conda run -n scannetpp  "+command, shell=True, check=True, executable="/bin/bash")    

    if args.organize:
        # Organize folder folowing gaussian splatting nerfstudio dataloader
        organize_like_gs(scenes, args.destination, args.scannetpp_data_path, False)
