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

def organize_for_ns_dataloader(scenes, destination, base_path, semantic):

    for scene in scenes:
        scene_path = os.path.join(base_path+"/data", scene)
        new_scene_path = os.path.join(base_path+destination, scene)
        os.makedirs(new_scene_path, exist_ok=True)
        print(f"Created folder: {new_scene_path}")

        # Create symbolic links for 
        folders = ["undistorted_images"]
        if semantic:
            semantic_path = os.path.join(scene_path, "undistorted_projected_semantic")
            assert os.path.exists(semantic_path), f"Could not find {semantic_path}"
            folders.append("undistorted_projected_semantic")

        for folder in folders:
            folder_path = os.path.join(scene_path, "dslr", folder)

            # Read images_folder content
            image_files = glob.glob(os.path.join(folder_path, "*"))

            # Create symbolic links for each image inside new_folder
            for image_file in image_files:
                image_link = os.path.basename(image_file)
                image_link_path = os.path.join(new_scene_path, image_link)
                try:
                    os.symlink(image_file, image_link_path)
                except Exception as e:
                    print(e)
                    break
            print(f"Created images symbolic links")

        # Read transforms.json and split into train and val
        transforms_file = os.path.join(scene_path, "dslr/nerfstudio/transforms_undistorted.json")
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


            file = os.path.join(new_scene_path, f"transforms_{data_set}.json")
            with open(file, "w") as f:
                json.dump(transform, f, indent=4)
            print(f"Created file: {file}")

        # Change pointcloud from .txt to .ply
        bin_path = os.path.join(scene_path, "dslr/colmap/points3D.txt")
        ply_path = os.path.join(new_scene_path, "points3D.ply")
        xyz, rgb, _ = read_points3D_text(bin_path)
        storePly(ply_path, xyz, rgb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create folders in a destination directory")
    parser.add_argument("--data_path", help="Origin directory")
    parser.add_argument("--out_path",  help="Destination directory")
    parser.add_argument("--scenes_list", type = str, default="./featsplat_split.txt")
    parser.add_argument("--semantic", action='store_true', default=False)
    args = parser.parse_args()
    
    with open(args.scenes_list) as f:
        scenes = f.read().splitlines()

    organize_for_ns_dataloader(scenes, args.out_path, args.data_path, args.semantic)
