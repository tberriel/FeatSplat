# ScanNet++ preprocessing

## Download data
To download ScanNet++ data register in the [official webpage](https://kaldir.vc.in.tum.de/scannetpp/) and follow the instructions.

The split of the scenes we used used is provided in featsplat_split.txt.

## Preprocess data
Copy the provided split file to `/<scannetpp_data_root>/splits/`, and update the provided configuration files `render.yml` and `undistort.yml` values of `<path_to_scannetpp_data_root>` with your path to the downloaded folder.

Follow the instructions from the [official toolkit](https://github.com/scannetpp/scannetpp
) to set up the scannetpp environment.

## Rasterizse depth and semantics
Semantic labels can be rasterized following the official toolkit instructions. Nevertheless, we propose an alternative solution developed before the official instructions to rasterize 2D semantics were published. For this modify the files common/render.py and dslr/undistort.py from ScanNet++ toolikt, with the provided `render.py` and `undistort.py` files.

Using the provided `render.yml` configuration file, run the rendering script.

```shell
(scannetpp) user@laptop:~/ScanNetpp/scannetpp python -m common.render common/configs/render.yml
```

## Undistort images
Using the provided `undistort.yml` configuration file, run the undistortion script. If you did not rasterize semantics, remove the field `input_sem_dir` from `undistort.yml`.

```shell
(scannetpp) user@laptop:~/ScanNetpp/scannetpp python -m dslr.undistort dslr/configs/undistort.yml
```

## Set Nerfstudio format
Finally, run the script `scannetpp_to_nerfstudio.py` to create a new data folder matching the structure expected by Nerfstudio dataloaders. The script will create symbolick links for images to reduce the memory overhead.

```shell
(featsplat) user@laptopt:~/FeatSplat/scannetpp python scannetpp_to_nerfstudio.py --data_path <path to undistorted data> --out_path <path to create new data folder> --scenes_list featsplat_split.txt --semantic
```
If not working with semantics, remove the flag `--semantic`.