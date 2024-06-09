 
# Feature Splatting for Better Novel View Synthesis with Low Overlap
Tomas Berriel Martins, Javier Civera<br>
| [Webpage]() | [Arxiv Paper](https://arxiv.org/abs/2405.15518) | <br>
| [Pre-trained Models (15 GB)](https://mega.nz/file/ObRwHYxK#cBEKdGkaVAIfu1GE3kRwfsv20BAFDySklOQG0ket_oo) [Evaluation Images (3.8 GB)](https://mega.nz/file/TPQWRbjK#DdrMNQRdezbxEdJP5OVtbeXUo7ftZLzA1L9eExHXkwY) |<br>

Official implementation of the paper "Feature Splatting for Better Novel View Synthesis with Low Overlap". We further provide the reference images used to create the error metrics reported in the paper, as well as recently created, pre-trained models. 

<a href="https://i3a.unizar.es/en"><img height="100" src="assets/logo_i3a.png"> </a>
<a href="https://ropert.i3a.es/"><img height="100" src="assets/logo_ropert.png"> </a>

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@Article{martins2024feature,
      title={Feature Splatting for Better Novel View Synthesis with Low Overlap}, 
      author={T. Berriel Martins and Javier Civera},
      year={2024},
      eprint={2405.15518},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}</code></pre>
  </div>
</section>


## Cloning the Repository

The repository contains submodules. To avoid missing dependencies clone using the recursive flag: 
```shell
# SSH
git clone git@github.com:tberriel/featsplat.git --recursive
```

## Overview
The code and repository is based on the official [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file) impementation. 

The codebase has 4 main components:
- A PyTorch-based optimizer to produce a 3D Gaussian model from SfM inputs
- A network viewer that allows to connect to and visualize the optimization process

The repository has been tested on Ubuntu Linux 20.04. 

### Data

The MipNeRF360 scenes are hosted by the paper authors [here](https://jonbarron.info/mipnerf360/). You can find INRIA SfM data sets for Tanks&Temples and Deep Blending [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip). If you do not provide an output model directory (```-m```), trained models are written to folders with randomized unique names inside the ```output``` directory. At this point, the trained models may be viewed with the real-time viewer (see further below).

## Optimizer

The optimizer uses PyTorch and CUDA extensions in a Python environment to produce trained models. 

### Hardware Requirements

- CUDA-ready GPU with Compute Capability 7.0+
- 24 GB VRAM (to train to paper evaluation quality)

### Software Requirements
- Conda (recommended for easy setup)
- C++ Compiler for PyTorch extensions
- CUDA SDK 11 for PyTorch extensions(we used 11.8, **known issues with 11.6**)
- C++ Compiler and CUDA SDK must be compatible

### Setup

Our default, provided install method is based on Conda package and environment management:
```shell
conda env create --file environment.yml
conda activate featsplat
```
Please note that this process assumes that you have CUDA SDK **11** installed, not **12**.


### Running

To run the optimizer with FeatSplat32 configuration, simply use

```shell
python featsplat/train.py -s <path to COLMAP or NeRF Synthetic dataset> --pixel_embedding --pos_embedding
```
 and for Semantic FeatSplat32

```shell
python featsplat/train.py -s <path to COLMAP or NeRF Synthetic dataset> --pixel_embedding --pos_embedding --n_classes 64 --semantic_classes_path <path to 64_most_common_classes>
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for train.py</span></summary>

  **Feature Splatting flages**
  #### --n_latents
  Size of Gaussians' feature vectors. Default = 32 . This value, should be equal to the value of N_CHANNELS in featsplat/submodules/diff-feat-gaussian-rasterization/ config.h . 
  #### --h_layers
  Number of hidden layer of the output MLP. Default = 0
  #### --n_neurons
  Number of neurons on the output MLP neurons. Default = 64.
  #### --pixel_embedding
  Add this flag to concatenate the pixel embedding to the feature vectors before the MLP.
  #### --pos_embedding
  Add this flag to concatenate the camera position embedding to the feature vectors before the MLP.
  #### --rot_embedding
  Add this flag to concatenate the camera rotation encoded as Euler angles to the feature vectors before the MLP.
  #### --images_extension = ".png"
  Extension of the RGB images filed. For ScanNet++ dataset use ".JPG", (i.e. --image_extension .JPG ). This flag is only taken into account for Blender like datasets. Default = ".png".
  #### --feature_lr_init
  Initial learning rate for Feature vectors. If --gaussian_splating is set, this will be the SHs learning rate. Default = 0.0025
  #### --feature_lr_final 
  Final learning rate for Feature vectors. If --gaussian_splating is set, this will be the SHs learning rate. Default = 0.00025
  #### --feature_lr_max_steps
  Number of steps to go from feature_lr_init to feature_lr_final. Default = 30_000
  #### --mlp_lr_init
  Initial learning rate for the output MLP. If --gaussian_splating is set, this will be the SHs learning rate. Default = 0.001
  #### --mlp_lr_final
  Final learning rate for the output MLP. If --gaussian_splating is set, this will be the SHs learning rate. Default = 0.0001
  #### --mlp_lr_max_steps
  Number of steps to go from mlp_lr_init to mlp_lr_final. Default = 30_000

  **Semantic flags**
  #### --n_classes
  Number of classes to perform closed-vocabulary semantic segmentation. Default = 0.
  #### --weighted_ce_loss
  Add this flag to use a weighted Cross-Entropy Loss for semantic segmentation training. If --n_classes = 0, this flag is ignored. Default is a normal Cross-Entropy Loss.
  #### --semantic_classes_path
  Path to the file 64_most_common_classes. If --n_classes = 0, this flag is ignored.
  #### --lambda_sem
  Weight of the Semantic Cross-Entropy Loss. Default is 0.001. If --n_classes = 0, this flag is ignored.

  #### --gaussian_splatting
  Add this flag to train a basic 3D Gaussian Splatting model. If set the follwing flags will be ignored: n_latents, h_layers, n_neurons, pixel_embedding, pos_embedding, rot_embedding, n_classes, weighted_ce_loss, semantic_classes_path, lambda_sem . **Do not add this flag to train a Feature Splatting model**.

  **Base 3DGS Flags**
  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --model_path / -m 
  Path where the trained model should be stored (```output/<random>``` by default).
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --resolution / -r
  Specifies resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. **If not set and input image width exceeds 1.6K pixels, inputs are automatically rescaled to this target.**
  #### --data_device
  Specifies where to put the source image data, ```cuda``` by default, recommended to use ```cpu``` if training on large/high-resolution dataset, will reduce VRAM consumption, but slightly slow down training. Thanks to [HrsPythonix](https://github.com/HrsPythonix).
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --sh_degree
  Order of spherical harmonics to be used (if --gaussian_splatting is set, --sh_degree should not be larger than 3). Default = 0.
  #### --convert_SHs_python
  Flag to make pipeline compute forward and backward of SHs with PyTorch instead of ours. If --gaussian_splatting is not set, this flag will be ignored.
  #### --convert_cov3D_python
  Flag to make pipeline compute forward and backward of the 3D covariance with PyTorch instead of ours.
  #### --debug
  Enables debug mode if you experience erros. If the rasterizer fails, a ```dump``` file is created that you may forward to us in an issue so we can take a look.
  #### --debug_from
  Debugging is **slow**. You may specify an iteration (starting from 0) after which the above debugging becomes active.
  #### --iterations
  Number of total iterations to train for, ```30_000``` by default.
  #### --ip
  IP to start GUI server on, ```127.0.0.1``` by default.
  #### --port 
  Port to use for GUI server, ```6009``` by default.
  #### --test_iterations
  Space-separated iterations at which the training script computes L1 and PSNR over test set, ```7_000, 21000, 30_000, 35_000, 42_000``` by default.
  #### --save_iterations
  Space-separated iterations at which the training script saves the Gaussian model, ```7_000, 21000, 30_000, 35_000, 42_000 <iterations>``` by default.
  #### --checkpoint_iterations
  Space-separated iterations at which to store a checkpoint for continuing later, saved in the model directory.
  #### --start_checkpoint
  Path to a saved checkpoint to continue training from.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 
  #### --opacity_lr
  Opacity learning rate, ```0.05``` by default.
  #### --scaling_lr
  Scaling learning rate, ```0.005``` by default.
  #### --rotation_lr
  Rotation learning rate, ```0.001``` by default.
  #### --position_lr_max_steps
  Number of steps (from 0) where position learning rate goes from ```initial``` to ```final```. ```30_000``` by default.
  #### --position_lr_init
  Initial 3D position learning rate, ```0.00016``` by default.
  #### --position_lr_final
  Final 3D position learning rate, ```0.0000016``` by default.
  #### --position_lr_delay_mult
  Position learning rate multiplier (cf. Plenoxels), ```0.01``` by default. 
  #### --densify_from_iter
  Iteration where densification starts, ```500``` by default. 
  #### --densify_until_iter
  Iteration where densification stops, ```15_000``` by default.
  #### --densify_grad_threshold
  Limit that decides if points should be densified based on 2D position gradient, ```0.0002``` by default.
  #### --densification_interval
  How frequently to densify, ```100``` (every 100 iterations) by default.
  #### --opacity_reset_interval
  How frequently to reset opacity, ```3_000``` by default. 
  #### --lambda_dssim
  Influence of SSIM on total loss from 0 to 1, ```0.2``` by default. 
  #### --percent_dense
  Percentage of scene extent (0--1) a point must exceed to be forcibly densified, ```0.01``` by default.

</details>
<br>

Following 3DGS and MipNeRF360, we target images at resolutions in the 1-1.6K pixel range. For convenience, arbitrary-size inputs can be passed and will be automatically resized if their width exceeds 1600 pixels. We recommend to keep this behavior, but you may force training to use your higher-resolution images by setting ```-r 1```.


### Evaluation
By default, the trained models use all available images in the dataset. To train them while withholding a test set for evaluation, use the ```--eval``` flag. This way, you can render training/test sets and produce error metrics as follows:
```shell
python featsplat/train.py -s <path to COLMAP or NeRF Synthetic dataset> --pixel_embedding --pos_embedding --eval # Train with train/test split
python featsplat/render.py -m <path to trained model> # Generate renderings
python featsplat/metrics.py -m <path to trained model> # Compute error metrics on renderings
python featsplat/computation_metrics.py -m <path to trained model> # Compute computational metrics
```

If you want to evaluate our [pre-trained models](https://mega.nz/file/ObRwHYxK#cBEKdGkaVAIfu1GE3kRwfsv20BAFDySklOQG0ket_oo), you will have to download the corresponding source data sets and indicate their location to ```render.py``` with an additional ```--source_path/-s``` flag. 
```shell
python featsplat/render.py -m <path to pre-trained model> -s <path to COLMAP dataset>
python featsplat/metrics.py -m <path to pre-trained model>
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for render.py</span></summary>

  #### --model_path / -m 
  Path to the trained model directory you want to create renderings for.
  #### --skip_train
  Flag to skip rendering the training set.
  #### --skip_test
  Flag to skip rendering the test set.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 
  #### --gaussian_splatting
  Add this flag to render a basic 3D Gaussian Splatting model.

  **The below parameters will be read automatically from the model path, based on what was used for training. However, you may override them by providing them explicitly on the command line.** 

  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --resolution / -r
  Changes the resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. ```1``` by default.
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --convert_SHs_python
  Flag to make pipeline render with computed SHs from PyTorch instead of CUDA's. If --gaussian_splatting is not set, this flag will be ignored.
  #### --convert_cov3D_python
  Flag to make pipeline render with computed 3D covariance from PyTorch instead of ours.

</details>

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for metrics.py</span></summary>

  #### --model_paths / -m 
  Space-separated list of model paths for which metrics should be computed.
</details>
<br>

We further provide a modified version of Like 3DGS' ```full_eval.py``` script. This script specifies the routine used in our evaluation and demonstrates the use of some additional parameters, e.g., ```--images (-i)``` to define alternative image directories within COLMAP data sets. If you have downloaded and extracted all the training data, you can run it like this:
```shell
python featsplat/full_eval.py -m360 <mipnerf360 folder> -tat <tanks and temples folder> -db <deep blending folder> -s <scannetpp> --cam_pos --pembedding
```
In the current version, this process takes about 10h for Mip-360, T\&T and DB, and 30h for ScanNet++ on our reference machine containing an A100. If you want to do the full evaluation on our pre-trained models, you can specify their download location and skip training. 
```shell
python featsplat/full_eval.py -o <directory with pretrained models> --skip_training -m360 <mipnerf360 folder> -tat <tanks and temples folder> -db <deep blending folder> -s <scannetpp>
```

Although we provide paper's [evaluation images](https://mega.nz/file/TPQWRbjK#DdrMNQRdezbxEdJP5OVtbeXUo7ftZLzA1L9eExHXkwY), they do not follow the directory structure expected by full_eval.py. In the next weeks we will update how to compute the metrics from those images. In the meantime, if you want to compute the metrics on our paper's, you can reorder the images to the following directory structure
```
directory with evaluation images
  |-- Dataset n
      |-- Scene n
        |-- test
          |-- ours_30000
            |-- gt
            |  |-- 00000.png
            |  |-- ...
            |-- renders
              |-- 00000.png
              |-- ...
```
and skip rendering. In this case it is not necessary to provide the source datasets. You can compute metrics for multiple image sets at a time. 
```shell
python full_eval.py -o <directory with evaluation images> --skip_training --skip_rendering
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for full_eval.py</span></summary>
  
  #### --skip_training
  Flag to skip training stage.
  #### --skip_rendering
  Flag to skip rendering stage.
  #### --skip_metrics
  Flag to skip metrics calculation stage.
  #### --skip_comp_metrics
  Flag to skip computational metrics calculation stage.
  #### --test_midway
  Flag to perform evaluation during training. If set, evaluation on the validation set will be performed at the default values of flag --test_iterations from train.py.
  #### --output_path
  Directory to put renderings and results in, ```./data/working/eval``` by default, set to pre-trained model location if evaluating them.
  #### --train_iterations
  Maximum number of steps to optimize the model. Default = 30_000.
  #### --n_latents
  Size of Gaussians' feature vectors. Default = 32 . This value, should be equal to the value of N_CHANNELS in featsplat/submodules/diff-feat-gaussian-rasterization/ config.h . 
  #### --h_layers
  Number of hidden layer of the output MLP. Default = 0
  #### --n_neurons
  Number of neurons on the output MLP neurons. Default = 64.
  #### --n_classes
  Number of classes to perform closed-vocabulary semantic segmentation. Default = 0.
  #### --lambda_sem
  Weight of the Semantic Cross-Entropy Loss. Default is 0.001. If --n_classes = 0, this flag is ignored.
  #### --pembedding
  Add this flag to concatenate the pixel embedding to the feature vectors before the MLP.
  #### --cam_pos
  Add this flag to concatenate the camera position embedding to the feature vectors before the MLP.
  #### --cam_rot
  Add this flag to concatenate the camera rotation encoded as Euler angles to the feature vectors before the MLP.
  #### --gs
  Add this flag to train a basic 3D Gaussian Splatting model. If set the follwing flags will be ignored: n_latents, h_layers, n_neurons, pembedding, cam_pos, cam_rot, n_classes, lambda_sem . **Do not add this flag to train a Feature Splatting model**.
  #### --data_device
  Specifies where to put the source image data, ```cuda``` by default, recommended to use ```cpu``` if training on large/high-resolution dataset, will reduce VRAM consumption, but slightly slow down training.  
  #### --sh_degree
  Order of spherical harmonics to be used (if --gaussian_splatting is set, --sh_degree should not be larger than 3). Default = 3 if --gs set, else 0.

  #### --mipnerf360 / -m360
  Path to MipNeRF360 source datasets. If not set, the dataset will not be optimized.
  #### --tanksandtemples / -tat
  Path to Tanks&Temples source datasets. If not set, the dataset will not be optimized.
  #### --deepblending / -db
  Path to Deep Blending source datasets. If not set, the dataset will not be optimized.
  #### --scannetpp / -s
  Path to ScanNet++ source datasets. If not set, the dataset will not be optimized.

  #### --mipnerf360_outdoor_scenes
  List of outdoor scenes from MipNeRF360 to optimize. Default = [bicycle, flowers, garden, stump, treehill].
  #### --mipnerf360_indoor_scenes
  List of indoor scenes from MipNeRF360 to optimize. Default = [room, counter, kitchen, bonsai].
  #### --tanks_and_temples_scenes
  List of scenes from Tank \& Temples to optimize. Default = [truck, train].
  #### --
  List of scenes from Deep Blending to optimize. Default = [drjohnson, playroom].
  #### --scannetpp_scenes
  List of scenes from ScanNet++ to opTimize. Default = [0a5c013435, f07340dfea, 7bc286c1b6, d2f44bf242, 85251de7d1, 0e75f3c4d9, 98fe276aa8, 7e7cd69a59, f3685d06a9, 21d970d8de, 8b5caf3398, ada5304e41, 4c5c60fa76, ebc200e928, a5114ca13d, 5942004064, 1ada7a0617,f6659a3107, 1a130d092a, 80ffca8a48, 08bbbdcc3d]

</details>
<br>

## Interactive Viewers
This repository is compatible with [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file) Network Viewer remote application developed using the [SIBR](https://sibr.gitlabpages.inria.fr/) framework. Currently it is not comaptible with Real-Time Viewer.

### Hardware Requirements
- OpenGL 4.5-ready GPU and drivers (or latest MESA software)
- 4 GB VRAM recommended
- CUDA-ready GPU with Compute Capability 7.0+ (only for Real-Time Viewer)

### Software Requirements
- Visual Studio or g++, **not Clang** (we used Visual Studio 2019 for Windows)
- CUDA SDK 11, install *after* Visual Studio (we used 11.8)
- CMake (recent version, we used 3.24)
- 7zip (only on Windows)

### Installation from Source
If you cloned with submodules (e.g., using ```--recursive```), the source code for the viewers is found in ```SIBR_viewers```. The network viewer runs within the SIBR framework for Image-based Rendering applications.

#### Ubuntu 22.04
You will need to install a few dependencies before running the project setup.
```shell
# Dependencies
sudo apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev
# Project setup
cd SIBR_viewers
cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release # add -G Ninja to build faster
cmake --build build -j24 --target install
``` 

#### Ubuntu 20.04
Backwards compatibility with Focal Fossa is not fully tested, but building SIBR with CMake should still work after invoking
```shell
git checkout fossa_compatibility
```

### Navigation in SIBR Viewers
The SIBR interface provides several methods of navigating the scene. By default, you will be started with an FPS navigator, which you can control with ```W, A, S, D, Q, E``` for camera translation and ```I, K, J, L, U, O``` for rotation. Alternatively, you may want to use a Trackball-style navigator (select from the floating menu). You can also snap to a camera from the data set with the ```Snap to``` button or find the closest camera with ```Snap to closest```. The floating menues also allow you to change the navigation speed. You can use the ```Scaling Modifier``` to control the size of the displayed Gaussians, or show the initial point cloud.

### Running the Network Viewer

After extracting or installing the viewers, you may run the compiled ```SIBR_remoteGaussian_app[_config]``` app in ```<SIBR install dir>/bin```, e.g.: 
```shell
./<SIBR install dir>/bin/SIBR_remoteGaussian_app
```
The network viewer allows you to connect to a running training process on the same or a different machine. If you are training on the same machine and OS, no command line parameters should be required: the optimizer communicates the location of the training data to the network viewer. By default, optimizer and network viewer will try to establish a connection on **localhost** on port **6009**. You can change this behavior by providing matching ```--ip``` and ```--port``` parameters to both the optimizer and the network viewer. If for some reason the path used by the optimizer to find the training data is not reachable by the network viewer (e.g., due to them running on different (virtual) machines), you may specify an override location to the viewer by using ```-s <source path>```. 

<details>
<summary><span style="font-weight: bold;">Primary Command Line Arguments for Network Viewer</span></summary>

  #### --path / -s
  Argument to override model's path to source dataset.
  #### --ip
  IP to use for connection to a running training script.
  #### --port
  Port to use for connection to a running training script. 
  #### --rendering-size 
  Takes two space separated numbers to define the resolution at which network rendering occurs, ```1200``` width by default.
  Note that to enforce an aspect that differs from the input images, you need ```--force-aspect-ratio``` too.
  #### --load_images
  Flag to load source dataset images to be displayed in the top view for each camera.
</details>
<br>

### Running the Real-Time Viewer
Currently, FeatSplat models are not compatible with SIBR's Real-Time Viewer. To visualize an optimized scene we provide a script to stream the rendered points of view to the Network Viewer

First run on a shell with base directory featsplat repository
```shell
python featsplat/stream.py -s <source_path> -m <model_path>
```
and then on a different shell run

```shell
./<SIBR install dir>/bin/SIBR_remoteGaussian_app --rendering-size 1752 1168 --force-aspect-ratio
```

## TODO
- [ ] Explain how to change Features vectors' size
