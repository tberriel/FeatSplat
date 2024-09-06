import argparse
import os
import time
from typing import Tuple

#import imageio
import nerfview
import torch
import viser
from argparse import  Namespace

from gsplat.rendering import rasterization
from scene.feat_gaussian_model import FeatGaussianModel

def load_gaussian(path):
    with open(os.path.join(path,"cfg_args"), 'r') as f:
        cfg = f.read()
    args = eval(cfg)
    
    pc = FeatGaussianModel(args)
    pc.load_ply(os.path.join(path, "point_cloud", "iteration_" + str(30000),"point_cloud.ply"))
    ckpt = {
        "means":pc.get_xyz,
        "quats":pc.get_rotation,
        "scales": pc.get_scaling,
        "opacities": pc.get_opacity.flatten(),
        "features": pc.get_features,
        "mlp": pc.mlp
    }
    return ckpt

def mlp_forward(projected_features, camera_pos, mlp):
        """ 
        - Input is n_latentsxHxW
        - Output is 3xHxW
        """
        _, h, w = projected_features.shape
        x = projected_features.flatten(1,2).permute(1,0)
        embeddings = []
        camera_pos = camera_pos[None,...].repeat((h*w, 1))
        embeddings.append(camera_pos)
            
        umap = torch.linspace(-1, 1, w, device = projected_features.device)
        vmap = torch.linspace(-1, 1, h, device = projected_features.device)
        umap, vmap = torch.meshgrid(umap, vmap, indexing='xy')
        points_2d = torch.stack((umap, vmap), -1).float()
        p_embedding =  points_2d.flatten(0,1)
        embeddings.append(p_embedding)

        x = torch.cat([x]+embeddings, axis=-1 )

        rendered_image = mlp(x)[None].reshape((h,w,3))
        rendered_image = torch.sigmoid(rendered_image)
        return rendered_image

def main(args):
    torch.manual_seed(42)
    device = torch.device("cuda")

    assert args.ckpt is not None
    means, quats, scales, opacities, features =  [], [], [], [], []
    for ckpt_path in args.ckpt:
        ckpt = load_gaussian(ckpt_path)
        means.append(ckpt["means"])
        quats.append(ckpt["quats"])
        scales.append(ckpt["scales"])
        opacities.append(ckpt["opacities"])
        features.append(ckpt["features"])

    mlp = ckpt["mlp"]
    means = torch.cat(means, dim=0)
    quats = torch.cat(quats, dim=0)
    scales = torch.cat(scales, dim=0)
    opacities = torch.cat(opacities, dim=0)
    features = torch.cat(features, dim=0)

    print("Number of Gaussians:", len(means))

    # register and open viewer
    @torch.no_grad()
    def viewer_render_fn(camera_state: nerfview.CameraState, img_wh: Tuple[int, int]):
        width, height = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(camera_state.c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = c2w.inverse()
        camera_center = c2w[:3, 3]

        latent_image, render_alphas, meta = rasterization(
            means,  # [N, 3]
            quats,  # [N, 4]
            scales,  # [N, 3]
            opacities,  # [N]
            features[None],  # [N, D]
            viewmat[None],  # [1, 4, 4]
            K[None],  # [1, 3, 3]
            width,
            height,
            # this is to speedup large-scale rendering by skipping far-away Gaussians.
            #radius_clip=3,
        )
        
        render_rgbs = mlp_forward(latent_image[0].permute(2,0,1), camera_center, mlp)
        return render_rgbs.cpu().numpy()

    server = viser.ViserServer(port=args.port, verbose=False)

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (2.4292,  2.2891, -1.7378)
        client.camera.up_direction =(0,0,-1)

    _ = nerfview.Viewer(
        server=server,
        render_fn=viewer_render_fn,
        mode="rendering",
    )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="results/", help="where to dump outputs"
    )
    parser.add_argument(
        "--scene_grid", type=int, default=1, help="repeat the scene into a grid of NxN"
    )
    parser.add_argument(
        "--ckpt", type=str, nargs="+", default=None, help="path to the .pt file"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="port for the viewer server"
    )
    parser.add_argument(
        "--backend", type=str, default="gsplat", help="gsplat, gsplat_legacy, inria"
    )
    args = parser.parse_args()
    assert args.scene_grid % 2 == 1, "scene_grid must be odd"
    main(args)
    #cli(main, args, verbose=True)
