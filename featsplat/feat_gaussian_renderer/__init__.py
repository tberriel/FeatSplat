import torch
from scene.gaussian_model import GaussianModel
from gsplat.rendering import rasterization

def rasterize(viewpoint_camera, pc : GaussianModel):
    means = pc.get_xyz
    quats = pc.get_rotation
    scales = pc.get_scaling
    opacities = pc.get_opacity.flatten()
    features = pc.get_features
    viewmats = viewpoint_camera.world_view_transform.transpose(0,1)
    Ks = viewpoint_camera.intrinsic_matrix
    width=int(viewpoint_camera.image_width)
    height=int(viewpoint_camera.image_height)

    # render
    latent_image, alphas, meta = rasterization(
       means, quats, scales, opacities, features[None], viewmats[None], Ks[None], width, height, packed=False
    )
    return latent_image, alphas, meta 

def render(viewpoint_camera, pc : GaussianModel, features_splatting = False):

    latent_image, alphas, meta = rasterize(viewpoint_camera, pc)

    if features_splatting:
        rendered_image, segmentation_image = pc.nn_forward(latent_image[0].permute(2,0,1),viewpoint_camera.camera_center, viewpoint_camera.camera_rot, None)
    else:
        rendered_image, segmentation_image = latent_image, None
    return {"render": rendered_image,
            "segmentation": None,
            "viewspace_points":  meta["means2d"],
            "visibility_filter" : meta["radii"][0] > 0,
            #"visibility_filter" : meta["gaussian_ids"],
            "radii": meta["radii"][0]}