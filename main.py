
import torch
import torch.nn as nn
import torchvision
import numpy as np
import skimage.io as io
import argparse
import random

import renderers
import signed_distance_functions as sdf
import constructive_solid_geometries as csg


def main(args):

    # ---------------- camera matrix ---------------- #

    camera_matrices = torch.tensor(
        [[[[
            [args.width / 2.0, 0.0, args.width / 2.0], 
            [0.0, args.height / 2.0, args.height / 2.0], 
            [0.0, 0.0, 1.0],
        ]]]], 
        device=args.device,
    )

    # ---------------- camera position ---------------- #

    camera_positions = torch.tensor(
        [[[[
            2.0 * np.cos(-np.pi / 5.0) * np.sin(-np.pi / 4.0),
            2.0 * np.sin(-np.pi / 5.0),
            2.0 * np.cos(-np.pi / 5.0) * np.cos(-np.pi / 4.0),
        ]]]], 
        device=args.device,
    )

    # ---------------- camera rotation ---------------- #

    target_positions = torch.zeros_like(camera_positions)
    up_directions = torch.tensor([[[[0.0, 1.0, 0.0]]]], device=args.device)

    camera_z_axes = target_positions - camera_positions
    camera_x_axes = torch.cross(up_directions, camera_z_axes, dim=-1)
    camera_y_axes = torch.cross(camera_z_axes, camera_x_axes, dim=-1)
    camera_rotations = torch.stack((camera_x_axes, camera_y_axes, camera_z_axes), dim=-1)
    camera_rotations = nn.functional.normalize(camera_rotations, dim=-2)

    # ---------------- directional light ---------------- #

    light_directions = torch.tensor([[[[1.0, 1.0, 1.0]]]], device=args.device)

    # ---------------- ray marching ---------------- #
    
    y_positions = torch.arange(args.height, dtype=camera_matrices.dtype, device=args.device)
    x_positions = torch.arange(args.width, dtype=camera_matrices.dtype, device=args.device)
    y_positions, x_positions = torch.meshgrid(y_positions, x_positions)
    z_positions = torch.ones_like(y_positions)
    ray_positions = torch.stack((x_positions, y_positions, z_positions), dim=-1)
    ray_positions = torch.einsum("b...mn,hwn->bhwm", torch.inverse(camera_matrices),  ray_positions)
    ray_positions = torch.einsum("b...mn,bhwn->bhwm", camera_rotations, ray_positions) + camera_positions
    ray_directions = nn.functional.normalize(ray_positions - camera_positions, dim=-1)

    # ---------------- rendering ---------------- #

    signed_distance_functions = [
        sdf.sphere(1.0),
        sdf.rounding(sdf.box(0.7), 0.1),
        csg.smooth_union(sdf.sphere(1.0), sdf.rounding(sdf.box(0.7), 0.1), 0.05),
        csg.smooth_intersection(sdf.sphere(1.0), sdf.rounding(sdf.box(0.7), 0.1), 0.05),
        csg.smooth_subtraction(sdf.sphere(1.0), sdf.rounding(sdf.box(0.7), 0.1), 0.05),
        csg.smooth_union(sdf.translation(sdf.sphere(0.5), torch.tensor([0.0, -0.2, 0.0], device=args.device)), sdf.rounding(sdf.box(torch.tensor([0.7, 0.1, 0.7], device=args.device)), 0.1), 0.05),
        csg.smooth_subtraction(sdf.translation(sdf.sphere(0.5), torch.tensor([0.0, -0.2, 0.0], device=args.device)), sdf.rounding(sdf.box(torch.tensor([0.7, 0.1, 0.7], device=args.device)), 0.1), 0.05),
        sdf.torus(1.0, 0.25),
        sdf.twist(sdf.torus(1.0, 0.25), 1.0),
        sdf.bend(sdf.rounding(sdf.box(torch.tensor([1.0, 0.1, 0.5], device=args.device)), 0.1), 1.0),
        sdf.rotation(sdf.infinite_repetition(sdf.sphere(0.5), 2.0), camera_rotations),
        sdf.rotation(sdf.infinite_repetition(sdf.rounding(sdf.box(0.35), 0.1), 2.0), camera_rotations),
    ]

    with torch.no_grad():

        grid_images = []        

        for signed_distance_function in signed_distance_functions:

            surface_positions, converged = renderers.sphere_tracing(
                signed_distance_function=signed_distance_function, 
                ray_positions=ray_positions, 
                ray_directions=ray_directions, 
                num_iterations=1000, 
                convergence_threshold=1e-3,
                bounding_radius=0.0,
            )
            surface_normals = renderers.compute_normal(
                signed_distance_function=signed_distance_function, 
                surface_positions=surface_positions,
                foreground_masks=converged,
            )
    
            images = renderers.phong_shading(
                surface_normals=surface_normals, 
                view_directions=camera_positions - surface_positions, 
                light_directions=-light_directions, 
                light_ambient_color=torch.ones(1, 1, 3, device=args.device),
                light_diffuse_color=torch.ones(1, 1, 3, device=args.device), 
                light_specular_color=torch.ones(1, 1, 3, device=args.device), 
                material_ambient_color=torch.full((1, 1, 3), 0.1, device=args.device) + (torch.rand(1, 1, 3, device=args.device) * 2 - 1) * 0.04,
                material_diffuse_color=torch.full((1, 1, 3), 0.7, device=args.device) + (torch.rand(1, 1, 3, device=args.device) * 2 - 1) * 0.16,
                material_specular_color=torch.full((1, 1, 3), 0.2, device=args.device),
                material_emission_color=torch.zeros(1, 1, 3, device=args.device),
                material_shininess=64.0,
            )
            images = torch.where(converged, images, torch.zeros_like(images))
                
            grid_images.append(images.permute(0, 3, 1, 2))

        torchvision.utils.save_image(torch.cat(grid_images), f"csg.png", nrow=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sphere Tracing")
    parser.add_argument("--width", type=str, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
