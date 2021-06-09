import torch
import torch.nn as nn


class SphereTracing(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, 
        signed_distance_function, 
        ray_positions, 
        ray_directions, 
        num_iterations, 
        convergence_threshold,
        foreground_masks=None,
        bounding_radius=None,
        *parameters,
    ):
        if foreground_masks is None:
            foreground_masks = torch.all(torch.isfinite(ray_positions), dim=-1, keepdim=True)

        if bounding_radius:
            a = torch.sum(ray_directions * ray_directions, dim=-1, keepdim=True)
            b = 2 * torch.sum(ray_directions * ray_positions, dim=-1, keepdim=True)
            c = torch.sum(ray_positions * ray_positions, dim=-1, keepdim=True) - bounding_radius ** 2
            d = b ** 2 - 4 * a * c
            t = (-b - torch.sqrt(d)) / (2 * a) 
            bounded = d >= 0
            ray_positions = torch.where(bounded, ray_positions + ray_directions * t, ray_positions)
            foreground_masks = foreground_masks & bounded

        # vanilla sphere tracing
        with torch.no_grad():
            for i in range(num_iterations):
                signed_distances = signed_distance_function(ray_positions)
                if i:
                    ray_positions = torch.where(foreground_masks & ~converged, ray_positions + ray_directions * signed_distances, ray_positions)
                else:
                    ray_positions = torch.where(foreground_masks, ray_positions + ray_directions * signed_distances, ray_positions)
                if bounding_radius:
                    bounded = torch.norm(ray_positions, dim=-1, keepdim=True) < bounding_radius
                    foreground_masks = foreground_masks & bounded
                converged = torch.abs(signed_distances) < convergence_threshold
                if torch.all(~foreground_masks | converged):
                    break

        # save tensors for backward pass
        ctx.save_for_backward(ray_positions, ray_directions, foreground_masks, converged)
        ctx.signed_distance_function = signed_distance_function
        ctx.parameters = parameters

        return ray_positions, converged

    @staticmethod
    def backward(ctx, grad_outputs, _):
        
        # restore tensors from forward pass
        ray_positions, ray_directions, foreground_masks, converged = ctx.saved_tensors
        signed_distance_function = ctx.signed_distance_function
        parameters = ctx.parameters

        # compute gradients using implicit differentiation
        # NOTE: Differentiable Volumetric Rendering: https://arxiv.org/abs/1912.07372
        with torch.enable_grad():
            ray_positions = ray_positions.detach()
            ray_positions.requires_grad_(True)
            signed_distances = signed_distance_function(ray_positions)
            grad_positions, = torch.autograd.grad(
                outputs=signed_distances, 
                inputs=ray_positions, 
                grad_outputs=torch.ones_like(signed_distances), 
                retain_graph=True,
            )
            grad_outputs_dot_directions = torch.sum(grad_outputs * ray_directions, dim=-1, keepdim=True)
            grad_positions_dot_directions = torch.sum(grad_positions * ray_directions, dim=-1, keepdim=True)
            # NOTE: avoid division by zero
            grad_positions_dot_directions = torch.where(
                grad_positions_dot_directions > 0,
                torch.max(grad_positions_dot_directions, torch.full_like(grad_positions_dot_directions, +1e-6)),
                torch.min(grad_positions_dot_directions, torch.full_like(grad_positions_dot_directions, -1e-6)),
            )
            grad_outputs = -grad_outputs_dot_directions / grad_positions_dot_directions
            # NOTE: zero gradients of unconverged points 
            grad_outputs = torch.where(converged, grad_outputs, torch.zeros_like(grad_outputs))
            grad_parameters = torch.autograd.grad(
                outputs=signed_distances, 
                inputs=parameters, 
                grad_outputs=grad_outputs, 
                retain_graph=True,
            )

        return (None, None, None, None, None, None, None, *grad_parameters)


def sphere_tracing(
    signed_distance_function, 
    ray_positions, 
    ray_directions, 
    num_iterations, 
    convergence_threshold,
    foreground_masks=None,
    bounding_radius=None,
):
    return SphereTracing.apply(
        signed_distance_function, 
        ray_positions, 
        ray_directions, 
        num_iterations, 
        convergence_threshold,
        foreground_masks,
        bounding_radius,
    )


def compute_shadows(
    signed_distance_function, 
    surface_positions, 
    surface_normals,
    light_directions, 
    num_iterations, 
    convergence_threshold,
    foreground_masks=None,
    bounding_radius=None,
):
    surface_positions, converged = sphere_tracing(
        signed_distance_function=signed_distance_function, 
        ray_positions=surface_positions + surface_normals * 1e-3, 
        ray_directions=light_directions, 
        num_iterations=num_iterations, 
        convergence_threshold=convergence_threshold,
        foreground_masks=foreground_masks,
        bounding_radius=bounding_radius,
    )
    return foreground_masks & converged


def compute_normal(
    signed_distance_function, 
    surface_positions, 
    finite_difference_epsilon=None,
):
    if finite_difference_epsilon:
        finite_difference_epsilon_x = surface_positions.new_tensor([finite_difference_epsilon, 0.0, 0.0])
        finite_difference_epsilon_y = surface_positions.new_tensor([0.0, finite_difference_epsilon, 0.0])
        finite_difference_epsilon_z = surface_positions.new_tensor([0.0, 0.0, finite_difference_epsilon])
        surface_normals_x = signed_distance_function(surface_positions + finite_difference_epsilon_x) - signed_distance_function(surface_positions - finite_difference_epsilon_x)
        surface_normals_y = signed_distance_function(surface_positions + finite_difference_epsilon_y) - signed_distance_function(surface_positions - finite_difference_epsilon_y)
        surface_normals_z = signed_distance_function(surface_positions + finite_difference_epsilon_z) - signed_distance_function(surface_positions - finite_difference_epsilon_z)
        surface_normals = torch.cat((surface_normals_x, surface_normals_y, surface_normals_z), dim=-1)

    else:
        create_graph = surface_positions.requires_grad
        surface_positions.requires_grad_(True)
        with torch.enable_grad():
            signed_distances = signed_distance_function(surface_positions)
            surface_normals, = torch.autograd.grad(
                outputs=signed_distances, 
                inputs=surface_positions, 
                grad_outputs=torch.ones_like(signed_distances),
                create_graph=create_graph,
            )

    surface_normals = nn.functional.normalize(surface_normals, dim=-1)
    return surface_normals


def phong_shading(
    surface_normals, 
    view_directions, 
    light_directions,
    light_ambient_color, 
    light_diffuse_color, 
    light_specular_color, 
    material_ambient_color,
    material_diffuse_color,
    material_specular_color,
    material_emission_color,
    material_shininess,
):
    surface_normals = nn.functional.normalize(surface_normals, dim=-1)
    view_directions = nn.functional.normalize(view_directions, dim=-1)
    light_directions = nn.functional.normalize(light_directions, dim=-1)

    reflected_directions = 2 * surface_normals * torch.sum(light_directions * surface_normals, dim=-1, keepdim=True) - light_directions

    diffuse_coefficients = nn.functional.relu(torch.sum(light_directions * surface_normals, dim=-1, keepdim=True))
    specular_coefficients = nn.functional.relu(torch.sum(reflected_directions * view_directions, dim=-1, keepdim=True)) ** material_shininess

    images = torch.clamp(
        material_emission_color + 
        material_ambient_color * light_ambient_color + 
        material_diffuse_color * light_diffuse_color * diffuse_coefficients + 
        material_specular_color * light_specular_color * specular_coefficients,
        0.0, 1.0
    )

    return images


def cube_mapping(
    surface_normals, 
    view_directions, 
    positive_x_images,
    negative_x_images,
    positive_y_images,
    negative_y_images,
    positive_z_images,
    negative_z_images,
):
    surface_normals = nn.functional.normalize(surface_normals, dim=-1)
    view_directions = nn.functional.normalize(view_directions, dim=-1)

    reflected_directions = 2 * surface_normals * torch.sum(view_directions * surface_normals, dim=-1, keepdim=True) - view_directions

    max_indices = torch.argmax(torch.abs(reflected_directions), dim=-1, keepdim=True)
    max_values = torch.gather(reflected_directions, -1, max_indices)

    texcoords_x, texcoords_y, texcoords_z = torch.unbind(reflected_directions / torch.abs(max_values), dim=-1)

    # OpenGL specifications
    # Chapter 8.13: Cube Map Texture Selection
    # https://www.khronos.org/registry/OpenGL/specs/gl/glspec46.core.pdf

    images = torch.stack((
        positive_x_images,
        negative_x_images,
        positive_y_images,
        negative_y_images,
        positive_z_images,
        negative_z_images,
    ), dim=-3)

    def linear_mapping(inputs, in_min, in_max, out_min, out_max):
        inputs = (inputs - in_min) / (in_max - in_min) * (out_max - out_min) + out_min
        return inputs

    grids = torch.where(
        max_indices == 0,
        torch.where(
            max_values > 0,
            torch.stack((-texcoords_z, texcoords_y, linear_mapping(torch.full_like(texcoords_x, 0), 0, 5, -1, 1)), dim=-1),
            torch.stack(( texcoords_z, texcoords_y, linear_mapping(torch.full_like(texcoords_x, 1), 0, 5, -1, 1)), dim=-1),
        ),
        torch.where(
            max_indices == 1,
            torch.where(
                max_values > 0,
                torch.stack((texcoords_x, -texcoords_z, linear_mapping(torch.full_like(texcoords_y, 2), 0, 5, -1, 1)), dim=-1),
                torch.stack((texcoords_x,  texcoords_z, linear_mapping(torch.full_like(texcoords_y, 3), 0, 5, -1, 1)), dim=-1),
            ),
            torch.where(
                max_values > 0,
                torch.stack(( texcoords_x, texcoords_y, linear_mapping(torch.full_like(texcoords_z, 4), 0, 5, -1, 1)), dim=-1),
                torch.stack((-texcoords_x, texcoords_y, linear_mapping(torch.full_like(texcoords_z, 5), 0, 5, -1, 1)), dim=-1),
            ),
        )
    )

    images = nn.functional.grid_sample(images, grids.unsqueeze(-4)).squeeze(-3)
    images = images.permute(0, 2, 3, 1)
    
    return images
