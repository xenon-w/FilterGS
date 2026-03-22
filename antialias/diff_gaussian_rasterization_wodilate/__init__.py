#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
    use_filter=True,
    ogfs_epsilon_scale=0.2,
    ogfs_epsilon_max=0.2,
    ogfs_energy_floor=0.05,
    ogfs_radius_ratio_min=0.7,
    ogfs_enable_stats=True
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        use_filter,
        float(ogfs_epsilon_scale),
        float(ogfs_epsilon_max),
        float(ogfs_energy_floor),
        float(ogfs_radius_ratio_min),
        bool(ogfs_enable_stats)
    )

class _RasterizeGaussians(torch.autograd.Function):
    _last_stats = None

    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        use_filter,
        ogfs_epsilon_scale,
        ogfs_epsilon_max,
        ogfs_energy_floor,
        ogfs_radius_ratio_min,
        ogfs_enable_stats
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            use_filter,
            raster_settings.debug,
            ogfs_epsilon_scale,
            ogfs_epsilon_max,
            ogfs_energy_floor,
            ogfs_radius_ratio_min,
            ogfs_enable_stats
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, point_id, point_weight_pixel, point_weight, radii, geomBuffer, binningBuffer, imgBuffer, stats = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, point_id, point_weight_pixel, point_weight, radii, geomBuffer, binningBuffer, imgBuffer, stats = _C.rasterize_gaussians(*args)
        _RasterizeGaussians._last_stats = stats
        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, point_id, point_weight_pixel, point_weight

    @staticmethod
    def backward(ctx, grad_out_color, radii, point_id, point_weight_pixel, point_weight):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
        # Add gradient contribution to dL_dmeans3D here
        # print(grad_opacities)
        if grad_colors_precomp.shape[-1] > 3:
            grad_colors_precomp = grad_colors_precomp[:, :3]
        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    ogfs_epsilon_scale: float = 0.2
    ogfs_epsilon_max: float = 0.2
    ogfs_energy_floor: float = 0.05
    ogfs_radius_ratio_min: float = 0.7
    ogfs_enable_stats: bool = True

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings
        self.latest_stats = None

    def compute_radius(self, positions, scales, rotations):
        with torch.no_grad():
            raster_settings = self.raster_settings
            radius = _C.compute_radius(
                positions,
                scales,
                rotations,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                raster_settings.image_height,
                raster_settings.image_width,
                )
        return radius
            
    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(
        self,
        means3D,
        means2D,
        opacities,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
        use_filter=True,
        ogfs_epsilon_scale=None,
        ogfs_epsilon_max=None,
        ogfs_energy_floor=None,
        ogfs_radius_ratio_min=None,
        ogfs_enable_stats=None,
    ):
        
        raster_settings = self.raster_settings
        eps_scale = raster_settings.ogfs_epsilon_scale if ogfs_epsilon_scale is None else ogfs_epsilon_scale
        eps_max = raster_settings.ogfs_epsilon_max if ogfs_epsilon_max is None else ogfs_epsilon_max
        energy_floor = raster_settings.ogfs_energy_floor if ogfs_energy_floor is None else ogfs_energy_floor
        radius_min = raster_settings.ogfs_radius_ratio_min if ogfs_radius_ratio_min is None else ogfs_radius_ratio_min
        enable_stats = raster_settings.ogfs_enable_stats if ogfs_enable_stats is None else ogfs_enable_stats

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        result = rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
            use_filter,
            float(eps_scale),
            float(eps_max),
            float(energy_floor),
            float(radius_min),
            bool(enable_stats)
        )
        stats_tensor = getattr(_RasterizeGaussians, "_last_stats", None)
        self.latest_stats = None
        if stats_tensor is not None and torch.is_tensor(stats_tensor):
            stats_list = stats_tensor.detach().cpu().tolist()
            if len(stats_list) >= 16:
                stats_dict = {
                    'bar_g': float(stats_list[0]),
                    'avg_kv_per_gaussian': float(stats_list[1]),
                    'avg_kv_per_tile': float(stats_list[2]),
                    'num_kv_pairs': float(stats_list[3]),
                    'num_tiles': float(stats_list[4]),
                    'num_gaussians': float(stats_list[5]),
                }
                used_kv_pairs = float(stats_list[6])
                unused_kv_pairs = float(stats_list[7])
                num_kpc_bins = int(stats_list[8])
                stats_dict['used_kv_pairs'] = used_kv_pairs
                stats_dict['unused_kv_pairs'] = unused_kv_pairs
                stats_dict['num_kpc_bins'] = num_kpc_bins
                remainder = stats_list[9:]
                num_tiles = int(stats_dict['num_tiles'])
                # stats layout: 9 + 2*kpc_slots + 2*num_tiles
                kpc_slots = 0
                if len(remainder) >= 2 * num_tiles:
                    kpc_slots = (len(remainder) - 2 * num_tiles) // 2
                if kpc_slots > 0:
                    kpc_part = remainder[:2 * kpc_slots]
                    edges_full = kpc_part[:kpc_slots]
                    hist_full = kpc_part[kpc_slots:]
                    stats_dict['kpc_edges'] = edges_full[:num_kpc_bins]
                    stats_dict['kpc_hist'] = hist_full[:num_kpc_bins]
                    remainder = remainder[2 * kpc_slots:]
                if remainder and num_tiles > 0:
                    if len(remainder) >= num_tiles:
                        stats_dict['tile_ratio'] = remainder[:num_tiles]
                        remainder = remainder[num_tiles:]
                    if len(remainder) >= num_tiles:
                        stats_dict['tile_total_kv'] = remainder[:num_tiles]
                self.latest_stats = stats_dict
        return result
