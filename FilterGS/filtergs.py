import torch
import torch.nn as nn
from .cuda.compute_radius import compute_radius_module, compute_visibility_radius
from .cuda.visible_flag import visible_flag_cuda_direct
from .cuda.gather_attributes import gather_attributes
from LoG.model.activation import Activation
import time

class FilterGS(nn.Module):
    def __init__(self, sh_degree=1, xyz_scale=1.):
        super().__init__()
        self.xyz_scale = xyz_scale
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0
        self.activation = Activation()
        self.keys = ['xyz', 'features_dc', 'features_rest', 'scaling', 'rotation', 'opacity']
        self.visibility_flag = None
        self.sh_c0 = 0.28209479177387814

    def prepare(self, rasterizer, camera, tree=None, tree_node_index_gpu=None, ancestor_path_gpu=None,
                radius_max=2.0, padding=0.5, radius_filter_threshold=700.0):
        """Stage 1: GPU visibility filtering for candidate Gaussians."""
        device = self.xyz.device
        use_cuda_timer = self.xyz.is_cuda

        def _time_start():
            if use_cuda_timer:
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                return ev
            return time.perf_counter()

        def _time_end(start_token):
            if use_cuda_timer:
                ev_end = torch.cuda.Event(enable_timing=True)
                ev_end.record()
                ev_end.synchronize()
                return float(start_token.elapsed_time(ev_end))
            return float((time.perf_counter() - start_token) * 1000.0)

        total_start = _time_start()
        stage_tree_start = _time_start()
        if tree_node_index_gpu is not None and ancestor_path_gpu is not None:
            node_index_gpu = tree_node_index_gpu
            ancestor_path_gpu = ancestor_path_gpu
        elif tree is not None and self.xyz.is_cuda:
            node_index_gpu = tree.node_index.to(device=device, dtype=torch.int32)
            ancestor_path_gpu = tree.ancestor_path.to(device=device, dtype=torch.int32)
        else:
            node_index_gpu, ancestor_path_gpu = None, None
        t_tree_ms = _time_end(stage_tree_start)

        # CUDA visibility filtering (frustum test + LOD + radius filtering).
        stage_vis_start = _time_start()
        vis_fallback = False
        all_mask, all_radius2d, all_depth = self._compute_visibility_radius_cuda(
            rasterizer, camera, node_index_gpu, ancestor_path_gpu,
            radius_max, padding, radius_filter_threshold
        )
        t_vis_ms = _time_end(stage_vis_start)

        # Pack visibility data
        stage_pack_start = _time_start()
        index_all = torch.nonzero(all_mask, as_tuple=True)[0]
        radius2d = torch.masked_select(all_radius2d, all_mask)
        depth_visible = torch.masked_select(all_depth, all_mask) if all_depth.numel() > 0 else all_depth
        self.visibility_flag = {
            'index': index_all,
            'flag': all_mask,
            'radius2d': radius2d,
            'depth': depth_visible,
            'raw_inputs': {
                'visible_index': index_all,
                'node_index_gpu': node_index_gpu,
                'ancestor_path_gpu': ancestor_path_gpu,
                'base_attrs': {
                    'xyz': self.xyz,
                    'scaling': self.scaling,
                    'rotation': self.rotation,
                    'opacity': self.opacity,
                    'colors': self.colors,
                    'shs': getattr(self, 'shs', None)
                }
            }
        }
        t_pack_ms = _time_end(stage_pack_start)
        t_total_ms = _time_end(total_start)
        self.last_prepare_timing = {
            "T_tree_ms": t_tree_ms,
            "T_vis_ms": t_vis_ms,
            "T_pack_ms": t_pack_ms,
            "T_prepare_total_ms": t_total_ms,
            "vis_fallback": vis_fallback,
            "num_visible": int(index_all.numel()),
        }
        return len(index_all)

    def get_all(self, camera):
        """Stage 2: gather filtered attributes in batch."""
        if self.visibility_flag is None:
            raise RuntimeError("Must call prepare() before get_all()")

        visible_index = self.visibility_flag['index']
        device = self.xyz.device

        # Gather attributes with CUDA.
        if visible_index.device != device:
            visible_index = visible_index.to(device=device, non_blocking=True)

        use_cuda_gather = self.xyz.is_cuda
        if use_cuda_gather:
            shs_tensor = getattr(self, 'shs', None) if self.max_sh_degree > 0 else None
            ret = gather_attributes(
                visible_index.contiguous(),
                self.xyz,
                self.scaling,
                self.rotation,
                self.opacity,
                self.colors,
                shs_tensor)
        else:
            # CPU fallback
            ret = {}
            for key in ['xyz', 'scaling', 'rotation', 'opacity', 'colors']:
                val = getattr(self, key)
                ret[key] = torch.index_select(val, 0, visible_index)
            if hasattr(self, 'shs'):
                ret['shs'] = torch.index_select(self.shs, 0, visible_index)

        # Apply parameter activations.
        ret = self.activation.activate_root_return(ret, camera, self.active_sh_degree)
        ret['scaling'] = ret['scaling']

        return ret

    def render(self, camera, rasterizer, background=None, enable_stats=True):
        """Stage 3: preprocess and render."""
        if self.visibility_flag is None:
            raise RuntimeError("Must call prepare() before render()")

        # For visual parity with LoG val flow, use activated scales+rotations path.
        # The current precomputed cov3d in prepare_preprocess_inputs is a simplified
        # diagonal approximation and can introduce rendering artifacts.
        activated = self.get_all(camera)

        # Prepare rasterizer inputs.
        xyz = activated['xyz']
        opacity = activated['opacity']
        colors = activated['colors']
        scales = activated['scaling']
        rotations = activated['rotation']
        cov3D = None

        if opacity.dim() == 1:
            opacity = opacity.unsqueeze(-1)

        # Guard against invalid/empty inputs that can trigger CUDA invalid-argument
        # in rasterizer kernels (e.g. launching with grid size 0).
        finite_mask = (
            torch.isfinite(xyz).all(dim=-1)
            & torch.isfinite(scales).all(dim=-1)
            & torch.isfinite(rotations).all(dim=-1)
            & torch.isfinite(opacity).all(dim=-1)
            & torch.isfinite(colors).all(dim=-1)
        )
        finite_mask = finite_mask & (scales.min(dim=-1).values > 0.0) & (opacity[:, 0] > 0.0)
        if not torch.all(finite_mask):
            xyz = xyz[finite_mask]
            opacity = opacity[finite_mask]
            colors = colors[finite_mask]
            scales = scales[finite_mask]
            rotations = rotations[finite_mask]

        if xyz.shape[0] == 0:
            rs = rasterizer.raster_settings
            h, w = int(rs.image_height), int(rs.image_width)
            if background is None:
                bg = torch.zeros(3, dtype=xyz.dtype, device=xyz.device)
            else:
                bg = background.to(device=xyz.device, dtype=xyz.dtype).reshape(-1)
                if bg.numel() >= 3:
                    bg = bg[:3]
                elif bg.numel() == 1:
                    bg = bg.repeat(3)
                else:
                    bg = torch.zeros(3, dtype=xyz.dtype, device=xyz.device)
            rendered_image = bg[:, None, None].expand(3, h, w).contiguous()
            return {
                "render": rendered_image,
                "timing": {"T_raster_ms": 0.0},
            }

        name_args = {
            'means3D': xyz,
            'means2D': torch.zeros_like(xyz, device=xyz.device, dtype=xyz.dtype),
            'shs': None,
            'colors_precomp': colors,
            'opacities': opacity
        }

        name_args['scales'] = scales
        name_args['rotations'] = rotations

        # Run rasterization.
        name_args['ogfs_enable_stats'] = bool(enable_stats)
        if xyz.is_cuda:
            ev_raster_start = torch.cuda.Event(enable_timing=True)
            ev_raster_end = torch.cuda.Event(enable_timing=True)
            ev_raster_start.record()
        render_outputs = rasterizer(**name_args)
        if xyz.is_cuda:
            ev_raster_end.record()
            ev_raster_end.synchronize()
            t_raster_ms = float(ev_raster_start.elapsed_time(ev_raster_end))
        else:
            t_raster_ms = 0.0
        rendered_image = render_outputs[0]

        ret = {
            "render": rendered_image[:3],
            "timing": {"T_raster_ms": t_raster_ms},
        }
        return ret

    def _visible_flag_by_camera_cuda(self, xyz, camera, padding=0.05):
        """CUDA visibility check."""
        full_proj_transform = camera['full_proj_transform']
        valid_flag, depth, p_proj = visible_flag_cuda_direct(xyz, full_proj_transform, padding=padding)
        return valid_flag, depth, p_proj

    @staticmethod
    def _visible_flag_by_camera_torch(xyz, camera, padding=0.05):
        full_proj_transform = camera['full_proj_transform']
        xyz1 = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
        xyz1RTK = xyz1 @ full_proj_transform
        pw = 1.0 / (xyz1RTK[..., 3:4] + 1e-7)
        p_proj = xyz1RTK[:, :3] * pw
        depth = p_proj[:, 2]
        valid_flag = (depth > 0.0) & (depth < 1.0) & \
            (p_proj[:, 0] > -1.0 - padding) & (p_proj[:, 0] < 1.0 + padding) & \
            (p_proj[:, 1] > -1.0 - padding) & (p_proj[:, 1] < 1.0 + padding)
        return valid_flag, depth, p_proj

    def _compute_visibility_radius_cuda(self, rasterizer, camera, tree_node_index_gpu, ancestor_path_gpu,
                                        radius_max, padding, radius_filter_threshold=100.0):
        """CUDA visibility and radius filtering (LoG-pro style)."""
        rs = rasterizer.raster_settings
        device = self.xyz.device

        tanfovx = float(rs.tanfovx)
        tanfovy = float(rs.tanfovy)
        image_width = float(rs.image_width)
        image_height = float(rs.image_height)
        focal_x = image_width / (2.0 * tanfovx)
        focal_y = image_height / (2.0 * tanfovy)
        radius_max_value = float(radius_max)

        proj_matrix = rs.projmatrix.to(device=device, dtype=torch.float32).contiguous()
        view_matrix = rs.viewmatrix.to(device=device, dtype=torch.float32).contiguous()

        scaling = self.scaling.detach().to(device=device).contiguous()
        rotation = self.rotation.detach().to(device=device).contiguous()

        # If tree buffers are not provided, fallback to frustum + radius filtering.
        if tree_node_index_gpu is None or ancestor_path_gpu is None:
            # Match LoG val behavior for visibility: robust torch frustum check.
            valid_flag, depth, _ = self._visible_flag_by_camera_torch(self.xyz, camera, padding=padding)
            radius2d = compute_radius_module.compute_radius(
                self.xyz.detach().contiguous(),
                scaling,
                rotation,
                proj_matrix,
                view_matrix,
                focal_x,
                focal_y,
                tanfovx,
                tanfovy
            )
            # In no-tree mode, do not drop large projected gaussians by radius_max.
            mask = valid_flag
            if radius_filter_threshold > 0:
                mask = mask & (radius2d <= radius_filter_threshold)
            return mask, radius2d, depth

        # Call the full CUDA kernel (requires compute_visibility_radius export).
        mask, radius2d, depth = compute_visibility_radius(
            self.xyz.detach().contiguous(),
            scaling,
            rotation,
            proj_matrix,
            view_matrix,
            tree_node_index_gpu,
            ancestor_path_gpu,
            focal_x,
            focal_y,
            tanfovx,
            tanfovy,
            radius_max_value,
            padding,
            radius_filter_threshold
        )
        return mask, radius2d, depth

    def register_by_pointcloud(self, xyz, colors, scales, init_opacity, **init_ply):
        """Register point cloud tensors to the model."""
        print(f'[{self.__class__.__name__}] {self.log_radius(scales)}')
        scales = torch.clamp(scales, min=scales.mean()/4, max=scales.mean()*4)
        print(f'[{self.__class__.__name__}] -> {self.log_radius(scales)}')

        scaling = self.activation.scaling_inverse_activation(scales)[:, None].repeat(1, 3)
        colors = self.activation.rgb_inverse(colors)

        # add sh
        if self.max_sh_degree > 0:
            features = torch.zeros((colors.shape[0], (self.max_sh_degree + 1) ** 2 - 1, 3), dtype=torch.float32)
            shs = features
        xyz = xyz
        opacity = torch.ones_like(xyz[:, :1]) * init_opacity
        opacity = self.activation.opacity_inverse_activation(opacity)
        rotation = self.init_rotation(xyz.shape[0], xyz.device)

        # Add ground points
        if 'height' in init_ply:
            local_min, local_max = xyz.min(dim=0), xyz.max(dim=0)
            xyz_ground, colors_ground, scaling_ground, opacity_ground = self.create_from_ground(
                local_min, local_max, init_ply['init_step'], init_ply['height'], init_ply['ground_opacity']
            )
            rotation_ground = self.init_rotation(xyz_ground.shape[0], xyz.device)
            print(f'[{self.__class__.__name__}] add {xyz_ground.shape[0]} ground points')
            xyz = torch.cat([xyz, xyz_ground], dim=0)
            opacity = torch.cat([opacity, opacity_ground], dim=0)
            colors = torch.cat([colors, colors_ground], dim=0)
            scaling_ground = self.activation.scaling_inverse_activation(scaling_ground)
            scaling = torch.cat([scaling, scaling_ground], dim=0)
            rotation = torch.cat([rotation, rotation_ground], dim=0)
            if self.max_sh_degree > 0:
                shs_ground = torch.zeros((xyz_ground.shape[0], *shs.shape[1:]), dtype=torch.float32)
                shs = torch.cat([shs, shs_ground], dim=0)

        self.register_buffer('scaling', scaling)
        self.register_buffer('colors', colors)
        self.register_buffer('xyz', xyz)
        self.register_buffer('opacity', opacity)
        self.register_buffer('rotation', rotation)
        self.keys.extend(['scaling', 'colors', 'xyz', 'opacity', 'rotation'])
        if self.max_sh_degree > 0:
            self.register_buffer('shs', shs)
            self.keys.append('shs')

    @staticmethod
    def create_from_ground(local_min, local_max, init_step, height, init_opacity=0.9, padding=0.05):
        """Create ground points."""
        x = torch.arange(local_min[0][0] - padding, local_max[0][0] + padding, init_step)
        y = torch.arange(local_min[0][1] - padding, local_max[0][1] + padding, init_step)
        x, y = torch.meshgrid(x, y)
        xyz = torch.stack((x, y), axis=-1).reshape(-1, 2)
        xyz = torch.cat([xyz, torch.zeros((xyz.shape[0], 1)) + height], dim=1)
        colors = torch.zeros_like(xyz) + 0.5
        scaling = torch.zeros_like(xyz) + init_step
        scaling[:, 2] = init_step * 0.1
        opacity = torch.zeros((xyz.shape[0], 1)) + init_opacity
        return xyz, colors, scaling, opacity

    def init_rotation(self, num_points, device):
        """Initialize quaternion rotations."""
        rot = torch.zeros((num_points, 4), dtype=torch.float32, device=device)
        rot[:, 0] = 1.
        return rot

    def log_radius(self, scales):
        """Format radius statistics for logging."""
        return f'scales: [{scales.min().item():.4f}~{scales.mean().item():.4f}~{scales.max().item():.4f}]'
