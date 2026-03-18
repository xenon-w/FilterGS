import os
import torch
from torch.utils.cpp_extension import load

_prepare_inputs_module = None


def _load_module():
    global _prepare_inputs_module
    if _prepare_inputs_module is None:
        src_path = os.path.join(os.path.dirname(__file__), 'prepare_preprocess_inputs_kernel.cu')
        _prepare_inputs_module = load(
            name='prepare_preprocess_inputs_ext',
            sources=[src_path],
            extra_cuda_cflags=['-O2']
        )
    return _prepare_inputs_module


def prepare_preprocess_inputs(visible_index, base_attrs, c0, camera_center, active_sh_degree):
    module = _load_module()

    shs = base_attrs.get('shs', None)

    tensors = [visible_index,
               base_attrs['xyz'],
               base_attrs['scaling'],
               base_attrs['rotation'],
               base_attrs['opacity'],
               base_attrs['colors'],
               camera_center]
    if shs is not None:
        tensors.append(shs)

    device = base_attrs['xyz'].device
    for tensor in tensors:
        assert tensor is not None and tensor.is_cuda, "All tensors must reside on CUDA device"
        assert tensor.device == device, "All tensors must be on the same device"

    visible_index = visible_index.to(dtype=torch.long, non_blocking=True).contiguous()
    camera_center = camera_center.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()

    outputs = module.prepare_preprocess_inputs_cuda(
        visible_index,
        base_attrs['xyz'],
        base_attrs['scaling'],
        base_attrs['rotation'],
        base_attrs['opacity'],
        base_attrs['colors'],
        shs,
        float(c0),
        int(active_sh_degree),
        camera_center
    )

    position, scaling, rotation, opacity, colors, cov3d = outputs

    return {
        'position': position,
        'scaling': scaling,
        'rotation': rotation,
        'opacity': opacity,
        'colors_rgb': colors,
        'cov3d': cov3d
    }

__all__ = ['prepare_preprocess_inputs']
