import os
from torch.utils.cpp_extension import load
import torch

_gather_module = None


def _get_module():
    global _gather_module
    if _gather_module is None:
        src_path = os.path.join(os.path.dirname(__file__), 'gather_attributes_kernel.cu')
        _gather_module = load(
            name='gather_attributes_ext',
            sources=[src_path],
            extra_cuda_cflags=['-O2']
        )
    return _gather_module


def gather_attributes(index, xyz, scaling, rotation, opacity, colors, shs=None):
    module = _get_module()

    tensors = [index, xyz, scaling, rotation, opacity, colors]
    if shs is not None:
        tensors.append(shs)
    for tensor in tensors:
        if tensor is not None:
            assert tensor.is_cuda, "All input tensors must be CUDA tensors"

    if index.dtype != torch.long:
        index = index.to(dtype=torch.long)
    index = index.contiguous()

    outputs = module.gather_attributes_cuda(index, xyz, scaling, rotation, opacity, colors, shs)

    keys = ['xyz', 'scaling', 'rotation', 'opacity', 'colors']
    result = {key: outputs[i] for i, key in enumerate(keys)}
    if shs is not None:
        result['shs'] = outputs[-1]
    return result

__all__ = ['gather_attributes']
