from torch.utils.cpp_extension import load
import os
import torch

compute_radius_module = load(
    verbose=True,
    name='compute_radius',
    sources=[os.path.join(os.path.dirname(__file__), 'compute_radius_kernel.cu')],
    extra_include_paths=['submodules/mydiffgaussian/third_party/glm'],
    extra_cuda_cflags=['-O2']
)


def compute_visibility_radius(means3d, scales, rotations, projmatrix, viewmatrix,
                              tree_node_index, ancestor_path,
                              focal_x, focal_y, tan_fovx, tan_fovy,
                              radius_max, padding=0.05, radius_filter_threshold=100.0):
    """Wrapper around CUDA kernel that simultaneously执行视锥裁剪、2D半径计算、LOD筛选与大半径过滤。"""
    tensors = (means3d, scales, rotations, projmatrix, viewmatrix, tree_node_index, ancestor_path)
    if not all(t.is_cuda for t in tensors):
        raise ValueError("All input tensors must be CUDA tensors")

    return compute_radius_module.compute_visibility_radius(
        means3d, scales, rotations,
        projmatrix, viewmatrix,
        tree_node_index, ancestor_path,
        float(focal_x), float(focal_y),
        float(tan_fovx), float(tan_fovy),
        float(padding), float(radius_max),
        float(radius_filter_threshold))
