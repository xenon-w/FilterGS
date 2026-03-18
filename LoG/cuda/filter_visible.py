from torch.utils.cpp_extension import load
import os

_filter_visible_module = None

def _load_module():
    global _filter_visible_module
    if _filter_visible_module is None:
        src_path = os.path.join(os.path.dirname(__file__), 'filter_visible_kernel.cu')
        _filter_visible_module = load(
            name='filter_visible_cuda_ext_v2',
            sources=[src_path],
            verbose=True
        )
    return _filter_visible_module


def filter_visible(all_visible_nodes, radius2d, radius_max, tree_node_index, ancestor_path):
    module = _load_module()
    return module.filter_visible_cuda(all_visible_nodes, radius2d, radius_max, tree_node_index, ancestor_path)

__all__ = ['filter_visible']
