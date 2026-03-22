import os
from torch.utils.cpp_extension import load

compute_radius_module = load(
    verbose=True,
    name="compute_radius",
    sources=[os.path.join(os.path.dirname(__file__), "compute_radius_kernel.cu")],
    extra_include_paths=["submodules/mydiffgaussian/third_party/glm"],
    extra_cuda_cflags=["-O2"],
)
