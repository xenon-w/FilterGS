from torch.utils.cpp_extension import load
import torch
import os

# Load the CUDA extension. If a prebuilt module is available, use it directly.
try:
    import visible_flag_cuda_kernel
    visible_flag_module = visible_flag_cuda_kernel
except ImportError:
    # Otherwise compile it on first use.
    visible_flag_module = load(
        verbose=False,
        name='visible_flag_cuda_kernel',
        sources=[os.path.join(os.path.dirname(__file__), 'visible_flag_kernel.cu')],
        extra_cuda_cflags=['-O2', '--use_fast_math']
    )


def visible_flag_cuda(xyz, camera, padding=0.05, use_optimized=True):
    """
    Compute visibility flags, depth, and projected coordinates on CUDA.

    Args:
        xyz: Tensor with shape [N, 3], must be CUDA.
        camera: Dict that contains key 'full_proj_transform'.
        padding: NDC padding for frustum culling.
        use_optimized: Whether to use the optimized CUDA kernel.

    Returns:
        (valid_flag, depth, p_proj)
    """
    if not isinstance(xyz, torch.Tensor):
        raise ValueError("xyz must be a torch.Tensor")
    if not xyz.is_cuda:
        raise ValueError("xyz must be on GPU")
    if xyz.dim() != 2 or xyz.size(1) != 3:
        raise ValueError("xyz must have shape [N, 3]")
    if not isinstance(camera, dict) or 'full_proj_transform' not in camera:
        raise ValueError("camera must be a dict containing 'full_proj_transform' key")

    proj_matrix = camera['full_proj_transform']
    if not proj_matrix.is_cuda:
        proj_matrix = proj_matrix.cuda()
    if proj_matrix.dim() != 2 or proj_matrix.size(0) != 4 or proj_matrix.size(1) != 4:
        raise ValueError("full_proj_transform must have shape [4, 4]")
    if proj_matrix.dim() == 3 and proj_matrix.size(0) == 1:
        proj_matrix = proj_matrix.squeeze(0)

    try:
        return visible_flag_module.visible_flag_cuda(xyz, proj_matrix, padding, use_optimized)
    except Exception as e:
        raise RuntimeError(f"CUDA computation failed: {str(e)}")


def visible_flag_cuda_direct(xyz, proj_matrix, padding=0.05, use_optimized=True):
    """
    Same as `visible_flag_cuda`, but pass projection matrix directly.
    """
    if not isinstance(xyz, torch.Tensor):
        raise ValueError("xyz must be a torch.Tensor")
    if not xyz.is_cuda:
        raise ValueError("xyz must be on GPU")
    if xyz.dim() != 2 or xyz.size(1) != 3:
        raise ValueError("xyz must have shape [N, 3]")
    if not isinstance(proj_matrix, torch.Tensor):
        raise ValueError("proj_matrix must be a torch.Tensor")
    if not proj_matrix.is_cuda:
        raise ValueError("proj_matrix must be on GPU")
    if proj_matrix.dim() != 2 or proj_matrix.size(0) != 4 or proj_matrix.size(1) != 4:
        raise ValueError("proj_matrix must have shape [4, 4]")

    try:
        return visible_flag_module.visible_flag_cuda(xyz, proj_matrix, padding, use_optimized)
    except Exception as e:
        raise RuntimeError(f"CUDA computation failed: {str(e)}")


def visible_flag_cuda_batch(xyz, camera, padding=0.05, use_optimized=True):
    """
    Batch version for xyz with shape [B, N, 3].
    """
    if xyz.dim() != 3:
        raise ValueError("xyz must have shape [B, N, 3] for batch processing")

    proj_matrix = camera['full_proj_transform']
    if not proj_matrix.is_cuda:
        proj_matrix = proj_matrix.cuda()
    if proj_matrix.dim() != 2 or proj_matrix.size(0) != 4 or proj_matrix.size(1) != 4:
        raise ValueError("full_proj_transform must have shape [4, 4]")

    try:
        return visible_flag_module.visible_flag_cuda_batch(xyz, proj_matrix, padding, use_optimized)
    except Exception as e:
        raise RuntimeError(f"CUDA batch computation failed: {str(e)}")


def visible_flag_cuda_batch_direct(xyz, proj_matrix, padding=0.05, use_optimized=True):
    """
    Batch version with explicit projection matrix.
    """
    if xyz.dim() != 3:
        raise ValueError("xyz must have shape [B, N, 3] for batch processing")
    if not isinstance(proj_matrix, torch.Tensor):
        raise ValueError("proj_matrix must be a torch.Tensor")
    if not proj_matrix.is_cuda:
        raise ValueError("proj_matrix must be on GPU")
    if proj_matrix.dim() != 2 or proj_matrix.size(0) != 4 or proj_matrix.size(1) != 4:
        raise ValueError("proj_matrix must have shape [4, 4]")

    try:
        return visible_flag_module.visible_flag_cuda_batch(xyz, proj_matrix, padding, use_optimized)
    except Exception as e:
        raise RuntimeError(f"CUDA batch computation failed: {str(e)}")


# Performance helper functions.
def benchmark_visible_flag(xyz, camera, padding=0.05, num_runs=100):
    """Benchmark CUDA visibility computation using camera dict input."""
    import time

    if not isinstance(camera, dict) or 'full_proj_transform' not in camera:
        raise ValueError("camera must be a dict containing 'full_proj_transform' key")

    proj_matrix = camera['full_proj_transform']

    # Warm up GPU.
    for _ in range(10):
        _ = visible_flag_cuda(xyz, camera, padding)
    torch.cuda.synchronize()

    # End-to-end timing through Python wrapper.
    start_time = time.time()
    for _ in range(num_runs):
        _ = visible_flag_cuda(xyz, camera, padding)
    torch.cuda.synchronize()
    cuda_time = time.time() - start_time

    # Native kernel benchmark from extension.
    try:
        benchmark_results = visible_flag_module.benchmark_visible_flag(
            xyz, proj_matrix, padding, num_runs
        ).cpu().numpy()
        cuda_kernel_time = benchmark_results[0] / 1000.0  # ms -> s
        cuda_kernel_avg_time = benchmark_results[1] / 1000.0  # ms -> s
        print(f"CUDA kernel benchmark: total={cuda_kernel_time:.4f}s, avg={cuda_kernel_avg_time:.6f}s")
    except Exception as e:
        print(f"CUDA kernel benchmark failed: {e}")
        cuda_kernel_time = None
        cuda_kernel_avg_time = None

    # Optional PyTorch baseline is not implemented in this file.
    pytorch_time = None

    results = {
        'cuda_time': cuda_time,
        'cuda_avg_time': cuda_time / num_runs,
        'cuda_kernel_time': cuda_kernel_time,
        'cuda_kernel_avg_time': cuda_kernel_avg_time,
        'num_runs': num_runs,
        'num_points': xyz.size(0)
    }

    if pytorch_time is not None:
        results['pytorch_time'] = pytorch_time
        results['pytorch_avg_time'] = pytorch_time / num_runs
        results['speedup'] = pytorch_time / cuda_time

    return results


def benchmark_visible_flag_direct(xyz, proj_matrix, padding=0.05, num_runs=100):
    """Benchmark CUDA visibility computation using direct projection matrix input."""
    import time

    # Warm up GPU.
    for _ in range(10):
        _ = visible_flag_cuda_direct(xyz, proj_matrix, padding)
    torch.cuda.synchronize()

    # End-to-end timing through Python wrapper.
    start_time = time.time()
    for _ in range(num_runs):
        _ = visible_flag_cuda_direct(xyz, proj_matrix, padding)
    torch.cuda.synchronize()
    cuda_time = time.time() - start_time

    # Native kernel benchmark from extension.
    try:
        benchmark_results = visible_flag_module.benchmark_visible_flag(
            xyz, proj_matrix, padding, num_runs
        ).cpu().numpy()
        cuda_kernel_time = benchmark_results[0] / 1000.0  # ms -> s
        cuda_kernel_avg_time = benchmark_results[1] / 1000.0  # ms -> s
        print(f"CUDA kernel benchmark: total={cuda_kernel_time:.4f}s, avg={cuda_kernel_avg_time:.6f}s")
    except Exception as e:
        print(f"CUDA kernel benchmark failed: {e}")
        cuda_kernel_time = None
        cuda_kernel_avg_time = None

    return {
        'cuda_time': cuda_time,
        'cuda_avg_time': cuda_time / num_runs,
        'cuda_kernel_time': cuda_kernel_time,
        'cuda_kernel_avg_time': cuda_kernel_avg_time,
        'num_runs': num_runs,
        'num_points': xyz.size(0)
    }
