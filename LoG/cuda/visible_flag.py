from torch.utils.cpp_extension import load
import torch
import os

# 加载CUDA扩展
try:
    # 尝试直接导入已编译的扩展
    import visible_flag_cuda_kernel
    visible_flag_module = visible_flag_cuda_kernel
except ImportError:
    # 如果没有已编译的扩展，则编译
    visible_flag_module = load(
        verbose=False,
        name='visible_flag_cuda_kernel',
        sources=[os.path.join(os.path.dirname(__file__), 'visible_flag_kernel.cu')],
        extra_cuda_cflags=['-O2', '--use_fast_math']
    )

def visible_flag_cuda(xyz, camera, padding=0.05, use_optimized=True):
    """
    CUDA加速的可见性标志计算
    
    Args:
        xyz (torch.Tensor): 3D坐标张量 [N, 3]，必须在GPU上
        camera (dict): 相机参数字典，必须包含 'full_proj_transform' 键
        padding (float): 边界填充值，默认0.05
        use_optimized (bool): 是否使用优化版本，默认True
    
    Returns:
        tuple: (valid_flag, depth, p_proj)
            - valid_flag (torch.Tensor): 可见性标志 [N]
            - depth (torch.Tensor): 深度值 [N] 
            - p_proj (torch.Tensor): 投影坐标 [N, 3]
    
    Raises:
        ValueError: 如果输入参数无效
        RuntimeError: 如果CUDA计算失败
    """
    # 输入检查
    if not isinstance(xyz, torch.Tensor):
        raise ValueError("xyz must be a torch.Tensor")
    
    if not xyz.is_cuda:
        raise ValueError("xyz must be on GPU")
    
    if xyz.dim() != 2 or xyz.size(1) != 3:
        raise ValueError("xyz must have shape [N, 3]")
    
    if not isinstance(camera, dict) or 'full_proj_transform' not in camera:
        raise ValueError("camera must be a dict containing 'full_proj_transform' key")
    
    proj_matrix = camera['full_proj_transform']
    
    # 确保投影矩阵在GPU上
    if not proj_matrix.is_cuda:
        proj_matrix = proj_matrix.cuda()
    
    # 检查投影矩阵形状
    if proj_matrix.dim() != 2 or proj_matrix.size(0) != 4 or proj_matrix.size(1) != 4:
        raise ValueError("full_proj_transform must have shape [4, 4]")
    
    # 处理squeeze情况（如果camera是batch格式）
    if proj_matrix.dim() == 3 and proj_matrix.size(0) == 1:
        proj_matrix = proj_matrix.squeeze(0)
    
    try:
        # 调用CUDA kernel
        valid_flag, depth, p_proj = visible_flag_module.visible_flag_cuda(
            xyz, proj_matrix, padding, use_optimized
        )
        
        return valid_flag, depth, p_proj
        
    except Exception as e:
        raise RuntimeError(f"CUDA computation failed: {str(e)}")

def visible_flag_cuda_direct(xyz, proj_matrix, padding=0.05, use_optimized=True):
    """
    CUDA加速的可见性标志计算（直接传递投影矩阵）
    
    Args:
        xyz (torch.Tensor): 3D坐标张量 [N, 3]，必须在GPU上
        proj_matrix (torch.Tensor): 4x4投影矩阵，必须在GPU上
        padding (float): 边界填充值，默认0.05
        use_optimized (bool): 是否使用优化版本，默认True
    
    Returns:
        tuple: (valid_flag, depth, p_proj)
            - valid_flag (torch.Tensor): 可见性标志 [N]
            - depth (torch.Tensor): 深度值 [N] 
            - p_proj (torch.Tensor): 投影坐标 [N, 3]
    """
    # 输入检查
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
        # 调用CUDA kernel
        valid_flag, depth, p_proj = visible_flag_module.visible_flag_cuda(
            xyz, proj_matrix, padding, use_optimized
        )
        
        return valid_flag, depth, p_proj
        
    except Exception as e:
        raise RuntimeError(f"CUDA computation failed: {str(e)}")

def visible_flag_cuda_batch(xyz, camera, padding=0.05, use_optimized=True):
    """
    批量处理版本的CUDA可见性标志计算
    
    Args:
        xyz (torch.Tensor): 3D坐标张量 [B, N, 3]，必须在GPU上
        camera (dict): 相机参数字典，必须包含 'full_proj_transform' 键
        padding (float): 边界填充值，默认0.05
        use_optimized (bool): 是否使用优化版本，默认True
    
    Returns:
        tuple: (valid_flag, depth, p_proj)
            - valid_flag (torch.Tensor): 可见性标志 [B, N]
            - depth (torch.Tensor): 深度值 [B, N] 
            - p_proj (torch.Tensor): 投影坐标 [B, N, 3]
    """
    if xyz.dim() != 3:
        raise ValueError("xyz must have shape [B, N, 3] for batch processing")
    
    batch_size = xyz.size(0)
    proj_matrix = camera['full_proj_transform']
    
    # 确保投影矩阵在GPU上
    if not proj_matrix.is_cuda:
        proj_matrix = proj_matrix.cuda()
    
    # 检查投影矩阵形状
    if proj_matrix.dim() != 2 or proj_matrix.size(0) != 4 or proj_matrix.size(1) != 4:
        raise ValueError("full_proj_transform must have shape [4, 4]")
    
    try:
        # 调用CUDA kernel的批量处理函数
        valid_flag, depth, p_proj = visible_flag_module.visible_flag_cuda_batch(
            xyz, proj_matrix, padding, use_optimized
        )
        
        return valid_flag, depth, p_proj
        
    except Exception as e:
        raise RuntimeError(f"CUDA batch computation failed: {str(e)}")

def visible_flag_cuda_batch_direct(xyz, proj_matrix, padding=0.05, use_optimized=True):
    """
    批量处理版本的CUDA可见性标志计算（直接传递投影矩阵）
    
    Args:
        xyz (torch.Tensor): 3D坐标张量 [B, N, 3]，必须在GPU上
        proj_matrix (torch.Tensor): 4x4投影矩阵，必须在GPU上
        padding (float): 边界填充值，默认0.05
        use_optimized (bool): 是否使用优化版本，默认True
    
    Returns:
        tuple: (valid_flag, depth, p_proj)
            - valid_flag (torch.Tensor): 可见性标志 [B, N]
            - depth (torch.Tensor): 深度值 [B, N] 
            - p_proj (torch.Tensor): 投影坐标 [B, N, 3]
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
        # 调用CUDA kernel的批量处理函数
        valid_flag, depth, p_proj = visible_flag_module.visible_flag_cuda_batch(
            xyz, proj_matrix, padding, use_optimized
        )
        
        return valid_flag, depth, p_proj
        
    except Exception as e:
        raise RuntimeError(f"CUDA batch computation failed: {str(e)}")

# 性能测试函数
def benchmark_visible_flag(xyz, camera, padding=0.05, num_runs=100):
    """
    性能测试：比较CUDA和PyTorch版本的性能
    
    Args:
        xyz (torch.Tensor): 测试用的3D坐标
        camera (dict): 相机参数
        padding (float): 边界填充
        num_runs (int): 测试运行次数
    
    Returns:
        dict: 包含性能统计的字典
    """
    import time
    
    # 获取投影矩阵
    if not isinstance(camera, dict) or 'full_proj_transform' not in camera:
        raise ValueError("camera must be a dict containing 'full_proj_transform' key")
    
    proj_matrix = camera['full_proj_transform']
    
    # 预热GPU
    for _ in range(10):
        _ = visible_flag_cuda(xyz, camera, padding)
    
    torch.cuda.synchronize()
    
    # 测试CUDA版本
    start_time = time.time()
    for _ in range(num_runs):
        valid_flag, depth, p_proj = visible_flag_cuda(xyz, camera, padding)
    torch.cuda.synchronize()
    cuda_time = time.time() - start_time
    
    # 使用CUDA kernel的benchmark函数
    try:
        benchmark_results = visible_flag_module.benchmark_visible_flag(
            xyz, proj_matrix, padding, num_runs
        )
        benchmark_results = benchmark_results.cpu().numpy()
        
        cuda_kernel_time = benchmark_results[0] / 1000.0  # 转换为秒
        cuda_kernel_avg_time = benchmark_results[1] / 1000.0  # 转换为秒
        
        print(f"CUDA kernel benchmark: total={cuda_kernel_time:.4f}s, avg={cuda_kernel_avg_time:.6f}s")
        
    except Exception as e:
        print(f"CUDA kernel benchmark failed: {e}")
        cuda_kernel_time = None
        cuda_kernel_avg_time = None
    
    # 测试PyTorch版本（如果可用）
    try:
        # 这里需要原始的Python实现
        # 由于我们没有原始的_visible_flag_by_camera实现，这里只是示例
        print("Note: PyTorch benchmark requires original implementation")
        pytorch_time = None
    except:
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
    """
    性能测试：直接使用投影矩阵（更高效）
    
    Args:
        xyz (torch.Tensor): 测试用的3D坐标
        proj_matrix (torch.Tensor): 4x4投影矩阵
        padding (float): 边界填充
        num_runs (int): 测试运行次数
    
    Returns:
        dict: 包含性能统计的字典
    """
    import time
    
    # 预热GPU
    for _ in range(10):
        _ = visible_flag_cuda_direct(xyz, proj_matrix, padding)
    
    torch.cuda.synchronize()
    
    # 测试CUDA版本
    start_time = time.time()
    for _ in range(num_runs):
        valid_flag, depth, p_proj = visible_flag_cuda_direct(xyz, proj_matrix, padding)
    torch.cuda.synchronize()
    cuda_time = time.time() - start_time
    
    # 使用CUDA kernel的benchmark函数
    try:
        benchmark_results = visible_flag_module.benchmark_visible_flag(
            xyz, proj_matrix, padding, num_runs
        )
        benchmark_results = benchmark_results.cpu().numpy()
        
        cuda_kernel_time = benchmark_results[0] / 1000.0  # 转换为秒
        cuda_kernel_avg_time = benchmark_results[1] / 1000.0  # 转换为秒
        
        print(f"CUDA kernel benchmark: total={cuda_kernel_time:.4f}s, avg={cuda_kernel_avg_time:.6f}s")
        
    except Exception as e:
        print(f"CUDA kernel benchmark failed: {e}")
        cuda_kernel_time = None
        cuda_kernel_avg_time = None
    
    results = {
        'cuda_time': cuda_time,
        'cuda_avg_time': cuda_time / num_runs,
        'cuda_kernel_time': cuda_kernel_time,
        'cuda_kernel_avg_time': cuda_kernel_avg_time,
        'num_runs': num_runs,
        'num_points': xyz.size(0)
    }
    
    return results
