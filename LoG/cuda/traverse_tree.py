import torch
import time
from torch.utils.cpp_extension import load

# 加载CUDA kernel - 使用正确的相对路径
traverse_tree_module = load(
    verbose=True,
    name='traverse_tree',
    sources=['LoG/cuda/traverse_tree_kernel.cu'],
    extra_cuda_cflags=['-O2', '-arch=sm_70']
)

def traverse_tree_fast_cuda(root_indices, tree):
    """使用CUDA快速遍历树结构，返回所有分裂节点和路径信息"""
    try:
        if not root_indices.is_cuda:
            root_indices = root_indices.cuda()
        
        # 从tree对象中提取必要的属性
        node_index = tree.node_index
        tree_data = tree.tree
        index_parent = tree.index_parent
        depth_array = tree.depth
        
        # 确保所有张量都在GPU上
        if not node_index.is_cuda:
            node_index = node_index.cuda()
        if not tree_data.is_cuda:
            tree_data = tree_data.cuda()
        if not index_parent.is_cuda:
            index_parent = index_parent.cuda()
        if not depth_array.is_cuda:
            depth_array = depth_array.cuda()
        
        # 调用CUDA kernel
        result = traverse_tree_module.traverse_tree_fast(
            root_indices, node_index, tree_data, index_parent, depth_array
        )
        
        # 解析结果
        return parse_traverse_result(result)
        
    except Exception as e:
        print(f"[CUDA ERROR] traverse_tree_fast: {e}")
        print(f"[CUDA ERROR] Error type: {type(e)}")
        import traceback
        print(f"[CUDA ERROR] Full traceback:")
        traceback.print_exc()
        return traverse_tree_fallback(root_indices, tree)

def parse_traverse_result(result_tensor):
    """解析CUDA kernel返回的张量结果"""
    if result_tensor.numel() == 0:
        return {}
    
    # 结果格式：[N, 23] 其中23列包含：node_idx, depth, path_length, 20个父节点路径
    num_nodes = result_tensor.size(0)
    
    # 解析结果
    result_dict = {}
    for i in range(num_nodes):
        node_idx = result_tensor[i, 0].item()
        depth = result_tensor[i, 1].item()
        path_length = result_tensor[i, 2].item()
        
        # 提取父节点路径
        parent_path = []
        for j in range(3, 3 + path_length):
            if j < result_tensor.size(1):
                parent_idx = result_tensor[i, j].item()
                if parent_idx >= 0:  # 过滤无效索引
                    parent_path.append(parent_idx)
        
        # 构建节点信息
        result_dict[node_idx] = {
            "root": parent_path[0] if parent_path else node_idx,  # 第一个父节点是根节点
            "depth": depth,
            "parents": parent_path[1:] if len(parent_path) > 1 else [],  # 除了根节点外的父节点
            "path_length": path_length
        }
    
    return result_dict

def traverse_tree_fallback(root_indices, tree):
    """回退的CPU实现 - 基于真实的树结构"""
    result_dict = {}
    
    for root in root_indices:
        root_idx = root.item()
        # 遍历从根节点开始的所有节点
        traverse_single_root_cpu(root_idx, tree, result_dict)
    
    return result_dict

def traverse_single_root_cpu(root_idx, tree, result_dict):
    """CPU端遍历单个根节点"""
    if root_idx < 0 or root_idx >= tree.node_index.size(0):
        return
    
    # 记录根节点
    result_dict[root_idx] = {
        "root": root_idx,
        "depth": tree.depth[root_idx].item(),
        "parents": [],
        "path_length": 1
    }
    
    # 使用栈进行遍历
    stack = [(root_idx, [root_idx])]  # (node_idx, path)
    
    while stack:
        current, path = stack.pop()
        
        if tree.node_index[current] != -1:  # 不是叶子节点
            tree_idx = tree.node_index[current]
            
            # 遍历所有子节点
            for i in range(tree.tree.size(1)):
                child_idx = tree.tree[tree_idx, i].item()
                
                if child_idx >= 0 and child_idx < tree.node_index.size(0):
                    # 记录子节点信息
                    result_dict[child_idx] = {
                        "root": path[0],  # 路径中的第一个节点是根节点
                        "depth": tree.depth[child_idx].item(),
                        "parents": path[1:],  # 除了根节点外的父节点
                        "path_length": len(path) + 1
                    }
                    
                    # 将子节点加入栈中继续遍历
                    new_path = path + [child_idx]
                    stack.append((child_idx, new_path))

def count_leaves_fast_cuda(root_indices, tree):
    """使用CUDA快速统计叶子节点数量"""
    try:
        if not root_indices.is_cuda:
            root_indices = root_indices.cuda()
        
        # 从tree对象中提取必要的属性
        node_index = tree.node_index
        tree_data = tree.tree
        
        # 确保所有张量都在GPU上
        if not node_index.is_cuda:
            node_index = node_index.cuda()
        if not tree_data.is_cuda:
            tree_data = tree_data.cuda()
        
        print(f"[CUDA FAST] Processing {len(root_indices)} root nodes...")
        print(f"[CUDA FAST] Tree size: {len(node_index)} nodes")
        
        # 调用CUDA kernel
        start_time = time.time()
        total_leaves = traverse_tree_module.count_leaves_fast(
            root_indices, node_index, tree_data
        )
        cuda_time = time.time() - start_time
        
        print(f"[CUDA FAST] Performance: {cuda_time*1000:.1f}ms")
        print(f"[CUDA FAST] Found {total_leaves} leaf nodes")
        
        return total_leaves
        
    except Exception as e:
        print(f"[CUDA ERROR] count_leaves_fast: {e}")
        return count_leaves_fallback(root_indices, tree)

def count_leaves_fallback(root_indices, tree):
    """回退的CPU实现 - 基于真实的树结构（保持向后兼容）"""
    total_leaves = 0
    for root in root_indices:
        count = _count_leaves_iterative(root.item(), tree)
        total_leaves += count
    return total_leaves

def _count_leaves_iterative(node_idx, tree):
    """迭代统计单个节点的叶子节点数量 - 避免递归深度问题（保持向后兼容）"""
    if tree.node_index[node_idx] == -1:
        return 1
    
    # 使用栈进行迭代遍历
    stack = [node_idx]
    leaf_count = 0
    
    while stack:
        current = stack.pop()
        
        if tree.node_index[current] == -1:
            leaf_count += 1
        else:
            # 获取当前节点在tree中的位置
            tree_idx = tree.node_index[current]
            
            # 遍历所有子节点
            for i in range(tree.tree.size(1)):  # 使用tree.tree的列数作为max_child
                child_idx = tree.tree[tree_idx, i]
                
                # 检查子节点是否有效
                if child_idx >= 0 and child_idx < tree.node_index.size(0):
                    stack.append(child_idx)
    
    return leaf_count

def benchmark_traverse_tree(root_indices, tree, num_runs=3):
    """性能测试 - 新增的遍历功能"""
    import time
    
    print(f"[BENCHMARK] Testing traverse_tree with {len(root_indices)} root nodes")
    
    # 测试CUDA
    try:
        start_time = time.time()
        for _ in range(num_runs):
            result_cuda = traverse_tree_fast_cuda(root_indices, tree)
        cuda_time = (time.time() - start_time) / num_runs
        print(f"[BENCHMARK] CUDA traverse_tree: {cuda_time*1000:.2f}ms")
        print(f"[BENCHMARK] CUDA result: {len(result_cuda)} nodes with path info")
    except Exception as e:
        cuda_time = float('inf')
        result_cuda = None
        print(f"[BENCHMARK] CUDA traverse_tree failed: {e}")
    
    # 测试CPU
    start_time = time.time()
    for _ in range(num_runs):
        result_cpu = traverse_tree_fallback(root_indices, tree)
    cpu_time = (time.time() - start_time) / num_runs
    print(f"[BENCHMARK] CPU traverse_tree: {cpu_time*1000:.2f}ms")
    print(f"[BENCHMARK] CPU result: {len(result_cpu)} nodes with path info")
    
    # 性能对比
    if cuda_time != float('inf'):
        speedup = cpu_time / cuda_time
        print(f"[BENCHMARK] Speedup: {speedup:.2f}x")
    
    return result_cuda, result_cpu

def benchmark_leaf_counting(root_indices, tree, num_runs=5):
    """性能测试 - 原有的叶子节点计数功能（保持向后兼容）"""
    import time
    
    print(f"[BENCHMARK] Testing leaf counting with {len(root_indices)} root nodes")
    
    # 测试CUDA
    try:
        start_time = time.time()
        for _ in range(num_runs):
            result_cuda = count_leaves_fast_cuda(root_indices, tree)
        cuda_time = (time.time() - start_time) / num_runs
        print(f"[BENCHMARK] CUDA leaf counting: {cuda_time*1000:.2f}ms")
    except:
        cuda_time = float('inf')
        result_cuda = None
    
    # 测试CPU
    start_time = time.time()
    for _ in range(num_runs):
        result_cpu = count_leaves_fallback(root_indices, tree)
    cpu_time = (time.time() - start_time) / num_runs
    print(f"[BENCHMARK] CPU leaf counting: {cpu_time*1000:.2f}ms")
    
    # 性能对比
    if cuda_time != float('inf'):
        speedup = cpu_time / cuda_time
        print(f"[BENCHMARK] Speedup: {speedup:.2f}x")
    
    return result_cuda, result_cpu
