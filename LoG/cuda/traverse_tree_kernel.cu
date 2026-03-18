#include <torch/extension.h>
#include <cuda_runtime.h>

// 设备端栈结构 - 使用int64_t保持一致性
struct DeviceStack {
    int64_t data[32];
    int64_t top;
    
    __device__ void push(int64_t value) {
        if (top < 32) data[top++] = value;
    }
    
    __device__ int64_t pop() {
        return top > 0 ? data[--top] : -1;
    }
    
    __device__ bool empty() { return top == 0; }
    __device__ void clear() { top = 0; }
};

// 简化的叶子节点信息结构
struct LeafInfo {
    int32_t node_idx;      // 叶子节点索引
    int32_t depth;         // 节点深度
    int32_t root_idx;      // 根节点索引
};

// 单个线程处理单个根节点，基于真实的树结构
__device__ int64_t count_leaves_single_root(
    const int32_t* node_index,        // 节点索引数组 (int32)
    const int32_t* tree,              // 树结构数组 [N, max_child] (int32)
    int64_t max_child,                // 最大子节点数
    int64_t root_idx,
    int64_t max_nodes
) {
    DeviceStack stack;
    stack.clear();
    
    int64_t leaf_count = 0;
    stack.push(root_idx);
    
    while (!stack.empty()) {
        int64_t current = stack.pop();
        
        if (current < 0 || current >= max_nodes) continue;
        
        // 如果是叶子节点，计数加1
        if (node_index[current] == -1) {
            leaf_count++;
        } else {
            // 如果不是叶子节点，遍历子节点
            int64_t tree_idx = node_index[current];
            
            for (int64_t i = 0; i < max_child; i++) {
                int64_t child_idx = tree[tree_idx * max_child + i];
                
                if (child_idx >= 0 && child_idx < max_nodes) {
                    stack.push(child_idx);
                }
            }
        }
    }
    
    return leaf_count;
}

// 主kernel：并行处理多个根节点
__global__ void count_leaves_kernel(
    const int64_t* root_indices,      // 根节点索引数组 (int64)
    const int32_t* node_index,        // 节点索引数组 (int32)
    const int32_t* tree,              // 树结构数组 [N, max_child] (int32)
    int64_t max_child,                // 最大子节点数
    int64_t num_roots,                // 根节点数量
    int64_t max_nodes,                // 最大节点数
    int64_t* leaf_counts              // 输出：每个根节点的叶子节点数量
) {
    int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    
    if (idx >= num_roots) return;
    
    int64_t root_idx = root_indices[idx];
    
    // 调用单个根节点的叶子计数函数
    int64_t leaf_count = count_leaves_single_root(node_index, tree, max_child, root_idx, max_nodes);
    
    // 存储结果
    leaf_counts[idx] = leaf_count;
}

// 主函数：快速统计叶子节点数量
int64_t count_leaves_fast(
    torch::Tensor& root_indices,
    torch::Tensor& node_index,
    torch::Tensor& tree_data
) {
    const int64_t num_roots = root_indices.size(0);
    const int64_t max_nodes = node_index.size(0);
    const int64_t max_child = tree_data.size(1);  // 从tree_data的列数获取max_child
    
    // 创建输出张量 - 确保所有张量都使用int64_t类型匹配PyTorch的Long
    auto leaf_counts = torch::zeros({num_roots}, torch::dtype(torch::kInt64).device(root_indices.device()));
    
    // 计算grid和block大小
    const int threads = 256;
    const int blocks = (int)((num_roots + threads - 1) / threads);
    
    // 调用CUDA kernel
    count_leaves_kernel<<<blocks, threads>>>(
        root_indices.contiguous().data_ptr<int64_t>(),
        node_index.contiguous().data_ptr<int32_t>(),
        tree_data.contiguous().data_ptr<int32_t>(),
        max_child,
        num_roots,
        max_nodes,
        leaf_counts.contiguous().data_ptr<int64_t>()
    );
    
    // 同步GPU
    cudaDeviceSynchronize();
    
    // 计算总叶子节点数
    int64_t total_leaves = leaf_counts.sum().item<int64_t>();
    
    return total_leaves;
}

PYBIND11_MODULE(traverse_tree, m) {
    m.def("count_leaves_fast", &count_leaves_fast, "Fast count of leaf nodes from root indices");
}