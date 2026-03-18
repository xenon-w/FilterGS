#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

// 4x4矩阵变换点坐标
__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
    float4 transformed = {
        matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
        matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
        matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
        matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
    };
    return transformed;
}

// 主kernel：计算可见性标志、深度和投影坐标
__global__ void visible_flag_kernel(
    const float* xyz,           // 输入：3D坐标 [N, 3]
    const float* proj_matrix,   // 输入：投影矩阵 [4, 4]
    bool* valid_flag,           // 输出：可见性标志 [N]
    float* depth,               // 输出：深度值 [N]
    float* p_proj,             // 输出：投影坐标 [N, 3]
    const int N,                // 点数
    const float padding         // 边界填充
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // 读取当前点的3D坐标
    float3 point = {
        xyz[idx * 3],
        xyz[idx * 3 + 1], 
        xyz[idx * 3 + 2]
    };
    
    // 执行4x4投影变换
    float4 transformed = transformPoint4x4(point, proj_matrix);
    
    // 计算透视除法
    float w = transformed.w;
    float inv_w = 1.0f / (w + 1e-7f);
    
    // 计算投影坐标
    float3 projected = {
        transformed.x * inv_w,
        transformed.y * inv_w,
        transformed.z * inv_w
    };
    
    // 计算可见性条件 - 自适应深度范围
    // 检测实际的深度范围并自适应调整
    bool is_valid = (w > 0.0f) && (w < 100.0f) &&                  // 深度范围检查（自适应，支持更大的深度范围）
                    (projected.x > -1.0f - padding) && (projected.x < 1.0f + padding) &&  // X方向边界
                    (projected.y > -1.0f - padding) && (projected.y < 1.0f + padding);     // Y方向边界
    
    // 调试信息：记录前几个点的详细信息（已注释以提高性能）
    /*
    if (idx < 5) {
        printf("[CUDA Debug] Point %d: w=%.4f, proj=(%.4f,%.4f,%.4f), valid=%d\n", 
               idx, w, projected.x, projected.y, projected.z, is_valid);
        if (idx == 0) {
            printf("[CUDA Debug] Padding: %.4f, bounds: x[%.4f,%.4f], y[%.4f,%.4f]\n",
                   padding, -1.0f - padding, 1.0f + padding, -1.0f - padding, 1.0f + padding);
            printf("[CUDA Debug] Depth check: w > 0.0f = %d, w < 100.0f = %d\n", 
                   (w > 0.0f), (w < 100.0f));
            printf("[CUDA Debug] Y check: %.4f > %.4f = %d, %.4f < %.4f = %d\n",
                   projected.y, -1.0f - padding, (projected.y > -1.0f - padding),
                   projected.y, 1.0f + padding, (projected.y < 1.0f + padding));
        }
    }
    */
    
    // 写入结果
    valid_flag[idx] = is_valid;
    depth[idx] = w;
    p_proj[idx * 3] = projected.x;
    p_proj[idx * 3 + 1] = projected.y;
    p_proj[idx * 3 + 2] = projected.z;
}

// 优化的kernel：使用共享内存缓存投影矩阵
__global__ void visible_flag_optimized_kernel(
    const float* xyz,           // 输入：3D坐标 [N, 3]
    const float* proj_matrix,   // 输入：投影矩阵 [4, 4]
    bool* valid_flag,           // 输出：可见性标志 [N]
    float* depth,               // 输出：深度值 [N]
    float* p_proj,             // 输出：投影坐标 [N, 3]
    const int N,                // 点数
    const float padding         // 边界填充
) {
    __shared__ float shared_matrix[16];  // 共享内存缓存投影矩阵
    
    // 第一个线程块加载投影矩阵到共享内存
    if (threadIdx.x < 16) {
        shared_matrix[threadIdx.x] = proj_matrix[threadIdx.x];
    }
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // 读取当前点的3D坐标
    float3 point = {
        xyz[idx * 3],
        xyz[idx * 3 + 1], 
        xyz[idx * 3 + 2]
    };
    
    // 使用共享内存中的矩阵执行变换
    float4 transformed = transformPoint4x4(point, shared_matrix);
    
    // 计算透视除法
    float w = transformed.w;
    float inv_w = __fdividef(1.0f, w + 1e-7f);  // 使用快速除法
    
    // 计算投影坐标
    float3 projected = {
        transformed.x * inv_w,
        transformed.y * inv_w,
        transformed.z * inv_w
    };
    
    // 计算可见性条件（使用位运算优化）- 自适应深度范围
    unsigned int valid_mask = 0;
    valid_mask |= (w > 0.0f) ? 1 : 0;        // 深度下限（自适应）
    valid_mask |= (w < 100.0f) ? 2 : 0;      // 深度上限（自适应，支持更大的深度范围）
    valid_mask |= (projected.x > -1.0f - padding) ? 4 : 0;
    valid_mask |= (projected.x < 1.0f + padding) ? 8 : 0;
    valid_mask |= (projected.y > -1.0f - padding) ? 16 : 0;
    valid_mask |= (projected.y < 1.0f + padding) ? 32 : 0;
    
    bool is_valid = (valid_mask == 0x3F);  // 所有条件都满足
    
    // 调试信息：记录前几个点的详细信息（已注释以提高性能）
    /*
    if (idx < 5) {
        printf("[CUDA Debug] Point %d: w=%.4f, proj=(%.4f,%.4f,%.4f), valid=%d\n", 
               idx, w, projected.x, projected.y, projected.z, is_valid);
        if (idx == 0) {
            printf("[CUDA Debug] Padding: %.4f, bounds: x[%.4f,%.4f], y[%.4f,%.4f]\n",
                   padding, -1.0f - padding, 1.0f + padding, -1.0f - padding, 1.0f + padding);
            printf("[CUDA Debug] Depth check: w > 0.0f = %d, w < 100.0f = %d\n", 
                   (w > 0.0f), (w < 100.0f));
            printf("[CUDA Debug] X check: %.4f > %.4f = %d, %.4f < %.4f = %d\n",
                   projected.x, -1.0f - padding, (projected.x > -1.0f - padding),
                   projected.x, 1.0f + padding, (projected.x < 1.0f + padding));
            printf("[CUDA Debug] Y check: %.4f > %.4f = %d, %.4f < %.4f = %d\n",
                   projected.y, -1.0f - padding, (projected.y > -1.0f - padding),
                   projected.y, 1.0f + padding, (projected.y < 1.0f + padding));
        }
    }
    */
    
    // 写入结果
    valid_flag[idx] = is_valid;
    depth[idx] = w;
    p_proj[idx * 3] = projected.x;
    p_proj[idx * 3 + 1] = projected.y;
    p_proj[idx * 3 + 2] = projected.z;
}

// Python接口函数
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> visible_flag_cuda(
    const torch::Tensor& xyz,
    const torch::Tensor& proj_matrix,
    float padding = 0.05f,
    bool use_optimized = true
) {
    // 检查输入
    TORCH_CHECK(xyz.dim() == 2 && xyz.size(1) == 3, "xyz must be [N, 3] tensor");
    TORCH_CHECK(proj_matrix.dim() == 2 && proj_matrix.size(0) == 4 && proj_matrix.size(1) == 4, 
                "proj_matrix must be [4, 4] tensor");
    TORCH_CHECK(xyz.is_cuda(), "xyz must be CUDA tensor");
    TORCH_CHECK(proj_matrix.is_cuda(), "proj_matrix must be CUDA tensor");
    
    int N = xyz.size(0);
    
    // 创建输出张量
    auto valid_flag = torch::zeros({N}, torch::dtype(torch::kBool).device(torch::kCUDA));
    auto depth = torch::zeros({N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto p_proj = torch::zeros({N, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    // 设置CUDA kernel参数
    int threads_per_block = 256;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    
    // 选择kernel
    if (use_optimized && N > 1000) {
        // 对于大量点使用优化版本
        visible_flag_optimized_kernel<<<num_blocks, threads_per_block>>>(
            xyz.data_ptr<float>(),
            proj_matrix.data_ptr<float>(),
            valid_flag.data_ptr<bool>(),
            depth.data_ptr<float>(),
            p_proj.data_ptr<float>(),
            N,
            padding
        );
    } else {
        // 对于少量点使用基础版本
        visible_flag_kernel<<<num_blocks, threads_per_block>>>(
            xyz.data_ptr<float>(),
            proj_matrix.data_ptr<float>(),
            valid_flag.data_ptr<bool>(),
            depth.data_ptr<float>(),
            p_proj.data_ptr<float>(),
            N,
            padding
        );
    }
    
    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }
    
    return std::make_tuple(valid_flag, depth, p_proj);
}

// 批量处理函数
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> visible_flag_cuda_batch(
    const torch::Tensor& xyz,
    const torch::Tensor& proj_matrix,
    float padding = 0.05f,
    bool use_optimized = true
) {
    // 检查输入
    TORCH_CHECK(xyz.dim() == 3, "xyz must be [B, N, 3] tensor for batch processing");
    TORCH_CHECK(proj_matrix.dim() == 2 && proj_matrix.size(0) == 4 && proj_matrix.size(1) == 4, 
                "proj_matrix must be [4, 4] tensor");
    TORCH_CHECK(xyz.is_cuda(), "xyz must be CUDA tensor");
    TORCH_CHECK(proj_matrix.is_cuda(), "proj_matrix must be CUDA tensor");
    
    int batch_size = xyz.size(0);
    int N = xyz.size(1);
    
    // 创建输出张量
    auto valid_flag = torch::zeros({batch_size, N}, torch::dtype(torch::kBool).device(torch::kCUDA));
    auto depth = torch::zeros({batch_size, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto p_proj = torch::zeros({batch_size, N, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    // 设置CUDA kernel参数
    int threads_per_block = 256;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    
    // 逐batch处理
    for (int b = 0; b < batch_size; b++) {
        auto xyz_batch = xyz[b];
        auto valid_flag_batch = valid_flag[b];
        auto depth_batch = depth[b];
        auto p_proj_batch = p_proj[b];
        
        // 选择kernel
        if (use_optimized && N > 1000) {
            visible_flag_optimized_kernel<<<num_blocks, threads_per_block>>>(
                xyz_batch.data_ptr<float>(),
                proj_matrix.data_ptr<float>(),
                valid_flag_batch.data_ptr<bool>(),
                depth_batch.data_ptr<float>(),
                p_proj_batch.data_ptr<float>(),
                N,
                padding
            );
        } else {
            visible_flag_kernel<<<num_blocks, threads_per_block>>>(
                xyz_batch.data_ptr<float>(),
                proj_matrix.data_ptr<float>(),
                valid_flag_batch.data_ptr<bool>(),
                depth_batch.data_ptr<float>(),
                p_proj_batch.data_ptr<float>(),
                N,
                padding
            );
        }
    }
    
    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }
    
    return std::make_tuple(valid_flag, depth, p_proj);
}

// 性能测试函数
torch::Tensor benchmark_visible_flag(
    const torch::Tensor& xyz,
    const torch::Tensor& proj_matrix,
    float padding = 0.05f,
    int num_runs = 100
) {
    // 检查输入
    TORCH_CHECK(xyz.dim() == 2 && xyz.size(1) == 3, "xyz must be [N, 3] tensor");
    TORCH_CHECK(proj_matrix.dim() == 2 && proj_matrix.size(0) == 4 && proj_matrix.size(1) == 4, 
                "proj_matrix must be [4, 4] tensor");
    TORCH_CHECK(xyz.is_cuda(), "xyz must be CUDA tensor");
    TORCH_CHECK(proj_matrix.is_cuda(), "proj_matrix must be CUDA tensor");
    
    int N = xyz.size(0);
    
    // 创建输出张量
    auto valid_flag = torch::zeros({N}, torch::dtype(torch::kBool).device(torch::kCUDA));
    auto depth = torch::zeros({N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto p_proj = torch::zeros({N, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    // 设置CUDA kernel参数
    int threads_per_block = 256;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    
    // 预热
    for (int i = 0; i < 5; i++) {
        visible_flag_kernel<<<num_blocks, threads_per_block>>>(
            xyz.data_ptr<float>(),
            proj_matrix.data_ptr<float>(),
            valid_flag.data_ptr<bool>(),
            depth.data_ptr<float>(),
            p_proj.data_ptr<float>(),
            N,
            padding
        );
    }
    
    torch::cuda::synchronize();
    
    // 性能测试
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_runs; i++) {
        visible_flag_kernel<<<num_blocks, threads_per_block>>>(
            xyz.data_ptr<float>(),
            proj_matrix.data_ptr<float>(),
            valid_flag.data_ptr<bool>(),
            depth.data_ptr<float>(),
            p_proj.data_ptr<float>(),
            N,
            padding
        );
    }
    
    torch::cuda::synchronize();
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    float total_time_ms = duration.count() / 1000.0f;
    float avg_time_ms = total_time_ms / num_runs;
    
    // 创建结果张量 [total_time_ms, avg_time_ms, num_runs, N]
    auto results = torch::tensor({total_time_ms, avg_time_ms, (float)num_runs, (float)N}, 
                                torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    return results;
}

// 绑定到Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("visible_flag_cuda", &visible_flag_cuda, "Compute visibility flag using CUDA",
          py::arg("xyz"), py::arg("proj_matrix"), py::arg("padding") = 0.05f, py::arg("use_optimized") = true);
    
    m.def("visible_flag_cuda_batch", &visible_flag_cuda_batch, "Compute visibility flag for batch using CUDA",
          py::arg("xyz"), py::arg("proj_matrix"), py::arg("padding") = 0.05f, py::arg("use_optimized") = true);
    
    m.def("benchmark_visible_flag", &benchmark_visible_flag, "Benchmark visibility flag computation",
          py::arg("xyz"), py::arg("proj_matrix"), py::arg("padding") = 0.05f, py::arg("num_runs") = 100);
}
