#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>


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


__global__ void visible_flag_kernel(
    const float* xyz,
    const float* proj_matrix,
    bool* valid_flag,
    float* depth,
    float* p_proj,
    const int N,
    const float padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    

    float3 point = {
        xyz[idx * 3],
        xyz[idx * 3 + 1], 
        xyz[idx * 3 + 2]
    };
    

    float4 transformed = transformPoint4x4(point, proj_matrix);
    

    float w = transformed.w;
    float inv_w = 1.0f / (w + 1e-7f);
    

    float3 projected = {
        transformed.x * inv_w,
        transformed.y * inv_w,
        transformed.z * inv_w
    };
    


    bool is_valid = (w > 0.0f) && (w < 100.0f) &&
                    (projected.x > -1.0f - padding) && (projected.x < 1.0f + padding) &&
                    (projected.y > -1.0f - padding) && (projected.y < 1.0f + padding);
    

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
    

    valid_flag[idx] = is_valid;
    depth[idx] = w;
    p_proj[idx * 3] = projected.x;
    p_proj[idx * 3 + 1] = projected.y;
    p_proj[idx * 3 + 2] = projected.z;
}


__global__ void visible_flag_optimized_kernel(
    const float* xyz,
    const float* proj_matrix,
    bool* valid_flag,
    float* depth,
    float* p_proj,
    const int N,
    const float padding
) {
    __shared__ float shared_matrix[16];
    

    if (threadIdx.x < 16) {
        shared_matrix[threadIdx.x] = proj_matrix[threadIdx.x];
    }
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    

    float3 point = {
        xyz[idx * 3],
        xyz[idx * 3 + 1], 
        xyz[idx * 3 + 2]
    };
    

    float4 transformed = transformPoint4x4(point, shared_matrix);
    

    float w = transformed.w;
    float inv_w = __fdividef(1.0f, w + 1e-7f);
    

    float3 projected = {
        transformed.x * inv_w,
        transformed.y * inv_w,
        transformed.z * inv_w
    };
    

    unsigned int valid_mask = 0;
    valid_mask |= (w > 0.0f) ? 1 : 0;
    valid_mask |= (w < 100.0f) ? 2 : 0;
    valid_mask |= (projected.x > -1.0f - padding) ? 4 : 0;
    valid_mask |= (projected.x < 1.0f + padding) ? 8 : 0;
    valid_mask |= (projected.y > -1.0f - padding) ? 16 : 0;
    valid_mask |= (projected.y < 1.0f + padding) ? 32 : 0;
    
    bool is_valid = (valid_mask == 0x3F);
    

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
    

    valid_flag[idx] = is_valid;
    depth[idx] = w;
    p_proj[idx * 3] = projected.x;
    p_proj[idx * 3 + 1] = projected.y;
    p_proj[idx * 3 + 2] = projected.z;
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> visible_flag_cuda(
    const torch::Tensor& xyz,
    const torch::Tensor& proj_matrix,
    float padding = 0.05f,
    bool use_optimized = true
) {

    TORCH_CHECK(xyz.dim() == 2 && xyz.size(1) == 3, "xyz must be [N, 3] tensor");
    TORCH_CHECK(proj_matrix.dim() == 2 && proj_matrix.size(0) == 4 && proj_matrix.size(1) == 4, 
                "proj_matrix must be [4, 4] tensor");
    TORCH_CHECK(xyz.is_cuda(), "xyz must be CUDA tensor");
    TORCH_CHECK(proj_matrix.is_cuda(), "proj_matrix must be CUDA tensor");
    
    int N = xyz.size(0);
    

    auto valid_flag = torch::zeros({N}, torch::dtype(torch::kBool).device(torch::kCUDA));
    auto depth = torch::zeros({N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto p_proj = torch::zeros({N, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    

    int threads_per_block = 256;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    

    if (use_optimized && N > 1000) {

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
    

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }
    
    return std::make_tuple(valid_flag, depth, p_proj);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> visible_flag_cuda_batch(
    const torch::Tensor& xyz,
    const torch::Tensor& proj_matrix,
    float padding = 0.05f,
    bool use_optimized = true
) {

    TORCH_CHECK(xyz.dim() == 3, "xyz must be [B, N, 3] tensor for batch processing");
    TORCH_CHECK(proj_matrix.dim() == 2 && proj_matrix.size(0) == 4 && proj_matrix.size(1) == 4, 
                "proj_matrix must be [4, 4] tensor");
    TORCH_CHECK(xyz.is_cuda(), "xyz must be CUDA tensor");
    TORCH_CHECK(proj_matrix.is_cuda(), "proj_matrix must be CUDA tensor");
    
    int batch_size = xyz.size(0);
    int N = xyz.size(1);
    

    auto valid_flag = torch::zeros({batch_size, N}, torch::dtype(torch::kBool).device(torch::kCUDA));
    auto depth = torch::zeros({batch_size, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto p_proj = torch::zeros({batch_size, N, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    

    int threads_per_block = 256;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    

    for (int b = 0; b < batch_size; b++) {
        auto xyz_batch = xyz[b];
        auto valid_flag_batch = valid_flag[b];
        auto depth_batch = depth[b];
        auto p_proj_batch = p_proj[b];
        

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
    

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));
    }
    
    return std::make_tuple(valid_flag, depth, p_proj);
}


torch::Tensor benchmark_visible_flag(
    const torch::Tensor& xyz,
    const torch::Tensor& proj_matrix,
    float padding = 0.05f,
    int num_runs = 100
) {

    TORCH_CHECK(xyz.dim() == 2 && xyz.size(1) == 3, "xyz must be [N, 3] tensor");
    TORCH_CHECK(proj_matrix.dim() == 2 && proj_matrix.size(0) == 4 && proj_matrix.size(1) == 4, 
                "proj_matrix must be [4, 4] tensor");
    TORCH_CHECK(xyz.is_cuda(), "xyz must be CUDA tensor");
    TORCH_CHECK(proj_matrix.is_cuda(), "proj_matrix must be CUDA tensor");
    
    int N = xyz.size(0);
    

    auto valid_flag = torch::zeros({N}, torch::dtype(torch::kBool).device(torch::kCUDA));
    auto depth = torch::zeros({N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto p_proj = torch::zeros({N, 3}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    

    int threads_per_block = 256;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    

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
    

    auto results = torch::tensor({total_time_ms, avg_time_ms, (float)num_runs, (float)N}, 
                                torch::dtype(torch::kFloat32).device(torch::kCUDA));
    
    return results;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("visible_flag_cuda", &visible_flag_cuda, "Compute visibility flag using CUDA",
          py::arg("xyz"), py::arg("proj_matrix"), py::arg("padding") = 0.05f, py::arg("use_optimized") = true);
    
    m.def("visible_flag_cuda_batch", &visible_flag_cuda_batch, "Compute visibility flag for batch using CUDA",
          py::arg("xyz"), py::arg("proj_matrix"), py::arg("padding") = 0.05f, py::arg("use_optimized") = true);
    
    m.def("benchmark_visible_flag", &benchmark_visible_flag, "Benchmark visibility flag computation",
          py::arg("xyz"), py::arg("proj_matrix"), py::arg("padding") = 0.05f, py::arg("num_runs") = 100);
}
