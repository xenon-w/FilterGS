#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <tuple>

namespace {

__global__ void mark_masks_kernel(
    const int64_t* __restrict__ visible_indices,
    const float* __restrict__ radius2d,
    int64_t num_visible,
    float radius_max,
    const int32_t* __restrict__ tree_node_index,
    bool* __restrict__ filtered_mask,
    uint8_t* __restrict__ internal_small_flags,
    uint8_t* __restrict__ leaf_flags,
    int32_t* __restrict__ leaf_total_counter,
    int32_t* __restrict__ internal_kept_counter)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_visible) {
        return;
    }

    int64_t point_idx = visible_indices[idx];
    int32_t tree_idx = tree_node_index[point_idx];
    bool is_leaf = (tree_idx == -1);
    bool small_enough = radius2d[idx] <= radius_max;
    bool pass_filter = (is_leaf || small_enough);

    filtered_mask[idx] = pass_filter;
    leaf_flags[idx] = static_cast<uint8_t>(is_leaf);

    if (!is_leaf && small_enough) {
        internal_small_flags[point_idx] = 1;
    }

    if (is_leaf) {
        atomicAdd(leaf_total_counter, 1);
    } else if (pass_filter) {
        atomicAdd(internal_kept_counter, 1);
    }
}

__global__ void ancestor_filter_kernel(
    const int64_t* __restrict__ visible_indices,
    const int32_t* __restrict__ ancestor_path,
    int max_level,
    const uint8_t* __restrict__ internal_small_flags,
    const uint8_t* __restrict__ leaf_flags,
    bool* __restrict__ filtered_mask,
    int64_t num_visible,
    int32_t* __restrict__ ancestor_filtered_counter,
    int32_t* __restrict__ leaf_kept_counter)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_visible) {
        return;
    }

    if (!leaf_flags[idx]) {
        return;
    }

    if (!filtered_mask[idx]) {
        return;
    }

    int64_t point_idx = visible_indices[idx];
    const int32_t* ancestor_row = ancestor_path + point_idx * static_cast<int64_t>(max_level);

    bool drop_leaf = false;
    for (int level = 0; level < max_level; ++level) {
        int32_t ancestor = ancestor_row[level];
        if (ancestor < 0) {
            break;
        }
        if (internal_small_flags[ancestor]) {
            drop_leaf = true;
            break;
        }
    }

    if (drop_leaf) {
        filtered_mask[idx] = false;
        atomicAdd(ancestor_filtered_counter, 1);
    } else {
        atomicAdd(leaf_kept_counter, 1);
    }
}

} // anonymous namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> filter_visible_cuda(
    const torch::Tensor& visible_indices,
    const torch::Tensor& radius2d,
    double radius_max,
    const torch::Tensor& tree_node_index,
    const torch::Tensor& ancestor_path)
{
    TORCH_CHECK(visible_indices.is_cuda(), "visible_indices must be a CUDA tensor");
    TORCH_CHECK(radius2d.is_cuda(), "radius2d must be a CUDA tensor");
    TORCH_CHECK(tree_node_index.is_cuda(), "tree_node_index must be a CUDA tensor");
    TORCH_CHECK(ancestor_path.is_cuda(), "ancestor_path must be a CUDA tensor");

    TORCH_CHECK(visible_indices.dim() == 1, "visible_indices must be 1D");
    TORCH_CHECK(radius2d.dim() == 1, "radius2d must be 1D");
    TORCH_CHECK(visible_indices.size(0) == radius2d.size(0), "visible_indices and radius2d must have same length");
    TORCH_CHECK(tree_node_index.dim() == 1, "tree_node_index must be 1D");
    TORCH_CHECK(ancestor_path.dim() == 2, "ancestor_path must be 2D");
    TORCH_CHECK(tree_node_index.size(0) == ancestor_path.size(0), "tree_node_index and ancestor_path must align on dim 0");

    auto num_visible = visible_indices.size(0);
    auto num_points = tree_node_index.size(0);
    auto max_level = ancestor_path.size(1);

    auto options_bool = torch::TensorOptions().dtype(torch::kBool).device(visible_indices.device());
    auto options_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(visible_indices.device());
    auto options_i32 = torch::TensorOptions().dtype(torch::kInt32).device(visible_indices.device());

    auto filtered_mask = torch::empty({num_visible}, options_bool);
    auto internal_small_flags = torch::zeros({num_points}, options_u8);
    auto leaf_flags = torch::zeros({num_visible}, options_u8);
    auto leaf_total_counter = torch::zeros({1}, options_i32);
    auto internal_kept_counter = torch::zeros({1}, options_i32);
    auto ancestor_filtered_counter = torch::zeros({1}, options_i32);
    auto leaf_kept_counter = torch::zeros({1}, options_i32);

    const int threads = 256;
    const int blocks = (static_cast<int>(num_visible) + threads - 1) / threads;
    mark_masks_kernel<<<blocks, threads>>>(
        visible_indices.data_ptr<int64_t>(),
        radius2d.data_ptr<float>(),
        num_visible,
        static_cast<float>(radius_max),
        tree_node_index.data_ptr<int32_t>(),
        filtered_mask.data_ptr<bool>(),
        internal_small_flags.data_ptr<uint8_t>(),
        leaf_flags.data_ptr<uint8_t>(),
        leaf_total_counter.data_ptr<int32_t>(),
        internal_kept_counter.data_ptr<int32_t>());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in mark_masks_kernel: ") + cudaGetErrorString(err));
    }

    if (num_visible > 0) {
        const int leaf_blocks = (static_cast<int>(num_visible) + threads - 1) / threads;
        ancestor_filter_kernel<<<leaf_blocks, threads>>>(
            visible_indices.data_ptr<int64_t>(),
            ancestor_path.data_ptr<int32_t>(),
            static_cast<int>(max_level),
            internal_small_flags.data_ptr<uint8_t>(),
            leaf_flags.data_ptr<uint8_t>(),
            filtered_mask.data_ptr<bool>(),
            num_visible,
            ancestor_filtered_counter.data_ptr<int32_t>(),
            leaf_kept_counter.data_ptr<int32_t>());

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA error in ancestor_filter_kernel: ") + cudaGetErrorString(err));
        }
    }

    return std::make_tuple(
        filtered_mask,
        ancestor_filtered_counter,
        leaf_total_counter,
        leaf_kept_counter,
        internal_kept_counter);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("filter_visible_cuda", &filter_visible_cuda, "Filter visible nodes kernel");
}
