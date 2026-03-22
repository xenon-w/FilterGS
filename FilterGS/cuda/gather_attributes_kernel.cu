#include <torch/extension.h>
#include <vector>

namespace {

__global__ void gather_attributes_kernel(
    const int64_t* __restrict__ index,
    int64_t N,
    const float* __restrict__ xyz_in,
    const float* __restrict__ scaling_in,
    const float* __restrict__ rotation_in,
    const float* __restrict__ opacity_in,
    const float* __restrict__ colors_in,
    const float* __restrict__ shs_in,
    int sh_coeffs,
    int sh_channels,
    float* __restrict__ xyz_out,
    float* __restrict__ scaling_out,
    float* __restrict__ rotation_out,
    float* __restrict__ opacity_out,
    float* __restrict__ colors_out,
    float* __restrict__ shs_out)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }

    int64_t src = index[idx];

    // xyz
    for (int i = 0; i < 3; ++i) {
        xyz_out[idx * 3 + i] = xyz_in[src * 3 + i];
    }

    // scaling
    for (int i = 0; i < 3; ++i) {
        scaling_out[idx * 3 + i] = scaling_in[src * 3 + i];
    }

    // rotation (4)
    for (int i = 0; i < 4; ++i) {
        rotation_out[idx * 4 + i] = rotation_in[src * 4 + i];
    }

    // opacity (assume shape [N, 1])
    opacity_out[idx] = opacity_in[src];

    // colors (3)
    for (int i = 0; i < 3; ++i) {
        colors_out[idx * 3 + i] = colors_in[src * 3 + i];
    }

    if (shs_in && shs_out) {
        int plane = sh_channels;
        int coeffs = sh_coeffs;
        int stride = plane;
        for (int i = 0; i < coeffs; ++i) {
            for (int j = 0; j < plane; ++j) {
                int dst_offset = idx * coeffs * plane + i * stride + j;
                int src_offset = src * coeffs * plane + i * stride + j;
                shs_out[dst_offset] = shs_in[src_offset];
            }
        }
    }
}

} // namespace

std::vector<torch::Tensor> gather_attributes_cuda(
    torch::Tensor index,
    torch::Tensor xyz,
    torch::Tensor scaling,
    torch::Tensor rotation,
    torch::Tensor opacity,
    torch::Tensor colors,
    c10::optional<torch::Tensor> shs_opt)
{
    TORCH_CHECK(index.is_cuda(), "index must be CUDA tensor");
    TORCH_CHECK(xyz.is_cuda(), "xyz must be CUDA tensor");
    TORCH_CHECK(scaling.is_cuda(), "scaling must be CUDA tensor");
    TORCH_CHECK(rotation.is_cuda(), "rotation must be CUDA tensor");
    TORCH_CHECK(opacity.is_cuda(), "opacity must be CUDA tensor");
    TORCH_CHECK(colors.is_cuda(), "colors must be CUDA tensor");

    auto idx_contig = index.contiguous();
    auto xyz_contig = xyz.contiguous();
    auto scaling_contig = scaling.contiguous();
    auto rotation_contig = rotation.contiguous();
    auto opacity_contig = opacity.contiguous();
    auto colors_contig = colors.contiguous();

    const int64_t N = idx_contig.size(0);

    auto xyz_out = torch::empty({N, 3}, xyz.options());
    auto scaling_out = torch::empty({N, 3}, scaling.options());
    auto rotation_out = torch::empty({N, 4}, rotation.options());
    auto opacity_out = torch::empty({N, 1}, opacity.options().dtype(torch::kFloat));
    auto colors_out = torch::empty({N, 3}, colors.options());

    float* shs_out_ptr = nullptr;
    const float* shs_in_ptr = nullptr;
    int sh_coeffs = 0;
    int sh_channels = 0;
    torch::Tensor shs_out;

    if (shs_opt.has_value()) {
        auto shs = shs_opt.value();
        TORCH_CHECK(shs.is_cuda(), "shs must be CUDA tensor");
        auto shs_contig = shs.contiguous();
        shs_in_ptr = shs_contig.data_ptr<float>();
        sh_coeffs = shs_contig.size(1);
        sh_channels = shs_contig.size(2);
        shs_out = torch::empty({N, sh_coeffs, sh_channels}, shs_contig.options());
        shs_out_ptr = shs_out.data_ptr<float>();
    }

    if (N > 0) {
        const int threads = 256;
        const int blocks = (static_cast<int>(N) + threads - 1) / threads;

        gather_attributes_kernel<<<blocks, threads>>>(
            idx_contig.data_ptr<int64_t>(),
            N,
            xyz_contig.data_ptr<float>(),
            scaling_contig.data_ptr<float>(),
            rotation_contig.data_ptr<float>(),
            opacity_contig.data_ptr<float>(),
            colors_contig.data_ptr<float>(),
            shs_in_ptr,
            sh_coeffs,
            sh_channels,
            xyz_out.data_ptr<float>(),
            scaling_out.data_ptr<float>(),
            rotation_out.data_ptr<float>(),
            opacity_out.data_ptr<float>(),
            colors_out.data_ptr<float>(),
            shs_out_ptr);
    }

    if (shs_opt.has_value()) {
        return {xyz_out, scaling_out, rotation_out, opacity_out, colors_out, shs_out};
    }
    return {xyz_out, scaling_out, rotation_out, opacity_out, colors_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather_attributes_cuda", &gather_attributes_cuda, "gather attributes kernel");
}
