#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

namespace {

__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f
};
__device__ const float SH_C3[] = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f
};
__device__ const float SH_C4[] = {
    2.5033429417967046f,
    -1.7701307697799304f,
    0.9461746957575601f,
    -0.6690465435572892f,
    0.10578554691520431f,
    -0.6690465435572892f,
    0.47308734787878004f,
    -1.7701307697799304f,
    0.6258357354491761f
};

__forceinline__ __device__ float3 load_sh(const float* shs_ptr, int idx, int stride)
{
    return make_float3(
        shs_ptr[idx * stride + 0],
        shs_ptr[idx * stride + 1],
        shs_ptr[idx * stride + 2]
    );
}

__forceinline__ __device__ void madd(float3& acc, float scale, const float3& value)
{
    acc.x += scale * value.x;
    acc.y += scale * value.y;
    acc.z += scale * value.z;
}

__forceinline__ __device__ float3 eval_sh_wobase(
    const float3& dir,
    const float* shs_ptr,
    int sh_coeffs,
    int sh_channels,
    int degree)
{
    if (degree <= 0 || shs_ptr == nullptr || sh_channels < 3 || sh_coeffs <= 0) {
        return make_float3(0.f, 0.f, 0.f);
    }

    const float x = dir.x;
    const float y = dir.y;
    const float z = dir.z;

    float3 result = make_float3(0.f, 0.f, 0.f);

    if (degree >= 1 && sh_coeffs >= 3) {
        float3 sh0 = load_sh(shs_ptr, 0, sh_channels);
        float3 sh1 = load_sh(shs_ptr, 1, sh_channels);
        float3 sh2 = load_sh(shs_ptr, 2, sh_channels);
        madd(result, -SH_C1 * y, sh0);
        madd(result,  SH_C1 * z, sh1);
        madd(result, -SH_C1 * x, sh2);
    }

    if (degree >= 2 && sh_coeffs >= 8) {
        const float xx = x * x;
        const float yy = y * y;
        const float zz = z * z;
        const float xy = x * y;
        const float yz = y * z;
        const float xz = x * z;

        float3 sh3 = load_sh(shs_ptr, 3, sh_channels);
        float3 sh4 = load_sh(shs_ptr, 4, sh_channels);
        float3 sh5 = load_sh(shs_ptr, 5, sh_channels);
        float3 sh6 = load_sh(shs_ptr, 6, sh_channels);
        float3 sh7 = load_sh(shs_ptr, 7, sh_channels);

        madd(result, SH_C2[0] * xy, sh3);
        madd(result, SH_C2[1] * yz, sh4);
        madd(result, SH_C2[2] * (2.0f * zz - xx - yy), sh5);
        madd(result, SH_C2[3] * xz, sh6);
        madd(result, SH_C2[4] * (xx - yy), sh7);
    }

    if (degree >= 3 && sh_coeffs >= 15) {
        const float xx = x * x;
        const float yy = y * y;
        const float zz = z * z;
        const float xy = x * y;
        const float yz = y * z;
        const float xz = x * z;

        float3 sh8  = load_sh(shs_ptr, 8,  sh_channels);
        float3 sh9  = load_sh(shs_ptr, 9,  sh_channels);
        float3 sh10 = load_sh(shs_ptr, 10, sh_channels);
        float3 sh11 = load_sh(shs_ptr, 11, sh_channels);
        float3 sh12 = load_sh(shs_ptr, 12, sh_channels);
        float3 sh13 = load_sh(shs_ptr, 13, sh_channels);
        float3 sh14 = load_sh(shs_ptr, 14, sh_channels);

        madd(result, SH_C3[0] * y * (3.0f * xx - yy), sh8);
        madd(result, SH_C3[1] * xy * z,             sh9);
        madd(result, SH_C3[2] * y * (4.0f * zz - xx - yy), sh10);
        madd(result, SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy), sh11);
        madd(result, SH_C3[4] * x * (4.0f * zz - xx - yy), sh12);
        madd(result, SH_C3[5] * z * (xx - yy), sh13);
        madd(result, SH_C3[6] * x * (xx - 3.0f * yy), sh14);
    }

    if (degree >= 4 && sh_coeffs >= 24) {
        const float xx = x * x;
        const float yy = y * y;
        const float zz = z * z;
        const float xy = x * y;
        const float yz = y * z;
        const float xz = x * z;

        float3 sh15 = load_sh(shs_ptr, 15, sh_channels);
        float3 sh16 = load_sh(shs_ptr, 16, sh_channels);
        float3 sh17 = load_sh(shs_ptr, 17, sh_channels);
        float3 sh18 = load_sh(shs_ptr, 18, sh_channels);
        float3 sh19 = load_sh(shs_ptr, 19, sh_channels);
        float3 sh20 = load_sh(shs_ptr, 20, sh_channels);
        float3 sh21 = load_sh(shs_ptr, 21, sh_channels);
        float3 sh22 = load_sh(shs_ptr, 22, sh_channels);
        float3 sh23 = load_sh(shs_ptr, 23, sh_channels);

        madd(result, SH_C4[0] * xy * (xx - yy), sh15);
        madd(result, SH_C4[1] * yz * (3.0f * xx - yy), sh16);
        madd(result, SH_C4[2] * xy * (7.0f * zz - 1.0f), sh17);
        madd(result, SH_C4[3] * yz * (7.0f * zz - 3.0f), sh18);
        madd(result, SH_C4[4] * (zz * (35.0f * zz - 30.0f) + 3.0f), sh19);
        madd(result, SH_C4[5] * xz * (7.0f * zz - 3.0f), sh20);
        madd(result, SH_C4[6] * (xx - yy) * (7.0f * zz - 1.0f), sh21);
        madd(result, SH_C4[7] * xz * (xx - 3.0f * yy), sh22);
        madd(result, SH_C4[8] * (xx * (xx - 3.0f * yy) - yy * (3.0f * xx - yy)), sh23);
    }

    return result;
}

__global__ void prepare_preprocess_inputs_kernel(
    const int64_t* __restrict__ indices,
    int64_t N,
    const float* __restrict__ xyz_in,
    const float* __restrict__ scaling_in,
    const float* __restrict__ rotation_in,
    const float* __restrict__ opacity_in,
    const float* __restrict__ colors_in,
    const float* __restrict__ shs_in,
    int sh_coeffs,
    int sh_channels,
    float c0,
    int active_sh_degree,
    const float* __restrict__ camera_center,
    float* __restrict__ position_out,
    float* __restrict__ scaling_out,
    float* __restrict__ rotation_out,
    float* __restrict__ opacity_out,
    float* __restrict__ colors_out,
    float* __restrict__ cov3d_out)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }

    const int64_t src = indices[idx];

    const float px = xyz_in[src * 3 + 0];
    const float py = xyz_in[src * 3 + 1];
    const float pz = xyz_in[src * 3 + 2];

    position_out[idx * 3 + 0] = px;
    position_out[idx * 3 + 1] = py;
    position_out[idx * 3 + 2] = pz;

    const float sx_raw = scaling_in[src * 3 + 0];
    const float sy_raw = scaling_in[src * 3 + 1];
    const float sz_raw = scaling_in[src * 3 + 2];

    const float sx = expf(sx_raw);
    const float sy = expf(sy_raw);
    const float sz = expf(sz_raw);

    scaling_out[idx * 3 + 0] = sx;
    scaling_out[idx * 3 + 1] = sy;
    scaling_out[idx * 3 + 2] = sz;

    const float op_in = opacity_in[src];
    const float op = 1.f / (1.f + expf(-op_in));
    opacity_out[idx] = op;

    const float qx = rotation_in[src * 4 + 0];
    const float qy = rotation_in[src * 4 + 1];
    const float qz = rotation_in[src * 4 + 2];
    const float qw = rotation_in[src * 4 + 3];
    const float norm = rsqrtf(qx * qx + qy * qy + qz * qz + qw * qw + 1e-8f);
    rotation_out[idx * 4 + 0] = qx * norm;
    rotation_out[idx * 4 + 1] = qy * norm;
    rotation_out[idx * 4 + 2] = qz * norm;
    rotation_out[idx * 4 + 3] = qw * norm;

    float3 color = make_float3(
        colors_in[src * 3 + 0] * c0 + 0.5f,
        colors_in[src * 3 + 1] * c0 + 0.5f,
        colors_in[src * 3 + 2] * c0 + 0.5f
    );

    if (active_sh_degree > 0 && shs_in != nullptr && sh_coeffs > 0) {
        const float cx = camera_center[0];
        const float cy = camera_center[1];
        const float cz = camera_center[2];
        float3 dir = make_float3(px - cx, py - cy, pz - cz);
        const float len_sq = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z + 1e-8f;
        const float inv_len = rsqrtf(len_sq);
        dir.x *= inv_len;
        dir.y *= inv_len;
        dir.z *= inv_len;

        const float* sh_ptr = shs_in + src * sh_coeffs * sh_channels;
        float3 sh_contrib = eval_sh_wobase(dir, sh_ptr, sh_coeffs, sh_channels, active_sh_degree);
        color.x += sh_contrib.x;
        color.y += sh_contrib.y;
        color.z += sh_contrib.z;
    }

    colors_out[idx * 3 + 0] = color.x;
    colors_out[idx * 3 + 1] = color.y;
    colors_out[idx * 3 + 2] = color.z;

    const float sx2 = sx * sx;
    const float sy2 = sy * sy;
    const float sz2 = sz * sz;

    cov3d_out[idx * 6 + 0] = sx2;
    cov3d_out[idx * 6 + 1] = sy2;
    cov3d_out[idx * 6 + 2] = sz2;
    cov3d_out[idx * 6 + 3] = 0.f;
    cov3d_out[idx * 6 + 4] = 0.f;
    cov3d_out[idx * 6 + 5] = 0.f;
}

} // anonymous namespace

std::vector<torch::Tensor> prepare_preprocess_inputs_cuda(
    torch::Tensor indices,
    torch::Tensor xyz,
    torch::Tensor scaling,
    torch::Tensor rotation,
    torch::Tensor opacity,
    torch::Tensor colors,
    c10::optional<torch::Tensor> shs_opt,
    double c0,
    int active_sh_degree,
    torch::Tensor camera_center)
{
    TORCH_CHECK(indices.is_cuda(), "indices must be CUDA tensor");
    TORCH_CHECK(xyz.is_cuda(), "xyz must be CUDA tensor");
    TORCH_CHECK(scaling.is_cuda(), "scaling must be CUDA tensor");
    TORCH_CHECK(rotation.is_cuda(), "rotation must be CUDA tensor");
    TORCH_CHECK(opacity.is_cuda(), "opacity must be CUDA tensor");
    TORCH_CHECK(colors.is_cuda(), "colors must be CUDA tensor");
    TORCH_CHECK(camera_center.is_cuda(), "camera_center must be CUDA tensor");

    auto idx_contig = indices.contiguous();
    auto xyz_contig = xyz.contiguous();
    auto scaling_contig = scaling.contiguous();
    auto rotation_contig = rotation.contiguous();
    auto opacity_contig = opacity.contiguous();
    auto colors_contig = colors.contiguous();
    auto camera_center_contig = camera_center.contiguous();

    const int64_t N = idx_contig.size(0);

    auto position_out = torch::empty({N, 3}, xyz_contig.options());
    auto scaling_out = torch::empty({N, 3}, scaling_contig.options());
    auto rotation_out = torch::empty({N, 4}, rotation_contig.options());
    auto colors_out = torch::empty({N, 3}, colors_contig.options());
    auto cov3d_out = torch::empty({N, 6}, xyz_contig.options());
    auto opacity_out = torch::empty({N}, opacity_contig.options());

    const float* shs_ptr = nullptr;
    int sh_coeffs = 0;
    int sh_channels = 0;
    if (shs_opt.has_value()) {
        auto shs = shs_opt.value();
        TORCH_CHECK(shs.is_cuda(), "shs must be CUDA tensor");
        auto shs_contig = shs.contiguous();
        shs_ptr = shs_contig.data_ptr<float>();
        sh_coeffs = shs_contig.size(1);
        sh_channels = shs_contig.size(2);
    }

    if (N > 0) {
        const int threads = 256;
        const int blocks = (static_cast<int>(N) + threads - 1) / threads;
        prepare_preprocess_inputs_kernel<<<blocks, threads>>>(
            idx_contig.data_ptr<int64_t>(),
            N,
            xyz_contig.data_ptr<float>(),
            scaling_contig.data_ptr<float>(),
            rotation_contig.data_ptr<float>(),
            opacity_contig.data_ptr<float>(),
            colors_contig.data_ptr<float>(),
            shs_ptr,
            sh_coeffs,
            sh_channels,
            static_cast<float>(c0),
            active_sh_degree,
            camera_center_contig.data_ptr<float>(),
            position_out.data_ptr<float>(),
            scaling_out.data_ptr<float>(),
            rotation_out.data_ptr<float>(),
            opacity_out.data_ptr<float>(),
            colors_out.data_ptr<float>(),
            cov3d_out.data_ptr<float>());
    }

    return {position_out, scaling_out, rotation_out, opacity_out, colors_out, cov3d_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prepare_preprocess_inputs_cuda", &prepare_preprocess_inputs_cuda, "prepare preprocess inputs");
}
