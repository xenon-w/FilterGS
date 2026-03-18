/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace CudaRasterizer
{
	struct OGFSParams
	{
		float epsilon_scale = 0.2f;
		float epsilon_max = 0.2f;
		float energy_floor = 0.05f;
		float radius_ratio_min = 0.7f;
		bool enable_stats = true;
	};
}

#ifdef FORWARD_CU_DEFINE_GLOBALS
__device__ float g_radius_scale_sum = 0.0f;
__device__ float g_radius_scale_min = 1e9f;
__device__ float g_radius_scale_max = 0.0f;
__device__ int g_radius_scale_count = 0;
#else
extern __device__ float g_radius_scale_sum;
extern __device__ float g_radius_scale_min;
extern __device__ float g_radius_scale_max;
extern __device__ int g_radius_scale_count;
#endif

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
			float* colors,
			float4* conic_opacity,
			const dim3 grid,
			uint32_t* tiles_touched,
			bool prefiltered,
			bool use_filter,
			CudaRasterizer::OGFSParams params = CudaRasterizer::OGFSParams{}
			);

	void compute_radius(
		int P,
		int H, int W,
		float* means3D,
		float* scales,
		float* rotations,
		float* viewmatrix,
		float* projmatrix,
		const float tan_fovx, float tan_fovy,
		float* radii);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		const float* depths,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	int* out_point_id,
	float* out_point_weight_pixel,
	float* out_point_weight,
	float* kv_alpha_sum,
	uint32_t* kv_alpha_count
		);
}


#endif
