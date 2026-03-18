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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <cerrno>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

namespace CudaRasterizer
{
static RenderStats g_last_stats{};
static std::vector<double> g_last_tile_ratio{};
static std::vector<double> g_last_tile_total_kv{};
static float get_kpc_used_thresh()
{
	const char* env = std::getenv("KPC_USED_THRESH");
	if (!env || env[0] == '\0')
		return 1e-4f;
	char* end = nullptr;
	errno = 0;
	float val = std::strtof(env, &end);
	if (errno != 0 || end == env || val <= 0.0f)
		return 1e-4f;
	return val;
}
}

namespace
{
	constexpr float KPC_BIN_EDGES[] = {
		0.01f,
		0.05f,
		0.10f,
		0.15f,
		0.20f,
		0.25f,
		0.30f,
		0.35f,
		0.40f,
		0.45f,
		0.50f
	};
	constexpr int KPC_BIN_COUNT = sizeof(KPC_BIN_EDGES) / sizeof(float);
}

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	const int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	int radius = radii[idx];
	if (radius <= 0)
		return;

	uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
	uint2 rect_min, rect_max;
	getRect(points_xy[idx], radius, rect_min, rect_max, grid);

	for (int y = rect_min.y; y < rect_max.y; y++)
	{
		for (int x = rect_min.x; x < rect_max.x; x++)
		{
			uint64_t key = static_cast<uint64_t>(y * grid.x + x);
			key <<= 32;
			key |= __float_as_uint(depths[idx]);
			gaussian_keys_unsorted[off] = key;
			gaussian_values_unsorted[off] = idx;
			off++;
		}
	}
}

__global__ void accumulateOpacityHistogram(
	int P,
	const uint32_t* tiles_touched,
	const float4* conic_opacity,
	unsigned long long* bin_counts)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= P)
		return;

	uint32_t tiles = tiles_touched[idx];
	if (tiles == 0)
		return;

	float opacity = conic_opacity[idx].w;
	opacity = fmaxf(0.0f, fminf(opacity, 0.999999f));
	int bin = min(static_cast<int>(opacity * 10.0f), 9);
	atomicAdd(&bin_counts[bin], static_cast<unsigned long long>(tiles));
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	const bool use_filter,
	float* out_color,
	int* out_point_id,
		float* out_point_weight_pixel,
		float* out_point_weight,
		int* radii,
		bool debug,
		const OGFSParams* ogfs_params)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);
	
	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	OGFSParams ogfs_config{};
	if (ogfs_params != nullptr)
	{
		ogfs_config = *ogfs_params;
	}
	const bool collect_stats = ogfs_config.enable_stats;

	g_last_stats = RenderStats{};
	g_last_tile_ratio.clear();
	g_last_tile_total_kv.clear();
	g_last_stats.num_gaussians = static_cast<uint32_t>(P);

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);
	g_last_stats.num_tiles = static_cast<uint32_t>(tile_grid.x * tile_grid.y);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	if (collect_stats)
	{
		// NOTE:
		// Avoid cudaMemcpyToSymbol on g_radius_scale_* here. In this extension build
		// mode (no RDC/device-link), cross-CU device symbols may report
		// "invalid device symbol" at runtime.
		// These globals are not required for current exported OGFS stats
		// (bar_g_tile / kv usage), so skip the reset to keep stats path stable.
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
			geomState.rgb,
			geomState.conic_opacity,
			tile_grid,
			geomState.tiles_touched,
			prefiltered,
			use_filter,
			ogfs_config
		), debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);
	g_last_stats.num_kv_pairs = static_cast<uint64_t>(num_rendered);
	if (P > 0)
	{
		g_last_stats.avg_kv_per_gaussian = static_cast<double>(num_rendered) / static_cast<double>(P);
	}
	g_last_stats.num_kpc_bins = 0;
	for (int bin = 0; bin < CudaRasterizer::RenderStats::MAX_KPC_BINS; ++bin)
	{
		g_last_stats.kpc_hist[bin] = 0.0;
		g_last_stats.kpc_bin_edges[bin] = 0.0;
	}
	if (g_last_stats.num_tiles > 0)
	{
		g_last_stats.avg_kv_per_tile = static_cast<double>(num_rendered) / static_cast<double>(g_last_stats.num_tiles);
	}

	if (collect_stats && num_rendered > 0)
	{
		unsigned long long* d_opacity_bins = nullptr;
		CHECK_CUDA(cudaMalloc(&d_opacity_bins, 10 * sizeof(unsigned long long)), debug);
		CHECK_CUDA(cudaMemset(d_opacity_bins, 0, 10 * sizeof(unsigned long long)), debug);

		int threads = 256;
		int blocks = (P + threads - 1) / threads;
		if (blocks > 0)
		{
			accumulateOpacityHistogram << <blocks, threads >> > (
				P,
				geomState.tiles_touched,
				geomState.conic_opacity,
				d_opacity_bins);
			CHECK_CUDA(, debug)
		}

		unsigned long long opacity_bins_host[10];
		CHECK_CUDA(cudaMemcpy(opacity_bins_host, d_opacity_bins, 10 * sizeof(unsigned long long), cudaMemcpyDeviceToHost), debug);
		CHECK_CUDA(cudaFree(d_opacity_bins), debug);

		(void)opacity_bins_host;
	}


	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
		const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
		const float* depth_ptr = geomState.depths;
	float* kv_alpha_sum = nullptr;
	uint32_t* kv_alpha_count = nullptr;
	if (collect_stats && num_rendered > 0)
	{
		CHECK_CUDA(cudaMalloc(&kv_alpha_sum, num_rendered * sizeof(float)), debug);
		CHECK_CUDA(cudaMalloc(&kv_alpha_count, num_rendered * sizeof(uint32_t)), debug);
			CHECK_CUDA(cudaMemset(kv_alpha_sum, 0, num_rendered * sizeof(float)), debug);
			CHECK_CUDA(cudaMemset(kv_alpha_count, 0, num_rendered * sizeof(uint32_t)), debug);
		}
		CHECK_CUDA(FORWARD::render(
			tile_grid, block,
			imgState.ranges,
			binningState.point_list,
			width, height,
			geomState.means2D,
			feature_ptr,
			depth_ptr,
			geomState.conic_opacity,
			imgState.accum_alpha,
			imgState.n_contrib,
			background,
			out_color,
			out_point_id,
			out_point_weight_pixel,
			out_point_weight,
			kv_alpha_sum,
			kv_alpha_count
		), debug)

	if (collect_stats && num_rendered > 0)
	{
		std::vector<float> kv_alpha_sum_host(num_rendered);
		std::vector<uint32_t> kv_alpha_count_host(num_rendered);
		CHECK_CUDA(cudaMemcpy(kv_alpha_sum_host.data(), kv_alpha_sum, num_rendered * sizeof(float), cudaMemcpyDeviceToHost), debug);
		CHECK_CUDA(cudaMemcpy(kv_alpha_count_host.data(), kv_alpha_count, num_rendered * sizeof(uint32_t), cudaMemcpyDeviceToHost), debug);
		std::vector<uint2> tile_ranges_host;
		if (g_last_stats.num_tiles > 0)
		{
			tile_ranges_host.resize(g_last_stats.num_tiles);
			CHECK_CUDA(cudaMemcpy(tile_ranges_host.data(), imgState.ranges, g_last_stats.num_tiles * sizeof(uint2), cudaMemcpyDeviceToHost), debug);
		}
			unsigned long long effective_bins[KPC_BIN_COUNT] = { 0 };
			unsigned long long used_kv = 0;
			unsigned long long unused_kv = 0;
				for (int idx = 0; idx < num_rendered; ++idx)
				{
					if (kv_alpha_count_host[idx] == 0)
					{
						unused_kv++;
						continue;
					}
					float kpc_val = kv_alpha_sum_host[idx];
					int bin = 0;
					while (bin < KPC_BIN_COUNT && kpc_val > KPC_BIN_EDGES[bin])
					{
						bin++;
					}
					if (bin >= KPC_BIN_COUNT)
					{
						if (kpc_val > KPC_BIN_EDGES[KPC_BIN_COUNT - 1])
							continue;
						bin = KPC_BIN_COUNT - 1;
					}
					effective_bins[bin] += 1;
					used_kv++;
				}
				if (!tile_ranges_host.empty())
				{
					double tile_gtc_sum = 0.0;
					g_last_tile_ratio.assign(g_last_stats.num_tiles, 0.0);
					g_last_tile_total_kv.assign(g_last_stats.num_tiles, 0.0);
					const float kpc_used_thresh = get_kpc_used_thresh();
					for (uint32_t tile_idx = 0; tile_idx < g_last_stats.num_tiles; ++tile_idx)
					{
						uint2 range = tile_ranges_host[tile_idx];
						uint32_t start = range.x;
						uint32_t end = range.y;
						if (end <= start)
							continue;
						double tile_sum = 0.0;
						uint32_t used_in_tile = 0;
						g_last_tile_total_kv[tile_idx] = static_cast<double>(end - start);
						for (uint32_t kv_idx = start; kv_idx < end; ++kv_idx)
						{
							tile_sum += static_cast<double>(kv_alpha_sum_host[kv_idx]);
							if (kv_alpha_count_host[kv_idx] > 0 && kv_alpha_sum_host[kv_idx] >= kpc_used_thresh)
								used_in_tile++;
						}
						double tile_avg = tile_sum / static_cast<double>(end - start);
						tile_gtc_sum += tile_avg;
						g_last_tile_ratio[tile_idx] = static_cast<double>(used_in_tile) / static_cast<double>(end - start);
					}
					if (g_last_stats.num_tiles > 0)
					{
						g_last_stats.bar_g_tile = tile_gtc_sum / static_cast<double>(g_last_stats.num_tiles);
					}
				}
				g_last_stats.num_kpc_bins = std::min(KPC_BIN_COUNT, CudaRasterizer::RenderStats::MAX_KPC_BINS);
				for (int bin = 0; bin < g_last_stats.num_kpc_bins; ++bin)
				{
					g_last_stats.kpc_hist[bin] = static_cast<double>(effective_bins[bin]);
					g_last_stats.kpc_bin_edges[bin] = static_cast<double>(KPC_BIN_EDGES[bin]);
				}

				g_last_stats.used_kv_pairs = used_kv;
				g_last_stats.unused_kv_pairs = unused_kv;
	}
	if (kv_alpha_sum != nullptr)
		CHECK_CUDA(cudaFree(kv_alpha_sum), debug);
	if (kv_alpha_count != nullptr)
		CHECK_CUDA(cudaFree(kv_alpha_count), debug);

		return num_rendered;
	}

CudaRasterizer::RenderStats CudaRasterizer::Rasterizer::getLastRenderStats()
{
	return g_last_stats;
}

const std::vector<double>& CudaRasterizer::Rasterizer::getLastTileRatio()
{
	return g_last_tile_ratio;
}

const std::vector<double>& CudaRasterizer::Rasterizer::getLastTileTotalKv()
{
	return g_last_tile_total_kv;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot), debug)
}
