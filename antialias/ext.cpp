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

#include <torch/extension.h>
#include "rasterize_points.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compute_radius", &compute_radius);
  m.def(
    "rasterize_gaussians",
    &RasterizeGaussiansCUDA,
    py::arg("background"),
    py::arg("means3D"),
    py::arg("colors"),
    py::arg("opacity"),
    py::arg("scales"),
    py::arg("rotations"),
    py::arg("scale_modifier"),
    py::arg("cov3D_precomp"),
    py::arg("viewmatrix"),
    py::arg("projmatrix"),
    py::arg("tan_fovx"),
    py::arg("tan_fovy"),
    py::arg("image_height"),
    py::arg("image_width"),
    py::arg("sh"),
    py::arg("degree"),
    py::arg("campos"),
    py::arg("prefiltered"),
    py::arg("use_filter"),
    py::arg("debug") = false,
    py::arg("ogfs_epsilon_scale") = 0.2f,
    py::arg("ogfs_epsilon_max") = 0.2f,
    py::arg("ogfs_energy_floor") = 0.05f,
    py::arg("ogfs_radius_ratio_min") = 0.7f,
    py::arg("ogfs_enable_stats") = true
  );
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
}
