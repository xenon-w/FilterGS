#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path

import cv2
import torch
from tqdm import tqdm
import yaml
import math
import warnings

sys.path.append(str(Path(__file__).parent.parent))

from FilterGS.filtergs import FilterGS
from LoG.utils.command import load_statedict, update_global_variable
from LoG.utils.config import Config, load_object
from LoG.utils.file import create_from_point
from LoG.utils.metric import psnr as compute_psnr, ssim as compute_ssim
from LoG.utils.trainer import prepare_batch


def _lambda_g_to_scale_max(lambda_g, block_x=16, block_y=16):
    epsilon_pix = 1.0 / 255.0
    target_tile = max(float(lambda_g), epsilon_pix)
    eps_base = max(0.0, 1.0 - epsilon_pix)
    block_pixels = float(block_x * block_y)
    lambda_G_raw = 1.0 - pow(eps_base, block_pixels)
    denom = lambda_G_raw - epsilon_pix
    if abs(denom) < 1e-6:
        epsilon_scale = 0.0
    else:
        epsilon_scale = (target_tile - epsilon_pix) / denom
    epsilon_scale = max(0.0, float(epsilon_scale))
    epsilon_max = max(target_tile, epsilon_pix)
    return epsilon_scale, float(epsilon_max)


def _apply_lambda_g_override(
    rasterizer,
    lambda_g,
    ogfs_energy_floor=None,
    ogfs_radius_ratio_min=None,
):
    if lambda_g is None and ogfs_energy_floor is None and ogfs_radius_ratio_min is None:
        return
    if not hasattr(rasterizer, "raster_settings"):
        return
    rs = rasterizer.raster_settings
    if not hasattr(rs, "_replace"):
        return
    fields = set(getattr(rs, "_fields", ()))
    if "ogfs_epsilon_scale" not in fields or "ogfs_epsilon_max" not in fields:
        return
    replace_kwargs = {}
    if lambda_g is not None:
        epsilon_scale, epsilon_max = _lambda_g_to_scale_max(lambda_g)
        replace_kwargs["ogfs_epsilon_scale"] = epsilon_scale
        replace_kwargs["ogfs_epsilon_max"] = epsilon_max
    if ogfs_energy_floor is not None and "ogfs_energy_floor" in fields:
        replace_kwargs["ogfs_energy_floor"] = float(ogfs_energy_floor)
    if ogfs_radius_ratio_min is not None and "ogfs_radius_ratio_min" in fields:
        replace_kwargs["ogfs_radius_ratio_min"] = float(ogfs_radius_ratio_min)
    if replace_kwargs:
        rasterizer.raster_settings = rs._replace(**replace_kwargs)


def load_filtergs_model(cfg, device):
    model_cfg = cfg.get('model', {})
    gaussian_cfg = model_cfg.get('args', {}).get('gaussian', {})
    init_cfg = gaussian_cfg.get('init_ply', {})

    filter_gs = FilterGS(
        sh_degree=gaussian_cfg.get('sh_degree', 1),
        xyz_scale=gaussian_cfg.get('xyz_scale', 1.0),
    )

    ply_path = init_cfg.get('filename', cfg.get('PLYNAME', None))
    if not ply_path:
        raise ValueError("Missing point cloud path: expected cfg.model.args.gaussian.init_ply.filename or cfg.PLYNAME")
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Point cloud not found: {ply_path}")

    xyz, colors, scales = create_from_point(
        filename=ply_path,
        scale3d=init_cfg.get('scale3d', cfg.get('scale3d', 1.0)),
        ret_scale=True,
    )
    filter_gs.register_by_pointcloud(
        xyz=xyz,
        colors=colors,
        scales=scales,
        init_opacity=init_cfg.get('init_opacity', 0.1),
    )
    filter_gs.to(device)
    filter_gs.eval()
    return filter_gs


def load_filtergs_from_log(cfg, device):
    """
    Build FilterGS from LoG checkpoint parameters so rendering is not done
    on randomly initialized point cloud values.
    """
    log_model = load_log_model(cfg, device)
    gaussian = log_model.gaussian
    filter_gs = FilterGS(
        sh_degree=int(gaussian.max_sh_degree),
        xyz_scale=float(getattr(gaussian, "xyz_scale", 1.0)),
    ).to(device)

    # Copy internal parameterization directly (inverse-space scaling/opacity, SH-DC colors).
    filter_gs.register_buffer("xyz", gaussian.xyz.detach().clone())
    filter_gs.register_buffer("scaling", gaussian.scaling.detach().clone())
    filter_gs.register_buffer("rotation", gaussian.rotation.detach().clone())
    filter_gs.register_buffer("opacity", gaussian.opacity.detach().clone())
    filter_gs.register_buffer("colors", gaussian.colors.detach().clone())
    if hasattr(gaussian, "shs"):
        filter_gs.register_buffer("shs", gaussian.shs.detach().clone())
    filter_gs.active_sh_degree = int(getattr(gaussian, "active_sh_degree", 0))
    filter_gs.eval()
    return filter_gs, log_model


def load_log_model(cfg, device):
    model = load_object(cfg.model.module, cfg.model.args)
    if 'ckptname' in cfg.val.keys():
        ckpt_path = cfg.val.ckptname
        print(f"[Render] Loading checkpoint from: {ckpt_path}")
        model.load_state_dict(load_statedict(ckpt_path))
    if 'model_state' in cfg.val:
        model.set_state(**cfg.val.model_state)
    model.to(device)
    model.eval()
    return model


def load_renderer(cfg, device):
    renderer = load_object(cfg.train.render.module, cfg.train.render.args)
    renderer.split = 'val'
    renderer.to(device)
    return renderer


def _build_lpips_metric(device):
    import lpips
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The parameter 'pretrained' is deprecated since 0.13",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Arguments other than a weight enum or `None` for 'weights' are deprecated",
            category=UserWarning,
        )
        try:
            metric = lpips.LPIPS(net='vgg', spatial=False, verbose=False).to(device)
        except TypeError:
            # Compatibility with older LPIPS versions without `verbose`.
            metric = lpips.LPIPS(net='vgg', spatial=False).to(device)
    metric.eval()
    return metric


def _round6(value):
    if value is None:
        return None
    return float(round(float(value), 6))


def _load_scene_avg_bar_g(scene_dir):
    scene_lambda_path = os.path.join(scene_dir, "scene_lambda.yml")
    if not os.path.exists(scene_lambda_path):
        return None
    try:
        with open(scene_lambda_path, "r") as f:
            data = yaml.safe_load(f) or {}
        avg_bar_g = data.get("avg_bar_g", None)
        if avg_bar_g is None:
            return None
        avg_bar_g = float(avg_bar_g)
        if not math.isfinite(avg_bar_g):
            return None
        return avg_bar_g
    except Exception as exc:
        print(f"[Render][Config] failed to read {scene_lambda_path}: {exc}")
        return None


def render_with_log(
    cfg,
    output_dir,
    device,
    skip_save=False,
    debug=False,
    lambda_g=None,
    ogfs_energy_floor=None,
    ogfs_radius_ratio_min=None,
):
    dataset = load_object(cfg.val.dataset.module, cfg.val.dataset.args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = load_log_model(cfg, device)
    renderer = load_renderer(cfg, device)

    os.makedirs(output_dir, exist_ok=True)
    print(f"[Render] write to {output_dir}")
    background = renderer.background
    per_frame_timing = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="render(val-flow)")):
            batch = prepare_batch(batch, device)
            camera, rasterizer, _ = renderer.prepare_camera(batch, 0, background)
            _apply_lambda_g_override(
                rasterizer,
                lambda_g=lambda_g,
                ogfs_energy_floor=ogfs_energy_floor,
                ogfs_radius_ratio_min=ogfs_radius_ratio_min,
            )
            if debug and hasattr(rasterizer, "raster_settings"):
                rs = rasterizer.raster_settings
                if hasattr(rs, "_replace"):
                    rasterizer.raster_settings = rs._replace(debug=True)

            if device.type == "cuda":
                ev_calcu_start = torch.cuda.Event(enable_timing=True)
                ev_calcu_end = torch.cuda.Event(enable_timing=True)
                ev_raster_start = torch.cuda.Event(enable_timing=True)
                ev_raster_end = torch.cuda.Event(enable_timing=True)
                ev_sync_start = torch.cuda.Event(enable_timing=True)
                ev_sync_end = torch.cuda.Event(enable_timing=True)

                ev_calcu_start.record()
                model.prepare(rasterizer, camera)
                ev_calcu_end.record()
                ev_calcu_end.synchronize()
                t_calcu_ms = float(ev_calcu_start.elapsed_time(ev_calcu_end))

                ev_sync_start.record()
                torch.cuda.synchronize()
                ev_sync_end.record()
                ev_sync_end.synchronize()
                t_synch_ms = float(ev_sync_start.elapsed_time(ev_sync_end))

                ev_raster_start.record()
                output, _ = renderer.render(camera, rasterizer, model)
                ev_raster_end.record()
                ev_raster_end.synchronize()
                t_raster_ms = float(ev_raster_start.elapsed_time(ev_raster_end))
            else:
                model.prepare(rasterizer, camera)
                output, _ = renderer.render(camera, rasterizer, model)
                t_calcu_ms, t_synch_ms, t_raster_ms = 0.0, 0.0, 0.0

            render = output['render'][0]
            per_frame_timing.append({
                "frame_idx": int(batch_idx),
                "imgname": str(batch['imgname'][0]),
                "T_calcu_ms": t_calcu_ms,
                "T_synch_ms": t_synch_ms,
                "T_raster_ms": t_raster_ms,
            })
            if not skip_save:
                vis = renderer.tensor_to_bgr(render)
                outname = os.path.join(output_dir, f"{batch_idx:06d}_{os.path.basename(batch['imgname'][0])}")
                cv2.imwrite(outname, vis)

    if per_frame_timing:
        def mean(values):
            vals = [float(v) for v in values]
            return float(sum(vals) / max(1, len(vals)))

        summary = {
            "num_frames": int(len(per_frame_timing)),
            "avg_T_calcu_ms": mean(x["T_calcu_ms"] for x in per_frame_timing),
            "avg_T_synch_ms": mean(x["T_synch_ms"] for x in per_frame_timing),
            "avg_T_raster_ms": mean(x["T_raster_ms"] for x in per_frame_timing),
        }
        denom = summary["avg_T_calcu_ms"] + summary["avg_T_synch_ms"] + summary["avg_T_raster_ms"]
        summary["fps"] = float(1000.0 / denom) if denom > 1e-8 else 0.0
        timing_path = os.path.join(os.path.dirname(output_dir), "timing_log.yml")
        with open(timing_path, "w") as f:
            yaml.safe_dump({"summary": summary, "frames": per_frame_timing}, f, sort_keys=False)
        print(f"[Timing] saved: {timing_path}")
        print(
            f"[Timing] LoG summary: frames={summary['num_frames']}, "
            f"avg_T_calcu_ms={summary['avg_T_calcu_ms']:.3f}, "
            f"avg_T_synch_ms={summary['avg_T_synch_ms']:.3f}, "
            f"avg_T_raster_ms={summary['avg_T_raster_ms']:.3f}, "
            f"fps={summary['fps']:.2f}"
        )


def render_with_filtergs(
    cfg,
    output_dir,
    device,
    skip_save=False,
    debug=False,
    lambda_g=None,
    ogfs_energy_floor=None,
    ogfs_radius_ratio_min=None,
):
    dataset = load_object(cfg.val.dataset.module, cfg.val.dataset.args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    # Use checkpoint-loaded LoG parameters for fair comparison with val mode.
    model, log_model = load_filtergs_from_log(cfg, device)
    renderer = load_renderer(cfg, device)

    os.makedirs(output_dir, exist_ok=True)
    background = renderer.background
    per_frame_timing = []
    psnr_values = []
    ssim_values = []
    lpips_values = []
    lpips_metric = None
    try:
        lpips_metric = _build_lpips_metric(device)
    except Exception as exc:
        print(f"[Metric] LPIPS unavailable: {exc}. LPIPS will be skipped.")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="render(filtergs)")):
            batch = prepare_batch(batch, device)
            camera, rasterizer, _ = renderer.prepare_camera(batch, 0, background)
            _apply_lambda_g_override(
                rasterizer,
                lambda_g=lambda_g,
                ogfs_energy_floor=ogfs_energy_floor,
                ogfs_radius_ratio_min=ogfs_radius_ratio_min,
            )
            if batch_idx == 0 and hasattr(rasterizer, "raster_settings"):
                rs0 = rasterizer.raster_settings
                print(
                    "[Render][OGFS] "
                    f"lambda_G={lambda_g}, "
                    f"epsilon_scale={float(getattr(rs0, 'ogfs_epsilon_scale', float('nan'))):.6f}, "
                    f"epsilon_max={float(getattr(rs0, 'ogfs_epsilon_max', float('nan'))):.6f}, "
                    f"energy_floor={float(getattr(rs0, 'ogfs_energy_floor', float('nan'))):.6f}, "
                    f"radius_ratio_min={float(getattr(rs0, 'ogfs_radius_ratio_min', float('nan'))):.6f}"
                )
            # Enable low-level CUDA sync checks in rasterizer when debugging,
            # so extension errors are reported at source line.
            if debug and hasattr(rasterizer, "raster_settings"):
                rs = rasterizer.raster_settings
                if hasattr(rs, "_replace"):
                    rasterizer.raster_settings = rs._replace(debug=True)
            tree_node_index_gpu = None
            ancestor_path_gpu = None
            if device.type == "cuda" and getattr(log_model, "tree", None) is not None:
                tree_node_index_gpu = log_model.tree.node_index.to(
                    device=device, dtype=torch.int32, non_blocking=True).contiguous()
                ancestor_path_gpu = log_model.tree.ancestor_path.to(
                    device=device, dtype=torch.int32, non_blocking=True).contiguous()

            ev_calcu_start = torch.cuda.Event(enable_timing=True)
            ev_calcu_end = torch.cuda.Event(enable_timing=True)
            ev_calcu_start.record()
            # LoG-pro style filtering via FilterGS (not LoG-ori prepare path).
            visible_count = model.prepare(
                rasterizer=rasterizer,
                camera=camera,
                tree=log_model.tree,
                tree_node_index_gpu=tree_node_index_gpu,
                ancestor_path_gpu=ancestor_path_gpu,
                radius_max=cfg.get('radius_max', 5.0),
                padding=cfg.get('padding', 0.1),
                radius_filter_threshold=cfg.get('radius_filter_threshold', 700.0),
            )
            ev_calcu_end.record()
            ev_calcu_end.synchronize()
            t_calcu_ms = float(ev_calcu_start.elapsed_time(ev_calcu_end))
            prepare_timing = getattr(model, "last_prepare_timing", {})
            t_tree_ms = float(prepare_timing.get("T_tree_ms", 0.0))
            t_vis_ms = float(prepare_timing.get("T_vis_ms", 0.0))
            t_pack_ms = float(prepare_timing.get("T_pack_ms", 0.0))
            t_prepare_total_ms = float(prepare_timing.get("T_prepare_total_ms", 0.0))
            vis_fallback = bool(prepare_timing.get("vis_fallback", False))

            t_sync_start = torch.cuda.Event(enable_timing=True)
            t_sync_end = torch.cuda.Event(enable_timing=True)
            t_sync_start.record()
            torch.cuda.synchronize()
            t_sync_end.record()
            t_sync_end.synchronize()
            t_synch_ms = float(t_sync_start.elapsed_time(t_sync_end))

            output = model.render(camera, rasterizer, background, enable_stats=debug)
            if debug and device.type == "cuda":
                # Force sync here so CUDA errors don't get deferred to later tensor ops.
                torch.cuda.synchronize()
            if not skip_save:
                vis = renderer.tensor_to_bgr(output['render'])
                outname = os.path.join(output_dir, f"{batch_idx:06d}_{os.path.basename(batch['imgname'][0])}")
                cv2.imwrite(outname, vis)

            pred = output['render'].detach().float().clamp(0.0, 1.0)
            gt = batch['image'][0].to(device=pred.device, dtype=pred.dtype)
            if gt.dim() == 3 and gt.shape[-1] == 3:
                gt = gt.permute(2, 0, 1)
            if gt.dim() == 3 and gt.shape[0] > 3:
                gt = gt[:3]
            if gt.shape != pred.shape:
                gt = torch.nn.functional.interpolate(
                    gt[None], size=(pred.shape[1], pred.shape[2]),
                    mode="bilinear", align_corners=False)[0]
            gt = gt.clamp(0.0, 1.0)

            corrected_pred = pred
            if getattr(log_model, 'view_correction', None) is not None and 'index' in batch:
                view_idx_tensor = batch['index']
                view_idx = None
                if torch.is_tensor(view_idx_tensor):
                    view_idx = int(view_idx_tensor.flatten()[0].detach().cpu().item())
                elif isinstance(view_idx_tensor, (list, tuple)) and len(view_idx_tensor) > 0:
                    view_idx = int(view_idx_tensor[0])
                else:
                    try:
                        view_idx = int(view_idx_tensor)
                    except Exception:
                        view_idx = None
                if view_idx is not None:
                    try:
                        vc = log_model.view_correction[view_idx].detach()
                        corrected_pred = torch.clamp(pred * vc.view(-1, 1, 1), 0.0, 1.0)
                    except Exception as corr_err:
                        print(f"[Metric] view_correction fallback: {corr_err}")

            # Keep PSNR aligned with LoG-pro validation: use corrected_pred.
            psnr_val = float(compute_psnr(corrected_pred, gt))
            ssim_val = float(compute_ssim(
                pred.permute(1, 2, 0)[None],
                gt.permute(1, 2, 0)[None],
                max_val=1.0))
            lpips_val = None
            if lpips_metric is not None:
                lpips_val = float(lpips_metric(pred[None], gt[None], normalize=True).item())
            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)
            if lpips_val is not None:
                lpips_values.append(lpips_val)

            timing = output.get("timing", {})
            t_raster_ms = float(timing.get("T_raster_ms", 0.0))
            frame_row = {
                "frame_idx": int(batch_idx),
                "imgname": str(batch['imgname'][0]),
                "T_calcu_ms": t_calcu_ms,
                "T_tree_ms": t_tree_ms,
                "T_vis_ms": t_vis_ms,
                "T_pack_ms": t_pack_ms,
                "T_prepare_total_ms": t_prepare_total_ms,
                "T_synch_ms": t_synch_ms,
                "T_raster_ms": t_raster_ms,
                "psnr": psnr_val,
                "ssim": ssim_val,
                "lpips": lpips_val,
                "num_visible": visible_count,
                "vis_fallback": vis_fallback,
            }
            if debug:
                ogfs_stats = getattr(rasterizer, "latest_stats", None) or {}
                kpc_edges = [float(x) for x in (ogfs_stats.get("kpc_edges") or [])]
                kpc_hist = [float(x) for x in (ogfs_stats.get("kpc_hist") or [])]
                num_kv_pairs = float(ogfs_stats.get("num_kv_pairs", 0.0))
                used_kv_pairs = float(ogfs_stats.get("used_kv_pairs", num_kv_pairs))
                unused_kv_pairs = float(ogfs_stats.get("unused_kv_pairs", max(0.0, num_kv_pairs - used_kv_pairs)))
                used_ratio = (used_kv_pairs / num_kv_pairs) if num_kv_pairs > 0 else 0.0
                frame_row.update({
                    "bar_g": float(ogfs_stats.get("bar_g", 0.0)),
                    "avg_kv_per_gaussian": float(ogfs_stats.get("avg_kv_per_gaussian", 0.0)),
                    "avg_kv_per_tile": float(ogfs_stats.get("avg_kv_per_tile", 0.0)),
                    "num_kv_pairs": num_kv_pairs,
                    "used_kv_pairs": used_kv_pairs,
                    "unused_kv_pairs": unused_kv_pairs,
                    "used_kv_ratio": used_ratio,
                    "kpc_edges": kpc_edges,
                    "kpc_hist": kpc_hist,
                })
            per_frame_timing.append(frame_row)

    if per_frame_timing:
        def trimmed_mean(values, trim_ratio=0.2):
            vals = sorted(float(v) for v in values)
            n = len(vals)
            if n == 0:
                return 0.0
            k = int(n * trim_ratio)
            if 2 * k >= n:
                return sum(vals) / n
            trimmed = vals[k:n - k]
            return sum(trimmed) / len(trimmed)

        count = len(per_frame_timing)
        avg_t_calcu_ms = trimmed_mean((x["T_calcu_ms"] for x in per_frame_timing), trim_ratio=0.2)
        avg_t_synch_ms = trimmed_mean((x["T_synch_ms"] for x in per_frame_timing), trim_ratio=0.2)
        avg_t_raster_ms = trimmed_mean((x["T_raster_ms"] for x in per_frame_timing), trim_ratio=0.2)
        fps_denom_ms = avg_t_calcu_ms + avg_t_synch_ms + avg_t_raster_ms
        summary = {
            "num_frames": count,
            "trim_ratio": 0.2,
            "avg_T_calcu_ms": avg_t_calcu_ms,
            "avg_T_tree_ms": trimmed_mean((x["T_tree_ms"] for x in per_frame_timing), trim_ratio=0.2),
            "avg_T_vis_ms": trimmed_mean((x["T_vis_ms"] for x in per_frame_timing), trim_ratio=0.2),
            "avg_T_pack_ms": trimmed_mean((x["T_pack_ms"] for x in per_frame_timing), trim_ratio=0.2),
            "avg_T_prepare_total_ms": trimmed_mean((x["T_prepare_total_ms"] for x in per_frame_timing), trim_ratio=0.2),
            "avg_T_synch_ms": avg_t_synch_ms,
            "avg_T_raster_ms": avg_t_raster_ms,
            "avg_fps": (1000.0 / fps_denom_ms) if fps_denom_ms > 0 else None,
            "avg_psnr": sum(psnr_values) / len(psnr_values) if psnr_values else None,
            "avg_ssim": sum(ssim_values) / len(ssim_values) if ssim_values else None,
            "avg_lpips": sum(lpips_values) / len(lpips_values) if lpips_values else None,
        }

        if debug:
            # OGFS / KPC summary
            bar_g_vals = [float(x.get("bar_g", 0.0)) for x in per_frame_timing]
            kv_total_vals = [float(x.get("num_kv_pairs", 0.0)) for x in per_frame_timing]
            kv_used_vals = [float(x.get("used_kv_pairs", 0.0)) for x in per_frame_timing]
            kv_unused_vals = [float(x.get("unused_kv_pairs", 0.0)) for x in per_frame_timing]
            kv_ratio_vals = [float(x.get("used_kv_ratio", 0.0)) for x in per_frame_timing]
            summary["avg_bar_g"] = (sum(bar_g_vals) / len(bar_g_vals)) if bar_g_vals else 0.0
            summary["avg_num_kv_pairs"] = (sum(kv_total_vals) / len(kv_total_vals)) if kv_total_vals else 0.0
            summary["avg_used_kv_pairs"] = (sum(kv_used_vals) / len(kv_used_vals)) if kv_used_vals else 0.0
            summary["avg_unused_kv_pairs"] = (sum(kv_unused_vals) / len(kv_unused_vals)) if kv_unused_vals else 0.0
            summary["avg_used_kv_ratio"] = (sum(kv_ratio_vals) / len(kv_ratio_vals)) if kv_ratio_vals else 0.0
            total_kv = sum(kv_total_vals)
            total_used = sum(kv_used_vals)
            summary["total_used_kv_ratio"] = (total_used / total_kv) if total_kv > 0 else 0.0

            # Aggregate KPC histogram by aligned edges (union over frames)
            edge_set = set()
            for frame in per_frame_timing:
                for edge in frame.get("kpc_edges", []):
                    if edge is None:
                        continue
                    edge_set.add(float(edge))
            sorted_edges = sorted(edge_set)
            if sorted_edges:
                edge_to_idx = {edge: idx for idx, edge in enumerate(sorted_edges)}
                kpc_hist_sum = [0.0] * len(sorted_edges)
                kpc_hist_count = [0] * len(sorted_edges)
                for frame in per_frame_timing:
                    edges = frame.get("kpc_edges", [])
                    hist = frame.get("kpc_hist", [])
                    for edge, val in zip(edges, hist):
                        if edge is None:
                            continue
                        idx = edge_to_idx[float(edge)]
                        fv = float(val)
                        if math.isfinite(fv):
                            kpc_hist_sum[idx] += fv
                            kpc_hist_count[idx] += 1
                summary["kpc_bin_edges"] = sorted_edges
                summary["kpc_hist_sum"] = kpc_hist_sum
                summary["kpc_hist_avg"] = [
                    (kpc_hist_sum[i] / kpc_hist_count[i]) if kpc_hist_count[i] > 0 else 0.0
                    for i in range(len(sorted_edges))
                ]

        if debug:
            timing_path = os.path.join(os.path.dirname(output_dir), "timing_filtergs.yml")
            with open(timing_path, "w") as f:
                yaml.safe_dump({"summary": summary, "frames": per_frame_timing}, f, sort_keys=False)
            print(f"[Timing] saved: {timing_path}")

            # Save scene-level redundancy indicators once.
            # This file is immutable once created; later debug runs only refresh timing_filtergs.yml.
            scene_lambda_path = os.path.join(os.path.dirname(output_dir), "scene_lambda.yml")
            if not os.path.exists(scene_lambda_path):
                scene_lambda = {
                    "avg_psnr": _round6(summary.get("avg_psnr", None)),
                    "avg_ssim": _round6(summary.get("avg_ssim", None)),
                    "avg_lpips": _round6(summary.get("avg_lpips", None)),
                    "avg_bar_g": _round6(summary.get("avg_bar_g", None)),
                    "avg_num_kv_pairs": _round6(summary.get("avg_num_kv_pairs", None)),
                    "avg_used_kv_pairs": _round6(summary.get("avg_used_kv_pairs", None)),
                    "avg_unused_kv_pairs": _round6(summary.get("avg_unused_kv_pairs", None)),
                    "avg_used_kv_ratio": _round6(summary.get("avg_used_kv_ratio", None)),
                    "total_used_kv_ratio": _round6(summary.get("total_used_kv_ratio", None)),
                }
                with open(scene_lambda_path, "w") as f:
                    yaml.safe_dump(scene_lambda, f, sort_keys=False)
                print(f"[SceneLambda] saved: {scene_lambda_path}")
            else:
                print(f"[SceneLambda] exists, skip update: {scene_lambda_path}")
        else:
            avg_psnr = summary.get("avg_psnr", None)
            avg_psnr_str = "N/A" if avg_psnr is None else f"{float(avg_psnr):.3f}"
            print(
                "[Timing][summary] "
                f"avg_T_calcu_ms={summary['avg_T_calcu_ms']:.3f}, "
                f"avg_T_synch_ms={summary['avg_T_synch_ms']:.3f}, "
                f"avg_T_raster_ms={summary['avg_T_raster_ms']:.3f}, "
                f"avg_fps={summary['avg_fps']:.3f}, "
                f"avg_psnr={avg_psnr_str}"
            )


def resolve_output_dirs(cfg_path, output_root):
    cfg_path = os.path.abspath(cfg_path)
    output_root = os.path.abspath(output_root)
    parts = cfg_path.split(os.sep)
    dataset_name = "dataset"
    scene_name = os.path.splitext(os.path.basename(cfg_path))[0]
    if "config" in parts:
        idx = parts.index("config")
        if idx + 1 < len(parts):
            dataset_name = parts[idx + 1]
        if idx + 2 < len(parts):
            scene_name = parts[idx + 2]
    scene_dir = os.path.join(output_root, dataset_name, scene_name)
    rendering_dir = os.path.join(scene_dir, "rendering")
    return scene_dir, rendering_dir


def main():
    parser = argparse.ArgumentParser(description="Render with FilterGS flow")
    parser.add_argument("--cfg", type=str, required=True, help="Config yaml")
    parser.add_argument("--output", type=str, default="output/render", help="Output directory")
    parser.add_argument("--skip-save", action="store_true", help="Skip saving rendered images")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to dump timing_filtergs.yml")
    args = parser.parse_args()

    cfg = Config.load(filename=args.cfg)
    cfg = update_global_variable(cfg, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_dir, rendering_dir = resolve_output_dirs(args.cfg, args.output)
    os.makedirs(scene_dir, exist_ok=True)
    os.makedirs(rendering_dir, exist_ok=True)
    print(f"[Render] scene output dir: {scene_dir}")

    # Read OGFS controls from config (render.yml), not CLI.
    lambda_g = cfg.get("ogfs_lambda_G", cfg.get("lambda_G", None))
    energy_floor = cfg.get("ogfs_energy_floor", None)
    radius_ratio_min = cfg.get("ogfs_radius_ratio_min", None)
    base_lambda_g = lambda_g

    # Scene-adaptive shrink boundary: tau = lambda_G * avg_bar_g
    avg_bar_g = _load_scene_avg_bar_g(scene_dir)
    if lambda_g is not None and avg_bar_g is not None:
        lambda_g = float(lambda_g) * float(avg_bar_g)
        print(
            "[Render][Config] apply scene_lambda: "
            f"base_lambda_G={base_lambda_g}, avg_bar_g={avg_bar_g}, tau={lambda_g}"
        )

    # LoG-pro-debug defaults are more sensitive to lambda_G changes.
    if lambda_g is not None and energy_floor is None:
        energy_floor = 0.01
    if lambda_g is not None and radius_ratio_min is None:
        radius_ratio_min = 0.1
    print(
        "[Render][Config] "
        f"ogfs_lambda_G_base={base_lambda_g}, "
        f"ogfs_lambda_G_effective={lambda_g}, "
        f"scene_avg_bar_g={avg_bar_g}, "
        f"ogfs_energy_floor={energy_floor}, "
        f"ogfs_radius_ratio_min={radius_ratio_min}"
    )

    render_with_filtergs(
        cfg,
        rendering_dir,
        device,
        skip_save=args.skip_save,
        debug=args.debug,
        lambda_g=lambda_g,
        ogfs_energy_floor=energy_floor,
        ogfs_radius_ratio_min=radius_ratio_min,
    )


if __name__ == "__main__":
    main()
