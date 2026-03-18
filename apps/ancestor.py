#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Rebuild TensorTree ancestor_path directly on a LoG checkpoint."""

import argparse
import os
import torch


def _load_state_dict(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        wrapper = ckpt
    else:
        state_dict = ckpt
        wrapper = None
    return wrapper, state_dict


def rebuild_ancestor_path(state_dict: dict):
    required_keys = [
        "tree.index_parent",
        "tree.depth",
    ]
    for key in required_keys:
        if key not in state_dict:
            raise KeyError(f"Missing '{key}' in checkpoint state_dict")

    index_parent = state_dict["tree.index_parent"].to(torch.long)
    depth = state_dict["tree.depth"].to(torch.long)

    ancestor_key = None
    for candidate in ("tree.ancestor_path", "tree.ancestor"):
        if candidate in state_dict:
            ancestor_key = candidate
            break

    if ancestor_key is not None:
        ancestor_old = state_dict[ancestor_key]
        max_level = ancestor_old.shape[1]
    else:
        max_level = int(depth.max().item()) + 1 if depth.numel() > 0 else 1
        ancestor_old = torch.empty(index_parent.numel(), max_level, dtype=torch.int32)

    num_points = depth.numel()
    max_level = ancestor_old.shape[1] if ancestor_old.ndim == 2 else max_level

    ancestor_new = torch.full_like(ancestor_old, -1)

    if num_points == 0 or max_level == 0:
        state_dict["tree.ancestor_path"] = ancestor_new
        if "tree.ancestor" in state_dict:
            state_dict["tree.ancestor"] = ancestor_new
        return ancestor_new, 0

    max_depth = int(depth.max().item())

    for level in range(1, max_depth + 1):
        mask = depth == level
        if not mask.any():
            continue

        nodes = mask.nonzero(as_tuple=False).squeeze(1)
        parents = index_parent[nodes]

        valid = parents >= 0
        if valid.any():
            parent_idx = parents[valid]
            ancestor_new[nodes[valid]] = ancestor_new[parent_idx]
            ancestor_new[nodes[valid], level - 1] = parent_idx.to(ancestor_new.dtype)

        orphan_mask = ~valid
        if orphan_mask.any():
            orphan_nodes = nodes[orphan_mask]
            ancestor_new[orphan_nodes] = -1

    state_dict["tree.ancestor_path"] = ancestor_new
    if ancestor_key == "tree.ancestor" or "tree.ancestor" in state_dict:
        state_dict["tree.ancestor"] = ancestor_new
    return ancestor_new, max_depth


def parse_args():
    parser = argparse.ArgumentParser(description="Rebuild ancestor_path buffer in LoG checkpoints")
    parser.add_argument("--ckpt", required=True, help="Input checkpoint path")
    parser.add_argument("--out", default=None, help="Output checkpoint path")
    return parser.parse_args()


def main():
    args = parse_args()

    wrapper, state_dict = _load_state_dict(args.ckpt)

    ancestor_new, max_depth = rebuild_ancestor_path(state_dict)
    print(f"[Info] Rebuilt ancestor_path: shape={tuple(ancestor_new.shape)}, max_depth={max_depth}")

    if wrapper is not None:
        wrapper["state_dict"] = state_dict
        payload = wrapper
    else:
        payload = state_dict

    if args.out is None:
        base, ext = os.path.splitext(args.ckpt)
        output_path = base + "_ancestor" + ext
    else:
        output_path = args.out

    torch.save(payload, output_path)
    print(f"[Info] Saved to {output_path}")


if __name__ == "__main__":
    main()
