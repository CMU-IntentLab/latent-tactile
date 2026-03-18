#!/usr/bin/env python3
"""
Debug script to compare tactile images extracted from video vs HDF5.
Run with: python compare_tactile_sources.py --video_path PATH --hdf5_path PATH --npz_path PATH
Saves comparison images to compare_tactile_output/
"""
import argparse
import os

import cv2
import numpy as np

# Add parent for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compute_reward_tactile import (
    load_frames_from_eval_video,
    load_tactile_frames_from_hdf5,
    extract_tactile_from_horizontal_frames,
    _normalize_tactile_for_markertracker,
    parse_trajectory_id_from_filename,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video_path", required=True, help="Eval result video")
    p.add_argument("--hdf5_path", required=True, help="Consolidated HDF5")
    p.add_argument("--npz_path", required=True, help="e.g. episode_0_traj_trajectory_0_pred_embeds.npz")
    p.add_argument("--frame_idx", type=int, default=0)
    p.add_argument("--output_dir", default="compare_tactile_output")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    traj_id = parse_trajectory_id_from_filename(args.npz_path)
    if not traj_id:
        print("Could not parse trajectory_id from npz path")
        return

    # Load from video - two extraction methods
    video_frames_h = load_frames_from_eval_video(args.video_path, extract="horizontal")
    video_frames_full = load_frames_from_eval_video(args.video_path, extract="full")
    if not video_frames_h or not video_frames_full:
        print("No video frames loaded")
        return
    n_steps = min(10, len(video_frames_h))
    frames_subset_h = [video_frames_h[min(int(i * len(video_frames_h) / max(n_steps, 1)), len(video_frames_h) - 1)] for i in range(n_steps)]

    gt_from_video_h = extract_tactile_from_horizontal_frames(frames_subset_h, panel="gt")
    gt_from_video_direct = [_normalize_tactile_for_markertracker(f) for f in load_frames_from_eval_video(args.video_path, extract="gt_tactile")[:n_steps]]
    gt_from_video = gt_from_video_h
    print(f"Video: {len(video_frames_h)} frames, horizontal frame shape: {video_frames_h[0].shape}")
    print(f"  Extracted GT tactile (from horizontal): {gt_from_video[0].shape}, dtype={gt_from_video[0].dtype}, range=[{gt_from_video[0].min()}, {gt_from_video[0].max()}]")

    # Load from HDF5
    try:
        gt_from_hdf5 = load_tactile_frames_from_hdf5(
            args.hdf5_path, traj_id, num_frames=n_steps, start_idx=0
        )
        print(f"HDF5: extracted GT tactile: {gt_from_hdf5[0].shape}, dtype={gt_from_hdf5[0].dtype}, range=[{gt_from_hdf5[0].min()}, {gt_from_hdf5[0].max()}]")
    except Exception as e:
        print(f"HDF5 load failed: {e}")
        gt_from_hdf5 = None

    # Save comparison for frame_idx
    idx = min(args.frame_idx, len(gt_from_video) - 1)
    cv2.imwrite(
        os.path.join(args.output_dir, "tactile_from_video_horizontal.png"),
        cv2.cvtColor(gt_from_video[idx], cv2.COLOR_RGB2BGR),
    )
    if gt_from_video_direct:
        cv2.imwrite(
            os.path.join(args.output_dir, "tactile_from_video_direct.png"),
            cv2.cvtColor(gt_from_video_direct[idx], cv2.COLOR_RGB2BGR),
        )
    if gt_from_hdf5:
        cv2.imwrite(
            os.path.join(args.output_dir, "tactile_from_hdf5.png"),
            cv2.cvtColor(gt_from_hdf5[idx], cv2.COLOR_RGB2BGR),
        )
        diff = np.abs(gt_from_video[idx].astype(np.float32) - gt_from_hdf5[idx].astype(np.float32))
        diff_vis = (np.clip(diff, 0, 255)).astype(np.uint8)
        cv2.imwrite(os.path.join(args.output_dir, "diff_video_vs_hdf5.png"), cv2.cvtColor(diff_vis, cv2.COLOR_RGB2BGR))

    # Save raw horizontal frame and mark extraction region
    raw_frame = frames_subset_h[idx]
    h, w = raw_frame.shape[:2]
    panel_w = w // 3
    cam_w = panel_w // 3
    # Draw rectangle around tactile region (right 1/3 of left panel)
    vis = raw_frame.copy()
    x1, y1 = 2 * cam_w, 0
    x2, y2 = panel_w, h
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(args.output_dir, "horizontal_frame_with_roi.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    print(f"Saved to {args.output_dir}/")
    print(f"Horizontal frame: {raw_frame.shape}, panel_w={panel_w}, cam_w={cam_w}")
    print(f"Tactile ROI: x=[{2*cam_w}:{panel_w}], y=[0:{h}]")


if __name__ == "__main__":
    main()
