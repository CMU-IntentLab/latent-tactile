#!/usr/bin/env python3
"""
Check gripper action-to-state latency in HDF5 trajectories.

Loops over each trajectory and measures:
- When gripper action closes: transition from -1 to 1, then two consecutive 1s in action last dim
- When gripper action opens: transition from 1 to -1, then two consecutive -1s in action last dim
- State closing: prev state < 0.07 and curr state > 0.03 (transition low->high)
- State opening: prev state > 0.07 and curr state < 0.03 (transition high->low)
- Latency: steps from action event to state event

Usage:
  python check_gripper_latency.py --hdf5_path /path/to/consolidated.h5
  python check_gripper_latency.py --hdf5_path /path/to/dir --no_consolidated
"""

import argparse
import json
import os

import h5py
import numpy as np


# Thresholds
ACTION_CLOSE = 1.0
ACTION_OPEN = -1.0
# ACTION_TOL = 0.3  # action >= 0.7 -> close, action <= -0.7 -> open
STATE_CLOSING_THRESH = 0.03  # state > this -> closing/closed
STATE_OPENING_THRESH = 0.07  # state < this -> opening/opened


def find_action_close_indices(actions: np.ndarray, gripper_idx: int = -1) -> list[int]:
    """Indices where action transitions from -1 to 1, then two consecutive 1s. Returns index of first 1."""
    g = actions[:, gripper_idx]
    out = []
    for i in range(1, len(g) - 1):
        if (g[i - 1] <= (ACTION_OPEN )
            and g[i] >= (ACTION_CLOSE )
            and g[i + 1] >= (ACTION_CLOSE )
            # and g[i + 2] == (ACTION_CLOSE )
            ):
            out.append(i)
    return out


def find_action_open_indices(actions: np.ndarray, gripper_idx: int = -1) -> list[int]:
    """Indices where action transitions from 1 to -1, then two consecutive -1s. Returns index of first -1."""
    g = actions[:, gripper_idx]
    out = []
    for i in range(1, len(g) - 3):
        if (g[i - 1] == (ACTION_CLOSE )
            and g[i] == (ACTION_OPEN )
            and g[i + 1] == (ACTION_OPEN ) and g[i+2] == (ACTION_OPEN ) 
            # and g[i+3] == (ACTION_OPEN ))
            ):
            out.append(i)
    return out


def find_first_state_closing_after(states: np.ndarray, after_idx: int, gripper_idx: int = -1) -> int | None:
    """First index i >= after_idx where prev state < opening thresh and curr state > closing thresh (transition low->high)."""
    g = states[:, gripper_idx]
    start = max(1, after_idx)
    for i in range(start, len(g)):
        if g[i - 1] > STATE_OPENING_THRESH and g[i] < STATE_OPENING_THRESH:
            return i
    return None


def find_first_state_opening_after(states: np.ndarray, after_idx: int, gripper_idx: int = -1) -> int | None:
    """First index i >= after_idx where prev state > opening thresh and curr state < closing thresh (transition high->low)."""
    g = states[:, gripper_idx]
    start = max(1, after_idx)
    for i in range(start, len(g)):
        if g[i - 1] < STATE_CLOSING_THRESH and g[i] > STATE_CLOSING_THRESH:
            return i
    return None


def find_high_frame_diff_indices(camera_2: np.ndarray, thresh: float = 0.001) -> list[int]:
    """
    For adjacent frames (i, i+1) of camera_2, compute mean(abs(frame_i/255 - frame_{i+1}/255)).
    Return indices i where this value > thresh.
    """
    T = len(camera_2)
    if T < 2:
        return []
    out = []
    for i in range(T - 1):
        f0 = camera_2[i].astype(np.float64) / 255.0
        f1 = camera_2[i + 1].astype(np.float64) / 255.0
        diff = np.mean(np.linalg.norm(f0 - f1))
        if diff > thresh:
            # print(f'frame diff: {diff} at index {i}')
            out.append(i)
    return out


def analyze_trajectory(
    actions: np.ndarray,
    states: np.ndarray,
    traj_id: str,
    gripper_action_idx: int = -1,
    gripper_state_idx: int = -1,
) -> dict:
    """
    Analyze one trajectory. Returns dict with close/open latencies and counts.
    """
    T = min(len(actions), len(states))
    actions = np.asarray(actions[:T], dtype=np.float64)
    states = np.asarray(states[:T], dtype=np.float64)

    close_action_idxs = find_action_close_indices(actions, gripper_action_idx)
    open_action_idxs = find_action_open_indices(actions, gripper_action_idx)
    # print(f'close action idxs: {close_action_idxs}')
    # print(f'open action idxs: {open_action_idxs}')
    # print(f'states: {states[:, gripper_state_idx]}')
    # print(f'actions: {actions[:, gripper_action_idx]}')
    close_latencies = []
    for i in close_action_idxs:
        j = find_first_state_closing_after(states, i, gripper_state_idx)
        if j is not None and j-i< 30:
            print(f'close action idx: {i}, state closing idx: {j}')
            close_latencies.append(j - i)

    open_latencies = []
    for i in open_action_idxs:
        j = find_first_state_opening_after(states, i, gripper_state_idx)
        if j is not None and j-i< 30:
            print(f'open action idx: {i}, state opening idx: {j}')

            open_latencies.append(j - i)

    return {
        "traj_id": traj_id,
        "T": T,
        "n_close_actions": len(close_action_idxs),
        "n_open_actions": len(open_action_idxs),
        "n_close_latencies": len(close_latencies),
        "n_open_latencies": len(open_latencies),
        "close_latencies": close_latencies,
        "open_latencies": open_latencies,
    }


def main():
    p = argparse.ArgumentParser(description="Check gripper action-to-state latency in HDF5")
    p.add_argument("--hdf5_path", required=True, help="Consolidated HDF5 or directory with .hdf5 files")
    p.add_argument("--no_consolidated", action="store_true", help="hdf5_path is a directory of .hdf5 files")
    p.add_argument("--gripper_action_idx", type=int, default=-1, help="Action dim for gripper (default -1)")
    p.add_argument("--gripper_state_idx", type=int, default=-1, help="State dim for gripper (default -1)")
    p.add_argument("--dt", type=float, default=None, help="Seconds per step (if set, report latency in seconds)")
    p.add_argument("--verbose", action="store_true", help="Print per-trajectory details")
    p.add_argument("--frame_diff_thresh", type=float, default=11.0,
                   help="Threshold for camera_2 adjacent-frame mean abs diff (default 0.001)")
    p.add_argument("--frame_diff_json", type=str, default=None,
                   help="Output JSON file for (traj_id, high_diff_idxs) per trajectory")
    args = p.parse_args()

    if args.no_consolidated:
        files = [
            os.path.join(args.hdf5_path, f)
            for f in os.listdir(args.hdf5_path)
            if f.endswith(".hdf5")
        ]
        files.sort()
        traj_items = [(f, os.path.basename(f)) for f in files]
    else:
        with h5py.File(args.hdf5_path, "r") as hf:
            traj_ids = [k for k in hf.keys() if k.startswith("trajectory_")]
            traj_ids.sort()
            traj_items = [(args.hdf5_path, tid) for tid in traj_ids]

    all_close_latencies = []
    all_open_latencies = []
    results = []
    all_frame_diff_indices = []  # (traj_id, indices) for camera_2 adjacent-frame diff > thresh

    for filepath, traj_id in traj_items:
        with h5py.File(filepath, "r") as hf:
            if args.no_consolidated:
                traj = hf["data"] if "data" in hf else hf
            else:
                traj = hf[traj_id]
            if "actions" not in traj or "states" not in traj:
                if args.verbose:
                    print(f"  Skip {traj_id}: no actions/states")
                continue
            actions = np.array(traj["actions"])
            states = np.array(traj["states"])

            # camera_2 adjacent-frame diff: indices where mean(abs(f_i/255 - f_{i+1}/255)) > thresh
            if "camera_2" in traj:
                cam2 = np.array(traj["camera_2"])
                T_cam = min(len(cam2), len(actions), len(states))
                cam2 = cam2[:T_cam]
                high_diff_idxs = find_high_frame_diff_indices(cam2, thresh=args.frame_diff_thresh)
                ## only get the first one and the last one 
                high_diff_idxs = [high_diff_idxs[0], high_diff_idxs[-1]]
                print(f'length of trajectory: {T_cam}')
                all_frame_diff_indices.append((traj_id, high_diff_idxs))
                if args.verbose and high_diff_idxs:
                    print(f"  {traj_id}: camera_2 frame diff > {args.frame_diff_thresh} at indices: {high_diff_idxs}")

            r = analyze_trajectory(
                actions,
                states,
                traj_id,
                gripper_action_idx=args.gripper_action_idx,
                gripper_state_idx=args.gripper_state_idx,
            )
            results.append(r)
            all_close_latencies.extend(r["close_latencies"])
            all_open_latencies.extend(r["open_latencies"])
            if args.verbose and (r["close_latencies"] or r["open_latencies"]):
                unit = "s" if args.dt else "steps"
                scale = args.dt or 1.0
                print(f"  {traj_id}: T={r['T']} | close actions={r['n_close_actions']} latencies={r['n_close_latencies']} | open actions={r['n_open_actions']} latencies={r['n_open_latencies']}")
                if r["close_latencies"]:
                    print(f"    close latencies ({unit}): {[round(x * scale, 3) for x in r['close_latencies']]}")
                if r["open_latencies"]:
                    print(f"    open latencies ({unit}): {[round(x * scale, 3) for x in r['open_latencies']]}")

    # Summary
    unit = "s" if args.dt else "steps"
    scale = args.dt or 1.0

    print("\n" + "=" * 60)
    print("GRIPPER ACTION-TO-STATE LATENCY SUMMARY")
    print("=" * 60)
    print(f"Trajectories: {len(results)}")
    print(f"Action close: -1 -> 1 transition, then two consecutive 1s (action[gripper] >= {ACTION_CLOSE })")
    print(f"Action open: 1 -> -1 transition, then two consecutive -1s (action[gripper] <= {ACTION_OPEN })")
    print(f"State closing: prev < {STATE_OPENING_THRESH} and curr > {STATE_CLOSING_THRESH} (transition low->high)")
    print(f"State opening: prev > {STATE_OPENING_THRESH} and curr < {STATE_CLOSING_THRESH} (transition high->low)")
    print()

    if all_close_latencies:
        arr = np.array(all_close_latencies) * scale
        print(f"Close latency ({unit}): n={len(arr)} | mean={arr.mean():.3f} | std={arr.std():.3f} | min={arr.min():.3f} | max={arr.max():.3f}")
    else:
        print("Close latency: no events found")

    if all_open_latencies:
        arr = np.array(all_open_latencies) * scale
        print(f"Open latency ({unit}): n={len(arr)} | mean={arr.mean():.3f} | std={arr.std():.3f} | min={arr.min():.3f} | max={arr.max():.3f}")
    else:
        print("Open latency: no events found")

    # camera_2 adjacent-frame diff summary
    if all_frame_diff_indices:
        print()
        print("CAMERA_2 ADJACENT-FRAME DIFF (mean(abs(f_i/255 - f_{{i+1}}/255)) > {})".format(args.frame_diff_thresh))
        print("-" * 60)
        for traj_id, idxs in all_frame_diff_indices:
            if idxs:
                print(f"  {traj_id}: indices {idxs}")
        total_high = sum(len(idxs) for _, idxs in all_frame_diff_indices if idxs)
        trajs_with_high = sum(1 for _, idxs in all_frame_diff_indices if idxs)
        print(f"  Trajectories with camera_2: {len(all_frame_diff_indices)}")
        print(f"  Indices with diff > thresh: {total_high} (across {trajs_with_high} trajectories)")
    else:
        print()
        print("CAMERA_2: no trajectories with camera_2 data, or all adjacent-frame diffs below threshold")

    if args.frame_diff_json:
        data = [{"traj_id": tid, "high_diff_idxs": idxs} for tid, idxs in all_frame_diff_indices]
        with open(args.frame_diff_json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Frame diff data saved to {args.frame_diff_json}")

    print("=" * 60)


if __name__ == "__main__":
    main()
