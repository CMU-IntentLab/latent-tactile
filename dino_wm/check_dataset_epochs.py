#!/usr/bin/env python3
"""
Print dataset size and epoch count for train_dino_decoder_tactile.

Usage:
  python check_dataset_epochs.py --hdf5_path /path/to/consolidated.h5
  python check_dataset_epochs.py --hdf5_path /path/to/dir --no_consolidated
"""

import argparse
from tactile_dataset import TactileTrajectoryDataset, CAMERA_CONFIG


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5_path", required=True, help="Path to consolidated HDF5 or directory")
    p.add_argument("--no_consolidated", action="store_true")
    p.add_argument("--num_test", type=int, default=20)
    p.add_argument("--segment_length", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--iters", type=int, default=5000)
    p.add_argument("--cameras", type=str, default="camera_0,camera_1,camera_2")
    args = p.parse_args()

    cameras = [c.strip() for c in args.cameras.split(",")]
    for c in cameras:
        if c not in CAMERA_CONFIG:
            raise SystemExit(f"Unknown camera: {c}")

    ds = TactileTrajectoryDataset(
        args.hdf5_path,
        cameras=cameras,
        segment_length=args.segment_length,
        split="train",
        num_test=args.num_test,
        is_consolidated=not args.no_consolidated,
        seed=42,
    )

    n_train = len(ds)
    n_trajs = len(ds.trajectory_ids)
    batches_per_epoch = (n_train + args.batch_size - 1) // args.batch_size
    epochs_for_iters = args.iters / batches_per_epoch if batches_per_epoch > 0 else 0

    print(f"Dataset: {args.hdf5_path}")
    print(f"  Train samples: {n_train}")
    print(f"  Train trajectories: {n_trajs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Batches per epoch: {batches_per_epoch}")
    print()
    print(f"With --iters {args.iters}:")
    print(f"  Epochs: {epochs_for_iters:.1f}")
    print()
    print("For RAE loss (paper uses 16 epochs), consider:")
    target_epochs = 16
    iters_16_epochs = int(batches_per_epoch * target_epochs)
    print(f"  --iters {iters_16_epochs}  (~{target_epochs} epochs)")


if __name__ == "__main__":
    main()
