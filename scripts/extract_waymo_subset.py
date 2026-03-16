"""
Extract Waymo subset tar shards into the directory layout expected by ImpromptuVLADataset:
  dataset_dir / subset / split / {id}.mp4, {id}.npy

Trajectories are parsed from .q7_answer.txt (same format as the tokenizer) and saved as .npy.
Video files are extracted as .mp4. Run this after download_waymo_subset.py and before
fitting the tokenizer / training.
"""

import os
import re
import tarfile
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

# Subset and splits that ImpromptuVLADataset uses by default
SUBSET = "Unconventional Dynamic Obstacles"
TRAIN_PREFIX = "waymo_train"
VAL_PREFIX = "waymo_val"


def parse_q7_answer(content: str):
    """Parse [x, y] trajectory from .q7_answer.txt text. Returns (N, 2) numpy array or None."""
    matches = re.findall(r"\[\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\]", content)
    if not matches:
        return None
    return np.array([[float(m[0]), float(m[1])] for m in matches], dtype=np.float32)


def get_split(tar_basename: str) -> str:
    """Return 'train' or 'val' based on tar filename."""
    if tar_basename.startswith(VAL_PREFIX):
        return "val"
    return "train"


def extract_tar(tar_path: str, out_root: str, subset: str) -> int:
    """
    Extract one tar into out_root/subset/train/ or .../val/.
    Groups members by sample id (path stem). For each sample that has .q7_answer.txt,
    writes .npy and extracts video as .mp4. Returns number of samples written.
    """
    out_root = Path(out_root)
    tar_path = Path(tar_path)
    tar_basename = tar_path.stem  # e.g. waymo_train_shard_0000
    split = get_split(tar_basename)
    out_dir = out_root / subset / split
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group members by sample id: (dir part, stem) -> list of (member, ext)
    groups = defaultdict(list)
    video_exts = (".mp4", ".webm", ".avi", ".mkv")

    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = member.name
            stem = Path(name).stem
            ext = Path(name).suffix.lower()
            # Use full path inside tar as group key so different dirs don't collide
            key = (os.path.dirname(name), stem)
            groups[key].append((member, ext))

    count = 0
    with tarfile.open(tar_path, "r") as tar:
        for (_d, stem), files in groups.items():
            q7_member = None
            vid_member = None
            for member, ext in files:
                if member.name.endswith(".q7_answer.txt"):
                    q7_member = member
                elif ext in video_exts:
                    vid_member = (member, ext)

            if not q7_member:
                continue

            # Parse trajectory
            f = tar.extractfile(q7_member)
            content = f.read().decode("utf-8") if f else ""
            traj = parse_q7_answer(content)
            if traj is None or len(traj) == 0:
                continue

            # Unique id: tar stem + sample stem to avoid collisions across shards
            sample_id = f"{tar_basename}_{stem}"
            npy_path = out_dir / f"{sample_id}.npy"
            mp4_path = out_dir / f"{sample_id}.mp4"

            # Save actions (horizon, 2) for tokenizer compatibility
            np.save(npy_path, traj)

            # Extract video if present
            if vid_member is not None:
                mem, _ = vid_member
                tar.extract(mem, path=out_dir)
                extracted = out_dir / mem.name
                if extracted != mp4_path and extracted.exists():
                    extracted.rename(mp4_path)
                # If extract put file in a subdir (e.g. waymo/000001.mp4), clean up
                if extracted.parent != out_dir and extracted.parent.exists():
                    try:
                        extracted.parent.rmdir()
                    except OSError:
                        pass
            else:
                # No video in this sample; skip so dataset doesn't load empty
                npy_path.unlink(missing_ok=True)
                continue

            count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Extract Waymo subset tars to mp4+npy for training.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/waymo_subset",
        help="Directory containing waymo/*.tar shards (output of download_waymo_subset.py)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=SUBSET,
        help=f"Subset name under dataset_dir (default: {SUBSET})",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}. Run download_waymo_subset.py first.")

    tar_files = sorted(dataset_dir.rglob("*.tar"))
    if not tar_files:
        raise FileNotFoundError(f"No .tar files under {dataset_dir}. Run download_waymo_subset.py first.")

    total = 0
    for tar_path in tar_files:
        n = extract_tar(str(tar_path), str(dataset_dir), args.subset)
        total += n
        print(f"  {tar_path.name}: {n} samples")

    print(f"Extracted {total} samples to {dataset_dir / args.subset}.")
    print("Next: python data/tokenizer.py --dataset_dir", args.dataset_dir)
    print("Then: python scripts/train.py --dataset_dir", args.dataset_dir)


if __name__ == "__main__":
    main()
