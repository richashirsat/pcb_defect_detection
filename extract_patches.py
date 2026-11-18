"""
patch extraction script for PCB defect classification.
extracts 224x224 patches from DeepPCB and DsPCBSD+ datasets.

usage:
    python extract_patches.py --data_dir /path/to/aps360-pcb/data --output_dir /path/to/aps360-pcb/patches

Expects:
    data_dir/
        DeepPCB/PCBData/          (cloned from GitHub)
        DsPCBSD/Data_YOLO/        (downloaded from Figshare)
"""

import os
import argparse
import random
import glob
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# class mapping
# DeepPCB class IDs (from annotation files)
DEEPPCB_CLASSES = {
    1: "open",
    2: "short",
    3: "mousebite",
    4: "spur",
    5: "spurious_copper",
    6: "missing_hole",       # will be merged into "hole_defect"
}

# DsPCBSD+ YOLO class IDs (0-indexed)
# order from the paper: SH=0, SP=1, SC=2, OP=3, MB=4, HB=5, CS=6, CFO=7, BMFO=8
DSPCBSD_CLASSES = {
    0: "short",
    1: "spur",
    2: "spurious_copper",
    3: "open",
    4: "mousebite",
    5: "hole_defect",        # hole breakout -> merged "hole_defect"
    # 6, 7, 8 are dropped (conductor scratch, conductor foreign obj, base material foreign obj)
}

# unified class names (6 defect types + no_defect = 7 classes)
UNIFIED_CLASSES = ["open", "short", "mousebite", "spur", "spurious_copper", "hole_defect", "no_defect"]

PATCH_SIZE = 224


def extract_deeppcb(data_dir, output_dir):
    """extract patches from DeepPCB dataset."""
    pcb_dir = os.path.join(data_dir, "DeepPCB", "PCBData")

    # find all annotation files
    ann_files = sorted(glob.glob(os.path.join(pcb_dir, "**", "*.txt"), recursive=True))
    # filter to only annotation files (not README etc.)
    ann_files = [f for f in ann_files if os.path.basename(f)[0].isdigit()]

    print(f"Found {len(ann_files)} DeepPCB annotation files")
    patch_count = Counter()
    no_defect_count = 0

    for ann_path in ann_files:
        fname = os.path.basename(ann_path).replace(".txt", "")
        ann_dir = os.path.dirname(ann_path)                    # e.g. .../group00041/00041_not
        img_dir = ann_dir.replace("_not", "")                  # e.g. .../group00041/00041
        test_img_path = os.path.join(img_dir, fname + "_test.jpg")
        temp_img_path = os.path.join(img_dir, fname + "_temp.jpg")  

        if not os.path.exists(test_img_path) or not os.path.exists(temp_img_path):
            continue

        test_img = cv2.imread(test_img_path)
        temp_img = cv2.imread(temp_img_path)
        if test_img is None or temp_img is None:
            continue

        img_h, img_w = test_img.shape[:2]
        img_id = fname

        # read annotations: x1,y1,x2,y2,type
        with open(ann_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) < 5:
                continue
            x1, y1, x2, y2 = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            cls_id = int(parts[4])

            if cls_id not in DEEPPCB_CLASSES:
                continue

            class_name = DEEPPCB_CLASSES[cls_id]
            if class_name == "missing_hole":
                class_name = "hole_defect"

            # crop and resize
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(img_w, x2), min(img_h, y2)
            patch = test_img[y1c:y2c, x1c:x2c]
            if patch.size == 0:
                continue
            patch = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE))

            # save
            cls_dir = os.path.join(output_dir, class_name)
            os.makedirs(cls_dir, exist_ok=True)
            fname = f"deeppcb_{img_id}_{i}.jpg"
            cv2.imwrite(os.path.join(cls_dir, fname), patch)
            patch_count[class_name] += 1

        # extract no-defect patches from template image
        # sample up to 2 random patches per template
        for j in range(2):
            rx = random.randint(0, max(0, img_w - PATCH_SIZE))
            ry = random.randint(0, max(0, img_h - PATCH_SIZE))
            patch = temp_img[ry:ry+PATCH_SIZE, rx:rx+PATCH_SIZE]
            if patch.shape[0] != PATCH_SIZE or patch.shape[1] != PATCH_SIZE:
                continue

            cls_dir = os.path.join(output_dir, "no_defect")
            os.makedirs(cls_dir, exist_ok=True)
            fname = f"deeppcb_{img_id}_nodefect_{j}.jpg"
            cv2.imwrite(os.path.join(cls_dir, fname), patch)
            no_defect_count += 1

    patch_count["no_defect"] = no_defect_count
    print(f"DeepPCB patches extracted: {dict(patch_count)}")
    return patch_count


def extract_dspcbsd(data_dir, output_dir):
    """Extract patches from DsPCBSD+ YOLO format."""
    yolo_dir = os.path.join(data_dir, "DsPCBSD", "Data_YOLO")
    patch_count = Counter()

    for split in ["train", "val"]:
        img_dir = os.path.join(yolo_dir, "images", split)
        lbl_dir = os.path.join(yolo_dir, "labels", split)

        if not os.path.isdir(img_dir):
            print(f"Warning: {img_dir} not found, skipping")
            continue

        img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        print(f"Found {len(img_files)} DsPCBSD+ {split} images")

        for img_path in img_files:
            img = cv2.imread(img_path)
            if img is None:
                continue

            img_h, img_w = img.shape[:2]
            img_id = Path(img_path).stem
            lbl_path = os.path.join(lbl_dir, img_id + ".txt")

            if not os.path.exists(lbl_path):
                continue

            with open(lbl_path, "r") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]

            has_defect = False
            for i, line in enumerate(lines):
                parts = line.split()
                if len(parts) < 5:
                    continue

                cls_id = int(parts[0])
                if cls_id not in DSPCBSD_CLASSES:
                    continue  # skip classes 6, 7, 8

                class_name = DSPCBSD_CLASSES[cls_id]
                has_defect = True

                # YOLO format: cx, cy, w, h (normalized)
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = int((cx - bw / 2) * img_w)
                y1 = int((cy - bh / 2) * img_h)
                x2 = int((cx + bw / 2) * img_w)
                y2 = int((cy + bh / 2) * img_h)

                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(img_w, x2), min(img_h, y2)
                patch = img[y1c:y2c, x1c:x2c]
                if patch.size == 0:
                    continue
                patch = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE))

                cls_dir = os.path.join(output_dir, class_name)
                os.makedirs(cls_dir, exist_ok=True)
                fname = f"dspcbsd_{split}_{img_id}_{i}.jpg"
                cv2.imwrite(os.path.join(cls_dir, fname), patch)
                patch_count[class_name] += 1

    print(f"DsPCBSD+ patches extracted: {dict(patch_count)}")
    return patch_count


def split_dataset(output_dir, split_dir):
    """Split patches into train/val/test (70/15/15), stratified, grouped by source image."""
    print("\nSplitting dataset...")

    all_files = []
    all_labels = []
    all_groups = []

    for cls_name in UNIFIED_CLASSES:
        cls_dir = os.path.join(output_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if not fname.endswith(".jpg"):
                continue
            all_files.append(os.path.join(cls_dir, fname))
            all_labels.append(cls_name)
            # Group by source image ID (everything before the last _N.jpg)
            parts = fname.rsplit("_", 1)
            group_id = parts[0] if len(parts) > 1 else fname
            all_groups.append(group_id)

    all_files = np.array(all_files)
    all_labels = np.array(all_labels)
    all_groups = np.array(all_groups)

    print(f"Total patches: {len(all_files)}")
    print(f"Class distribution: {dict(Counter(all_labels))}")

    # first split: 70% train, 30% temp
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss1.split(all_files, all_labels, all_groups))

    # second split: split the 30% into 15/15 (50/50 of temp)
    temp_files = all_files[temp_idx]
    temp_labels = all_labels[temp_idx]
    temp_groups = all_groups[temp_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(gss2.split(temp_files, temp_labels, temp_groups))

    splits = {
        "train": all_files[train_idx],
        "val": temp_files[val_idx],
        "test": temp_files[test_idx],
    }

    # copy files into split directories
    for split_name, files in splits.items():
        for fpath in files:
            cls_name = os.path.basename(os.path.dirname(fpath))
            dst_dir = os.path.join(split_dir, split_name, cls_name)
            os.makedirs(dst_dir, exist_ok=True)
            fname = os.path.basename(fpath)
            dst = os.path.join(dst_dir, fname)
            # Copy file
            img = cv2.imread(fpath)
            if img is not None:
                cv2.imwrite(dst, img)

    # print split stats
    for split_name, files in splits.items():
        labels = [os.path.basename(os.path.dirname(f)) for f in files]
        print(f"\n{split_name}: {len(files)} patches")
        for cls, count in sorted(Counter(labels).items()):
            print(f"  {cls}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Extract PCB defect patches")
    parser.add_argument("--data_dir", required=True, help="Path to data/ folder containing DeepPCB and DsPCBSD")
    parser.add_argument("--output_dir", default=None, help="Where to save raw patches (default: data_dir/patches_raw)")
    parser.add_argument("--split_dir", default=None, help="Where to save train/val/test split (default: data_dir/patches_split)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, "patches_raw")
    if args.split_dir is None:
        args.split_dir = os.path.join(args.data_dir, "patches_split")

    random.seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("STEP 1: Extracting DeepPCB patches")
    print("=" * 60)
    deeppcb_counts = extract_deeppcb(args.data_dir, args.output_dir)

    print("\n" + "=" * 60)
    print("STEP 2: Extracting DsPCBSD+ patches")
    print("=" * 60)
    dspcbsd_counts = extract_dspcbsd(args.data_dir, args.output_dir)

    # Combined stats
    total = deeppcb_counts + dspcbsd_counts
    print("\n" + "=" * 60)
    print("COMBINED CLASS DISTRIBUTION")
    print("=" * 60)
    for cls in UNIFIED_CLASSES:
        print(f"  {cls}: {total[cls]}")
    print(f"  TOTAL: {sum(total.values())}")

    print("\n" + "=" * 60)
    print("STEP 3: Splitting into train/val/test")
    print("=" * 60)
    split_dataset(args.output_dir, args.split_dir)

    print("\nDone! Patches saved to:")
    print(f"  Raw:   {args.output_dir}")
    print(f"  Split: {args.split_dir}")


if __name__ == "__main__":
    main()
