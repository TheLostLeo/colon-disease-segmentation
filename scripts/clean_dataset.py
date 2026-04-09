#!/usr/bin/env python3
"""Create a cleaned EBHI-SEG dataset with only valid image/mask pairs.

Outputs (inside the output folder, default: data):
- cleaned/<class>/image/*.png
- cleaned/<class>/label/*.png (multiclass-ready mask: 0=background, class_id=foreground)
- master_pairs.csv
- missing_masks.csv
- missing_images.csv
- class_distribution.csv
- summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image


KNOWN_CLASS_ORDER = [
    "Adenocarcinoma",
    "High-grade IN",
    "Low-grade IN",
    "Normal",
    "Polyp",
    "Serrated adenoma",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean EBHI-SEG by removing unpaired files")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("EBHI-SEG"),
        help="Source dataset directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data"),
        help="Output directory for cleaned data and reports",
    )
    return parser.parse_args()


def list_png_files(folder: Path) -> Dict[str, Path]:
    if not folder.exists():
        return {}
    return {
        p.name: p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".png"
    }


def ensure_output_layout(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir = output_dir / "cleaned"

    if cleaned_dir.exists():
        shutil.rmtree(cleaned_dir)
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    for report_name in [
        "master_pairs.csv",
        "missing_masks.csv",
        "missing_images.csv",
        "class_distribution.csv",
        "summary.json",
    ]:
        report_path = output_dir / report_name
        if report_path.exists():
            report_path.unlink()

    return cleaned_dir


def write_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_class_mapping(class_names: List[str]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}

    next_id = 1
    for class_name in KNOWN_CLASS_ORDER:
        if class_name in class_names:
            mapping[class_name] = next_id
            next_id += 1

    for class_name in class_names:
        if class_name not in mapping:
            mapping[class_name] = next_id
            next_id += 1

    return mapping


def convert_mask_to_class_id(src_mask: Path, dst_mask: Path, class_id: int) -> None:
    """Convert a grayscale mask to multiclass format for one class.

    Rule:
    - background pixels -> 0
    - foreground pixels (raw mask > 0) -> class_id
    """
    with Image.open(src_mask) as im:
        raw_mask = np.array(im.convert("L"), dtype=np.uint8)

    converted_mask = np.zeros_like(raw_mask, dtype=np.uint8)
    converted_mask[raw_mask > 0] = np.uint8(class_id)

    Image.fromarray(converted_mask, mode="L").save(dst_mask)


def main() -> int:
    args = parse_args()

    source_dir = args.source.resolve()
    output_dir = args.output.resolve()

    if not source_dir.exists():
        print(f"ERROR: Source folder not found: {source_dir}")
        return 1

    class_dirs = sorted([p for p in source_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    class_names = [p.name for p in class_dirs]

    if not class_dirs:
        print(f"ERROR: No class directories found under {source_dir}")
        return 1

    class_to_id = build_class_mapping(class_names)
    cleaned_dir = ensure_output_layout(output_dir)

    master_rows: List[dict] = []
    missing_mask_rows: List[dict] = []
    missing_image_rows: List[dict] = []
    distribution_rows: List[dict] = []

    sample_id = 1

    totals = {
        "images": 0,
        "masks": 0,
        "paired": 0,
        "missing_masks": 0,
        "missing_images": 0,
    }

    for class_dir in class_dirs:
        class_name = class_dir.name
        class_id = class_to_id[class_name]

        image_dir = class_dir / "image"
        mask_dir = class_dir / "label"

        image_files = list_png_files(image_dir)
        mask_files = list_png_files(mask_dir)

        image_names = set(image_files)
        mask_names = set(mask_files)

        paired_names = sorted(image_names & mask_names)
        missing_masks = sorted(image_names - mask_names)
        missing_images = sorted(mask_names - image_names)

        totals["images"] += len(image_names)
        totals["masks"] += len(mask_names)
        totals["paired"] += len(paired_names)
        totals["missing_masks"] += len(missing_masks)
        totals["missing_images"] += len(missing_images)

        class_clean_img_dir = cleaned_dir / class_name / "image"
        class_clean_mask_dir = cleaned_dir / class_name / "label"
        class_clean_img_dir.mkdir(parents=True, exist_ok=True)
        class_clean_mask_dir.mkdir(parents=True, exist_ok=True)

        for filename in paired_names:
            src_image = image_files[filename]
            src_mask = mask_files[filename]

            dst_image = class_clean_img_dir / filename
            dst_mask = class_clean_mask_dir / filename

            shutil.copy2(src_image, dst_image)
            convert_mask_to_class_id(src_mask, dst_mask, class_id)

            master_rows.append(
                {
                    "sample_id": sample_id,
                    "class_name": class_name,
                    "class_id": class_id,
                    "filename": filename,
                    "image_path": str(Path("cleaned") / class_name / "image" / filename),
                    "mask_path": str(Path("cleaned") / class_name / "label" / filename),
                }
            )
            sample_id += 1

        for filename in missing_masks:
            missing_mask_rows.append(
                {
                    "class_name": class_name,
                    "class_id": class_id,
                    "filename": filename,
                    "image_path": str(image_dir / filename),
                }
            )

        for filename in missing_images:
            missing_image_rows.append(
                {
                    "class_name": class_name,
                    "class_id": class_id,
                    "filename": filename,
                    "mask_path": str(mask_dir / filename),
                }
            )

        distribution_rows.append(
            {
                "class_name": class_name,
                "class_id": class_id,
                "paired_samples": len(paired_names),
                "images_total": len(image_names),
                "masks_total": len(mask_names),
                "missing_masks": len(missing_masks),
                "missing_images": len(missing_images),
            }
        )

    write_csv(
        output_dir / "master_pairs.csv",
        ["sample_id", "class_name", "class_id", "filename", "image_path", "mask_path"],
        master_rows,
    )
    write_csv(
        output_dir / "missing_masks.csv",
        ["class_name", "class_id", "filename", "image_path"],
        missing_mask_rows,
    )
    write_csv(
        output_dir / "missing_images.csv",
        ["class_name", "class_id", "filename", "mask_path"],
        missing_image_rows,
    )
    write_csv(
        output_dir / "class_distribution.csv",
        [
            "class_name",
            "class_id",
            "paired_samples",
            "images_total",
            "masks_total",
            "missing_masks",
            "missing_images",
        ],
        distribution_rows,
    )

    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "classes": class_names,
        "class_to_id": class_to_id,
        "totals": totals,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Clean dataset build complete")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(
        "Totals: "
        f"images={totals['images']} "
        f"masks={totals['masks']} "
        f"paired={totals['paired']} "
        f"missing_masks={totals['missing_masks']} "
        f"missing_images={totals['missing_images']}"
    )
    print(f"Master CSV: {output_dir / 'master_pairs.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
