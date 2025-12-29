import re
import random
import shutil
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

# Map UA-DETRAC vehicle types to YOLO class IDs
CLASS_MAP = {"car": 0, "bus": 1, "van": 2, "others": 3}

# Supported image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def clamp01(x: float) -> float:
    """Clamp a value to [0, 1] for safe YOLO normalization."""
    return max(0.0, min(1.0, x))


def natural_sort_key(s: str):
    """Sort strings with numbers correctly (img2 < img10)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def infer_frame_num(filename: str) -> int | None:
    """Extract frame number from filename (img00001.jpg → 1)."""
    m = re.search(r"(\d+)", filename)
    return int(m.group(1)) if m else None


def image_size(path: Path):
    """Read (W,H) of an image for normalization (requires Pillow)."""
    from PIL import Image
    with Image.open(path) as im:
        return im.size  # (W, H)


def parse_xml_boxes(xml_path: Path):
    """
    Parse UA-DETRAC XML and return:
    frame_num -> list of (class_id, left, top, width, height)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    frame_to_boxes = {}
    for frame_el in root.findall(".//frame"):
        fnum = int(frame_el.attrib.get("num"))
        boxes = []

        for target in frame_el.findall(".//target"):
            box_el = target.find("box")
            if box_el is None:
                continue

            left = float(box_el.attrib.get("left", 0))
            top = float(box_el.attrib.get("top", 0))
            w = float(box_el.attrib.get("width", 0))
            h = float(box_el.attrib.get("height", 0))

            attr_el = target.find("attribute")
            vtype = ""
            if attr_el is not None:
                vtype = (attr_el.attrib.get("vehicle_type") or "").strip().lower()

            class_id = CLASS_MAP.get(vtype, CLASS_MAP["others"])
            boxes.append((class_id, left, top, w, h))

        frame_to_boxes[fnum] = boxes

    return frame_to_boxes


def convert_sequence(seq_dir: Path, xml_path: Path, out_img_dir: Path, out_lbl_dir: Path):
    """Convert one sequence folder (MVI_xxxx) + its XML into YOLO images/labels."""
    seq_name = seq_dir.name
    frame_to_boxes = parse_xml_boxes(xml_path)

    images = sorted(
        [p for p in seq_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS],
        key=lambda p: natural_sort_key(p.name),
    )

    if not images:
        print(f"[WARN] No images found in {seq_dir}")
        return

    for img_path in images:
        fnum = infer_frame_num(img_path.name)
        if fnum is None:
            continue

        # Copy image to output split folder
        out_img_name = f"{seq_name}_{img_path.name}"
        shutil.copy2(img_path, out_img_dir / out_img_name)

        W, H = image_size(img_path)

        # Write YOLO label file
        label_path = out_lbl_dir / f"{seq_name}_{img_path.stem}.txt"
        boxes = frame_to_boxes.get(fnum, [])

        lines = []
        for class_id, left, top, w, h in boxes:
            x_center = (left + w / 2.0) / W
            y_center = (top + h / 2.0) / H
            w_norm = w / W
            h_norm = h / H

            x_center = clamp01(x_center)
            y_center = clamp01(y_center)
            w_norm = clamp01(w_norm)
            h_norm = clamp01(h_norm)

            if w_norm <= 0 or h_norm <= 0:
                continue

            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def write_data_yaml(out_root: Path):
    """Write YOLO data.yaml (only used for train mode)."""
    txt = f"""path: {out_root.as_posix()}
train: images/train
val: images/val

names:
  0: car
  1: bus
  2: van
  3: others
"""
    (out_root / "data.yaml").write_text(txt)


def main():
    """Prepare UA-DETRAC for YOLO: train mode creates train/val; test mode creates test only."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="train",
                    help="train: build train/val YOLO dataset. test: build test YOLO dataset.")
    ap.add_argument("--images_root", required=True,
                    help="Path to DETRAC-images/DETRAC-images (contains MVI_xxxx folders)")
    ap.add_argument("--ann_root", required=True,
                    help="Path to annotation XML folder (Train-Annotations-XML or Test-Annotations-XML)")
    ap.add_argument("--out_root", required=True,
                    help="Output YOLO dataset folder")
    ap.add_argument("--val_ratio", type=float, default=0.2,
                    help="Validation split ratio by sequence (train mode only)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for train/val split (train mode only)")
    args = ap.parse_args()

    split_mode = args.split
    images_root = Path(args.images_root).resolve()
    ann_root = Path(args.ann_root).resolve()
    out_root = Path(args.out_root).resolve()

    if not images_root.exists():
        raise RuntimeError(f"images_root not found: {images_root}")
    if not ann_root.exists():
        raise RuntimeError(f"ann_root not found: {ann_root}")

    # Find all sequence directories
    seq_dirs = sorted(
        [p for p in images_root.iterdir() if p.is_dir() and p.name.startswith("MVI_")],
        key=lambda p: p.name
    )

    # Keep only sequences that have corresponding XML in the provided ann_root
    sequences = []
    for sd in seq_dirs:
        xml_path = ann_root / f"{sd.name}.xml"
        if xml_path.exists():
            sequences.append((sd, xml_path))
        else:
            # In train mode, this is usually a test sequence; in test mode, usually a train sequence
            print(f"[INFO] {sd.name} has no annotation in {ann_root.name}, skipping.")

    if not sequences:
        raise RuntimeError("No matching image/XML sequences found. Check images_root and ann_root.")

    # Output folder structure differs by mode
    if split_mode == "train":
        (out_root / "images/train").mkdir(parents=True, exist_ok=True)
        (out_root / "images/val").mkdir(parents=True, exist_ok=True)
        (out_root / "labels/train").mkdir(parents=True, exist_ok=True)
        (out_root / "labels/val").mkdir(parents=True, exist_ok=True)

        # Split by sequences
        random.seed(args.seed)
        random.shuffle(sequences)
        n_val = max(1, int(len(sequences) * args.val_ratio))
        val_seqs = set(sd.name for sd, _ in sequences[:n_val])

        for sd, xml_path in sequences:
            split_name = "val" if sd.name in val_seqs else "train"
            print(f"[{split_name}] {sd.name}")
            convert_sequence(
                sd,
                xml_path,
                out_root / f"images/{split_name}",
                out_root / f"labels/{split_name}",
            )

        write_data_yaml(out_root)
        print("\n✅ Train/Val YOLO dataset ready:", out_root)
        print("✅ data.yaml:", out_root / "data.yaml")

    else:
        # test mode
        (out_root / "images/test").mkdir(parents=True, exist_ok=True)
        (out_root / "labels/test").mkdir(parents=True, exist_ok=True)

        for sd, xml_path in sequences:
            print(f"[test] {sd.name}")
            convert_sequence(
                sd,
                xml_path,
                out_root / "images/test",
                out_root / "labels/test",
            )

        print("\n✅ Test YOLO dataset ready:", out_root)


if __name__ == "__main__":
    main()
