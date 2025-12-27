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
    """Clamp a value to the range [0, 1] (YOLO safety normalization)."""
    return max(0.0, min(1.0, x))


def natural_sort_key(s: str):
    """Sort filenames numerically (img2 < img10 instead of img10 < img2)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def infer_frame_num(filename: str) -> int | None:
    """Extract frame number from image filename (img00001.jpg → 1)."""
    m = re.search(r"(\d+)", filename)
    return int(m.group(1)) if m else None


def image_size(path: Path):
    """Read image width and height (required for YOLO normalization)."""
    from PIL import Image
    with Image.open(path) as im:
        return im.size  # (W, H)


def parse_xml_boxes(xml_path: Path):
    """
    Parse a UA-DETRAC XML file and return:
    frame_num → list of (class_id, left, top, width, height)
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

            # Bounding box in pixel coordinates
            left = float(box_el.attrib.get("left", 0))
            top = float(box_el.attrib.get("top", 0))
            w = float(box_el.attrib.get("width", 0))
            h = float(box_el.attrib.get("height", 0))

            # Vehicle type → class id
            attr_el = target.find("attribute")
            vtype = ""
            if attr_el is not None:
                vtype = (attr_el.attrib.get("vehicle_type") or "").strip().lower()

            class_id = CLASS_MAP.get(vtype, CLASS_MAP["others"])
            boxes.append((class_id, left, top, w, h))

        frame_to_boxes[fnum] = boxes

    return frame_to_boxes


def convert_sequence(seq_dir: Path, xml_path: Path, out_img_dir: Path, out_lbl_dir: Path):
    """
    Convert one video sequence (MVI_xxxx) from UA-DETRAC format
    into YOLO image + label files.
    """
    seq_name = seq_dir.name
    frame_to_boxes = parse_xml_boxes(xml_path)

    # Collect and sort frames
    images = sorted(
        [p for p in seq_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS],
        key=lambda p: natural_sort_key(p.name),
    )

    if not images:
        print(f"WARNING: No images found in {seq_dir}")
        return

    for img_path in images:
        fnum = infer_frame_num(img_path.name)
        if fnum is None:
            continue

        # Copy image to YOLO dataset
        out_img_name = f"{seq_name}_{img_path.name}"
        dst_img = out_img_dir / out_img_name
        shutil.copy2(img_path, dst_img)

        # Image dimensions
        W, H = image_size(img_path)

        # Create YOLO label file
        label_path = out_lbl_dir / f"{seq_name}_{img_path.stem}.txt"
        boxes = frame_to_boxes.get(fnum, [])

        lines = []
        for class_id, left, top, w, h in boxes:
            # Convert from pixel box → YOLO normalized format
            x_center = (left + w / 2.0) / W
            y_center = (top + h / 2.0) / H
            w_norm = w / W
            h_norm = h / H

            # Safety clamp
            x_center = clamp01(x_center)
            y_center = clamp01(y_center)
            w_norm = clamp01(w_norm)
            h_norm = clamp01(h_norm)

            if w_norm <= 0 or h_norm <= 0:
                continue

            lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            )

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def write_data_yaml(out_root: Path):
    """Create YOLO data.yaml file describing paths and class names."""
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
    """Entry point: split sequences, convert data, and build YOLO dataset."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_root", required=True,
                    help="Path to DETRAC-images/DETRAC-images")
    ap.add_argument("--ann_root", required=True,
                    help="Path to DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML")
    ap.add_argument("--out_root", required=True,
                    help="Output YOLO dataset folder")
    ap.add_argument("--val_ratio", type=float, default=0.2,
                    help="Validation split ratio (by sequence, not frames)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    images_root = Path(args.images_root).resolve()
    ann_root = Path(args.ann_root).resolve()
    out_root = Path(args.out_root).resolve()

    # Find all sequences with matching XML files
    seq_dirs = sorted(
        [p for p in images_root.iterdir() if p.is_dir() and p.name.startswith("MVI_")]
    )

    sequences = []
    for sd in seq_dirs:
        xml_path = ann_root / f"{sd.name}.xml"
        if xml_path.exists():
            sequences.append((sd, xml_path))
        else:
            print(f"WARNING: Missing annotation for {sd.name}")

    if not sequences:
        raise RuntimeError("No matching image/XML sequences found.")

    # Split sequences into train/val (prevents data leakage)
    random.seed(args.seed)
    random.shuffle(sequences)
    n_val = max(1, int(len(sequences) * args.val_ratio))
    val_seqs = set(sd.name for sd, _ in sequences[:n_val])

    # Create YOLO directory structure
    (out_root / "images/train").mkdir(parents=True, exist_ok=True)
    (out_root / "images/val").mkdir(parents=True, exist_ok=True)
    (out_root / "labels/train").mkdir(parents=True, exist_ok=True)
    (out_root / "labels/val").mkdir(parents=True, exist_ok=True)

    # Convert each sequence
    for sd, xml_path in sequences:
        split = "val" if sd.name in val_seqs else "train"
        print(f"[{split}] {sd.name}")
        convert_sequence(
            sd,
            xml_path,
            out_root / f"images/{split}",
            out_root / f"labels/{split}",
        )

    write_data_yaml(out_root)
    print("\n✅ YOLO dataset ready:", out_root)


if __name__ == "__main__":
    main()
