from __future__ import annotations

import argparse
from pathlib import Path


def train_yolo(
    data_yaml: str | Path,
    base_model: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,   # e.g. "0" or "cpu"
    project: str | Path = "runs",
    name: str = "uadetrac_yolo",
):
    """
    Train/finetune a YOLO model on a YOLO-format dataset (train/val already prepared).
    Output weights will be saved under: {project}/{name}/weights/best.pt
    """
    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    # Import here so the file can be imported without ultralytics installed
    from ultralytics import YOLO

    model = YOLO(base_model)

    kwargs = dict(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(project),
        name=name,
    )
    if device is not None:
        kwargs["device"] = device

    model.train(**kwargs)


def main():
    ap = argparse.ArgumentParser(description="Train YOLO on UA-DETRAC (YOLO-ready train/val split).")
    ap.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    ap.add_argument("--model", default="yolov8n.pt", help="Base model: yolov8n.pt, yolov8s.pt, ...")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help='e.g. "0" for GPU, "cpu" for CPU, default auto')
    ap.add_argument("--project", default="runs", help="Output directory root")
    ap.add_argument("--name", default="uadetrac_yolo", help="Run folder name")
    args = ap.parse_args()

    train_yolo(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
