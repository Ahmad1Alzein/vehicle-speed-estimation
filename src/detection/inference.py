from __future__ import annotations

import argparse
from pathlib import Path


def run_test_inference(
    weights: str | Path,
    test_images_root: str | Path,
    imgsz: int = 640,
    conf: float = 0.25,
    save: bool = False,
    save_txt: bool = True,
    project: str | Path = "runs",
    name: str = "test_predictions",
):
    """
    Run YOLO inference on the TEST split (images only), and save results.
    - save_txt=True will write YOLO-format prediction txt files.
    Outputs go to: {project}/{name}/
    """
    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    test_images_root = Path(test_images_root)
    if not test_images_root.exists():
        raise FileNotFoundError(f"test images root not found: {test_images_root}")

    from ultralytics import YOLO

    model = YOLO(str(weights))

    model.predict(
        source=str(test_images_root),
        imgsz=imgsz,
        conf=conf,
        save=save,
        save_txt=save_txt,
        project=str(project),
        name=name,
    )


def main():
    ap = argparse.ArgumentParser(description="Run YOLO inference on UA-DETRAC test images.")
    ap.add_argument("--weights", required=True, help="Path to best.pt")
    ap.add_argument("--source", required=True, help="Path to test images folder (e.g. .../uadetrac_test/images/test)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--save", action="store_true", help="Save annotated images")
    ap.add_argument("--save_txt", action="store_true", default=True, help="Save YOLO txt predictions")
    ap.add_argument("--project", default="runs", help="Output directory root")
    ap.add_argument("--name", default="test_predictions", help="Run folder name")
    args = ap.parse_args()

    run_test_inference(
        weights=args.weights,
        test_images_root=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        save=args.save,
        save_txt=args.save_txt,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
