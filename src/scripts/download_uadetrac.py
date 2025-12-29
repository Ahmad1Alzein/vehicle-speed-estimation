from pathlib import Path
import kagglehub
import json

def main(output_file: Path):
    ua_root = kagglehub.dataset_download("bratjay/ua-detrac-orig")

    ua_root = Path(ua_root)
    if not ua_root.exists():
        raise RuntimeError("UA-DETRAC download failed")

    # Save the resolved path for other steps
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({"ua_detrac_root": str(ua_root)}, f, indent=2)

    print(f"UA-DETRAC ready at: {ua_root}")
    print(f"Path saved to: {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="runtime/paths.json",
        help="Where to store resolved dataset paths"
    )
    args = parser.parse_args()

    main(Path(args.out))
