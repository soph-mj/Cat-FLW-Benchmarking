"""
inspect_annotations.py

Run this AFTER downloading CatFLW (download_catflw.py) to see exactly what
format the real annotation file is in. This tells you what to change in
load_annotations() inside catflw_dataset.py.

"""

import argparse
import json
import os


def inspect_directory(data_dir):
    print(f"\n{'='*60}\nContents of {data_dir}\n{'='*60}")
    for root, dirs, files in os.walk(data_dir):
        level = root.replace(data_dir, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = "  " * (level + 1)
        for f in files[:10]:  # cap so huge image folders don't flood output
            print(f"{subindent}{f}")
        if len(files) > 10:
            print(f"{subindent}... and {len(files) - 10} more files")


def inspect_file(path):
    ext = os.path.splitext(path)[1].lower()
    print(f"\n{'='*60}\nInspecting: {path}\n{'='*60}")

    if ext == ".json":
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            print(f"Type: list of {len(data)} records")
            print("First record:")
            print(json.dumps(data[0], indent=2)[:1000])
        elif isinstance(data, dict):
            print(f"Type: dict with keys: {list(data.keys())[:20]}")
            first_key = list(data.keys())[0]
            print(f"\nExample value for key '{first_key}':")
            print(json.dumps(data[first_key], indent=2)[:1000])

    elif ext == ".csv":
        with open(path, "r") as f:
            lines = [next(f) for _ in range(5)]
        print("First 5 lines:")
        for line in lines:
            print(line.strip()[:300])

    else:
        print(f"Unrecognized extension '{ext}' -- open manually to inspect.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/catflw")
    parser.add_argument("--file", type=str, default=None,
                         help="Specific annotation file to inspect in detail (json or csv)")
    args = parser.parse_args()

    inspect_directory(args.data_dir)

    if args.file:
        inspect_file(args.file)
    else:
        print(
            "\nNEXT STEP: find the annotation file (likely .json or .csv) in the "
            "listing above, then re-run with:\n"
            f"  python inspect_annotations.py --data_dir {args.data_dir} --file <path_to_annotation_file>\n"
            "Paste the output back if you want help adapting load_annotations()."
        )
