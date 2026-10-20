import argparse
import os
import subprocess
import sys


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", "georgemartvel/catflw",
             "-p", out_dir, "--unzip"],
            check=True,
        )
        print(f"Downloaded and extracted CatFLW to: {out_dir}")
    except FileNotFoundError:
        print(
            "The 'kaggle' CLI is not installed or not on PATH.\n"
            "Install with: pip install kaggle\n"
            "Then set up your API token as described in this script's docstring."
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(
            "Download failed. Common causes:\n"
            "  - kaggle.json API token not set up (~/.kaggle/kaggle.json)\n"
            "  - You haven't accepted the dataset's terms on the Kaggle website yet\n"
            f"Original error: {e}"
        )
        sys.exit(1)

    print(
        "\nNEXT STEP: inspect the extracted folder structure and check "
        "annotation format (likely a CSV or JSON with 48 landmark coordinates "
        "per image). Update `catflw_dataset.py`'s `load_annotations()` to match "
        "whatever the actual file structure/format turns out to be -- dataset "
        "annotation formats sometimes differ slightly from their paper description."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/catflw")
    args = parser.parse_args()
    main(args.out_dir)
