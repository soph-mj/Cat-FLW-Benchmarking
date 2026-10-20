"""
compare_experiments.py

Runs the full comparison grid -- {coord, heatmap} x {each backbone} -- and
produces a summary table + bar chart for your report. This is the main
script tying together the "end-to-end CNN approach comparison" ablation.

Usage:
    python compare_experiments.py --data_dir data/dummy --epochs 20
"""

import argparse
import json
import os

import pandas as pd

from models import AVAILABLE_BACKBONES
from eval_utils import plot_comparison_bar
from train import main as train_main


class Args:
    """Simple namespace mimicking argparse.Namespace, so we can call train.main()
    programmatically for each grid cell."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def main(data_dir, out_dir, epochs, batch_size, lr, seed, pretrained=True):
    approaches = ["coord", "heatmap"]
    results = {}

    for approach in approaches:
        for backbone in AVAILABLE_BACKBONES:
            key = f"{approach}_{backbone}"
            print(f"\n{'='*60}\nRunning: {key}\n{'='*60}")
            args = Args(
                approach=approach, backbone=backbone, data_dir=data_dir,
                out_dir=os.path.join(out_dir, key), epochs=epochs,
                batch_size=batch_size, lr=lr, seed=seed, pretrained=pretrained,
            )
            test_nrmse = train_main(args)
            results[key] = test_nrmse

    # Save results table
    df = pd.DataFrame(
        [{"approach": k.split("_")[0], "backbone": "_".join(k.split("_")[1:]), "test_nrmse_pct": v}
         for k, v in results.items()]
    )
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "comparison_results.csv"), index=False)
    with open(os.path.join(out_dir, "comparison_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n\nFINAL COMPARISON TABLE")
    print(df.to_string(index=False))

    plot_comparison_bar(results, metric_name="Test NRMSE (%)",
                         save_path=os.path.join(out_dir, "comparison_chart.png"))
    print(f"\nSaved results to {out_dir}/comparison_results.csv and comparison_chart.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/dummy")
    parser.add_argument("--out_dir", type=str, default="runs/comparison")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    args = parser.parse_args()
    main(args.data_dir, args.out_dir, args.epochs, args.batch_size, args.lr, args.seed, args.pretrained)
