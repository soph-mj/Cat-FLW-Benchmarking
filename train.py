"""
train.py

Trains ONE model (a given approach x backbone combination) and evaluates it.
Use compare_experiments.py to run the full grid across approaches/backbones.

"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from catflw_dataset import CatFLWDataset, load_annotations, train_val_test_split
from models import build_model, wing_loss, heatmap_mse_loss, soft_argmax, HEATMAP_SIZE
from eval_utils import compute_nrmse, visualize_predictions


def train_one_epoch(model, loader, optimizer, device, approach):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()
        preds = model(images)

        if approach == "coord" and not args.mse_loss:
            loss = wing_loss(preds, targets)
        elif approach == "coord" and args.mse_loss:
            import torch.nn.functional as F
            loss = F.mse_loss(preds, targets)
        else:
            loss = heatmap_mse_loss(preds, targets)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, approach, img_size=224, heatmap_size=HEATMAP_SIZE):
    model.eval()
    all_pred, all_gt = [], []

    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(device)
        gt = batch["landmarks_raw"].to(device)  # (batch, N, 2) in image-pixel coords
        preds = model(images)

        if approach == "coord":
            preds = preds.view(gt.shape)  # (batch, N, 2), already in image-pixel coords
        else:
            coords = soft_argmax(preds)  # (batch, N, 2) in heatmap-pixel coords
            preds = coords * (img_size / heatmap_size)  # rescale to image-pixel coords

        all_pred.append(preds.cpu())
        all_gt.append(gt.cpu())

    all_pred = torch.cat(all_pred, dim=0)
    all_gt = torch.cat(all_gt, dim=0)
    nrmse = compute_nrmse(all_pred, all_gt, image_size=img_size)
    return nrmse, all_pred, all_gt


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    records, image_dir = load_annotations(args.data_dir)
    train_recs, val_recs, test_recs = train_val_test_split(records, seed=args.seed)

    target_type = args.approach
    train_ds = CatFLWDataset(image_dir, train_recs, target_type=target_type,
                              use_alignment=args.alignment, use_edge_filter=args.edge_filter, augment=True, seed=args.seed)
    val_ds = CatFLWDataset(image_dir, val_recs, target_type=target_type, use_alignment=args.alignment, use_edge_filter=args.edge_filter)
    test_ds = CatFLWDataset(image_dir, test_recs, target_type=target_type, use_alignment=args.alignment, use_edge_filter=args.edge_filter)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args.approach, args.backbone, pretrained=args.pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_nrmse = float("inf")
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"{args.approach}_{args.backbone}.pt")

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.approach)
        val_nrmse, _, _ = evaluate(model, val_loader, device, args.approach)
        print(f"[{args.approach}/{args.backbone}] epoch {epoch+1}/{args.epochs} "
              f"train_loss={train_loss:.4f} val_nrmse={val_nrmse:.2f}%")

        if val_nrmse < best_val_nrmse:
            best_val_nrmse = val_nrmse
            torch.save(model.state_dict(), ckpt_path)

    # Final test evaluation using best checkpoint
    model.load_state_dict(torch.load(ckpt_path))
    test_nrmse, test_pred, test_gt = evaluate(model, test_loader, device, args.approach)
    print(f"\nFINAL TEST NRMSE [{args.approach}/{args.backbone}]: {test_nrmse:.2f}%")

    # Save a couple of qualitative visualizations for the report
    vis_dir = os.path.join(args.out_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    sample_batch = next(iter(test_loader))
    for i in range(min(3, sample_batch["image"].size(0))):
        visualize_predictions(
            sample_batch["image"][i],
            test_pred[i] if i < len(test_pred) else test_pred[0],
            test_gt[i] if i < len(test_gt) else test_gt[0],
            save_path=os.path.join(vis_dir, f"{args.approach}_{args.backbone}_sample{i}.png"),
            title=f"{args.approach} / {args.backbone} (test NRMSE {test_nrmse:.2f}%)",
        )

    return test_nrmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", choices=["coord", "heatmap"], required=True)
    parser.add_argument("--backbone", type=str, default="shufflenet_v2_x1_0")
    parser.add_argument("--data_dir", type=str, default="data/dummy")
    parser.add_argument("--out_dir", type=str, default="runs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alignment", action="store_true", default=True)
    parser.add_argument("--no-alignment", dest="alignment", action="store_false")
    parser.add_argument("--edge-filter", action="store_true", default=False)
    parser.add_argument("--mse-loss", action="store_true", default=False)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                         help="Train backbone from scratch (e.g. if offline / no internet access to download weights)")
    args = parser.parse_args()
    main(args)
