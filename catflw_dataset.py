"""
catflw_dataset.py

Dataset loading + preprocessing for CatFLW (48 facial landmarks per cat face).

REAL CATFLW FORMAT (confirmed from actual download):
  <data_dir>/CatFLW dataset/images/<name>.png
  <data_dir>/CatFLW dataset/labels/<name>.json   -- one label file per image, same stem

  Each label JSON looks like:
      {
        "labels": [[x1, y1], [x2, y2], ..., 48 points total],
        "bounding_boxes": [...]
      }

`load_annotations()` below scans the labels/ folder, matches each label file
to its image by filename stem, and returns records shaped like:
    {"image": <path relative to image_dir>, "landmarks": [[x, y], ...] (48 points),
     "bbox": [...] (optional)}

`generate_dummy_data()` produces synthetic placeholder data of the same shape,
for testing before/without the real download.
"""

import glob
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_SIZE = 224
N_LANDMARKS = 48
HEATMAP_SIZE = 112  # 1/4 of IMG_SIZE, typical for heatmap-based models
HEATMAP_SIGMA = 2.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _find_images_and_labels_dirs(data_dir):
    """CatFLW's extracted zip nests an extra folder (e.g. 'CatFLW dataset/')
    between data_dir and images/labels -- search for them rather than
    hardcoding the exact nesting, since this can vary by download source."""
    images_dirs = glob.glob(os.path.join(data_dir, "**", "images"), recursive=True)
    labels_dirs = glob.glob(os.path.join(data_dir, "**", "labels"), recursive=True)
    if not images_dirs or not labels_dirs:
        raise FileNotFoundError(
            f"Could not find 'images' and 'labels' folders under {data_dir}. "
            f"Found images dirs: {images_dirs}, labels dirs: {labels_dirs}"
        )
    return images_dirs[0], labels_dirs[0]


def load_annotations(data_dir):
    """
    Scans the real CatFLW images/ and labels/ folders and builds the record
    list. `data_dir` should be the folder passed to download_catflw.py
    (e.g. "data/catflw") -- NOT a path to a specific annotation file.

    Returns: (records, image_dir)
      records: list of {"image": <filename>, "landmarks": [[x,y],...], "bbox": [...]}
      image_dir: absolute path to the images/ folder (pass this to CatFLWDataset)
    """
    images_dir, labels_dir = _find_images_and_labels_dirs(data_dir)

    label_files = sorted(glob.glob(os.path.join(labels_dir, "*.json")))
    records = []
    skipped = 0

    for label_path in label_files:
        stem = os.path.splitext(os.path.basename(label_path))[0]
        # Images are .png per the confirmed listing; adjust if some are .jpg
        image_path = os.path.join(images_dir, f"{stem}.png")
        if not os.path.exists(image_path):
            skipped += 1
            continue

        with open(label_path, "r") as f:
            label_data = json.load(f)

        landmarks = label_data.get("labels")
        if landmarks is None or len(landmarks) != N_LANDMARKS:
            skipped += 1
            continue

        image_path_check = os.path.join(images_dir, f"{stem}.png")
        img_check = cv2.imread(image_path_check)
        if img_check is None:
            print(f"load_annotations: skipping unreadable image {stem}.png")
            skipped += 1
            continue

        records.append({
            "image": f"{stem}.png",
            "landmarks": landmarks,
            "bbox": label_data.get("bounding_boxes"),
        })

    if skipped:
        print(f"load_annotations: skipped {skipped} label files (missing image or "
              f"unexpected landmark count)")
    print(f"load_annotations: loaded {len(records)} valid records")

    return records, images_dir


def generate_dummy_data(out_dir, n_images=200, seed=0):
    """Creates placeholder images + 48-point landmark annotations shaped like
    CatFLW, so the pipeline can be built/tested before the real download."""
    rng = np.random.default_rng(seed)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    records = []
    for i in range(n_images):
        img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 200
        center = (IMG_SIZE // 2, IMG_SIZE // 2)
        axes = (int(rng.integers(70, 90)), int(rng.integers(80, 100)))
        color = tuple(int(c) for c in rng.integers(120, 200, size=3))
        cv2.ellipse(img, center, axes, 0, 0, 360, color, -1)
        noise = rng.normal(0, 10, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        fname = f"cat_{i:04d}.png"
        cv2.imwrite(os.path.join(img_dir, fname), img)

        angles = np.linspace(0, 2 * np.pi, N_LANDMARKS, endpoint=False)
        radii = rng.normal(75, 5, size=N_LANDMARKS)
        c = np.array([IMG_SIZE / 2, IMG_SIZE / 2])
        pts = np.stack([c[0] + radii * np.cos(angles), c[1] + radii * np.sin(angles)], axis=1)
        pts += rng.normal(0, 2, pts.shape)

        records.append({"image": fname, "landmarks": pts.tolist()})

    with open(os.path.join(out_dir, "annotations.json"), "w") as f:
        json.dump(records, f, indent=2)

    print(f"Generated {n_images} dummy images + annotations at: {out_dir}")
    return records


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def align_face(img, landmarks, left_idx=0, right_idx=24):
    """Rotates the image so a chosen pair of reference landmarks (e.g. outer
    eye corners) is horizontal. Update left_idx/right_idx once you know
    CatFLW's actual landmark ordering/scheme."""
    p1, p2 = np.array(landmarks[left_idx]), np.array(landmarks[right_idx])
    angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
    center = tuple(np.mean(landmarks, axis=0))
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned_img = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))
    ones = np.ones((landmarks.shape[0], 1))
    aligned_landmarks = (rot_mat @ np.hstack([landmarks, ones]).T).T
    return aligned_img, aligned_landmarks


def apply_edge_filter(img):
    """Laplacian edge detection blended with the original image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F, ksize=3))
    edge_3ch = cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(img, 0.7, edge_3ch, 0.3, 0)


def random_augment(img, landmarks, rng):
    """Horizontal flip (identity flip-order placeholder -- update once you
    know CatFLW's left/right landmark correspondence) + brightness jitter."""
    FLIP_ORDER = list(range(N_LANDMARKS))
    if rng.random() < 0.5:
        img = cv2.flip(img, 1)
        landmarks = landmarks.copy()
        landmarks[:, 0] = img.shape[1] - landmarks[:, 0]
        landmarks = landmarks[FLIP_ORDER]
    brightness = rng.uniform(0.8, 1.2)
    img = np.clip(img.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    return img, landmarks


def make_heatmaps(landmarks, img_size=IMG_SIZE, heatmap_size=HEATMAP_SIZE, sigma=HEATMAP_SIGMA):
    """Renders one Gaussian heatmap per landmark, for the heatmap-regression
    approach. Landmarks are in original image pixel coordinates; this rescales
    them to the (smaller) heatmap resolution."""
    scale = heatmap_size / img_size
    heatmaps = np.zeros((len(landmarks), heatmap_size, heatmap_size), dtype=np.float32)
    xs = np.arange(heatmap_size)
    ys = np.arange(heatmap_size)
    grid_x, grid_y = np.meshgrid(xs, ys)

    for i, (x, y) in enumerate(landmarks):
        hx, hy = x * scale, y * scale
        heatmaps[i] = np.exp(-((grid_x - hx) ** 2 + (grid_y - hy) ** 2) / (2 * sigma ** 2))

    return heatmaps


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CatFLWDataset(Dataset):
    """
    Yields either:
      - "coord" mode: (image_tensor, landmarks_tensor [N*2])
      - "heatmap" mode: (image_tensor, heatmaps_tensor [N, H, W])
    depending on `target_type`, so the same dataset feeds both model approaches.
    """

    def __init__(self, image_dir, records, target_type="coord",
                 use_alignment=True, use_edge_filter=False, augment=False, seed=0):
        assert target_type in ("coord", "heatmap")
        self.image_dir = image_dir
        self.records = records
        self.target_type = target_type
        self.use_alignment = use_alignment
        self.use_edge_filter = use_edge_filter
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = cv2.imread(os.path.join(self.image_dir, rec["image"]))
        landmarks = np.array(rec["landmarks"], dtype=np.float32)

        if self.use_alignment:
            img, landmarks = align_face(img, landmarks)
        if self.augment:
            img, landmarks = random_augment(img, landmarks, self.rng)
        if self.use_edge_filter:
            img = apply_edge_filter(img)

        # resize to canonical size, adjusting landmark scale accordingly
        h0, w0 = img.shape[:2]
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        landmarks[:, 0] *= IMG_SIZE / w0
        landmarks[:, 1] *= IMG_SIZE / h0

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        if self.target_type == "coord":
            target = torch.from_numpy(landmarks.flatten()).float()
        else:
            heatmaps = make_heatmaps(landmarks)
            target = torch.from_numpy(heatmaps).float()

        return {"image": img_t, "target": target, "landmarks_raw": torch.from_numpy(landmarks).float()}


def train_val_test_split(records, train_frac=0.7, val_frac=0.15, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(records))
    n_train, n_val = int(len(records) * train_frac), int(len(records) * val_frac)
    to_recs = lambda ids: [records[i] for i in ids]
    return (
        to_recs(idx[:n_train]),
        to_recs(idx[n_train:n_train + n_val]),
        to_recs(idx[n_train + n_val:]),
    )
