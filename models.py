"""
models.py

Two landmark detection approaches sharing the same pretrained CNN backbones,
to directly compare them (this is the "end-to-end CNN" ablation).

  1. CoordRegressionModel  - backbone -> FC head -> N*2 raw (x, y) values.
     Simple, fast, but can struggle with precise localization since it must
     learn exact pixel coordinates as a single regression target.

  2. HeatmapRegressionModel - backbone -> upsampling/deconv head -> N heatmaps.
     Predicts a spatial probability map per landmark instead of raw
     coordinates; the peak of each heatmap gives the predicted location.
     Generally more accurate for fine-grained localization (e.g. ears,
     whiskers) because it preserves spatial structure instead of collapsing
     the image straight to numbers.

Both use the same set of backbones (ShuffleNetV2, EfficientNetB0,
MobileNetV3), so the comparison grid is: {approach} x {backbone}.
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

N_LANDMARKS = 48
HEATMAP_SIZE = 112

AVAILABLE_BACKBONES = [
    "shufflenet_v2_x1_0",
    "efficientnet_b0",
    "mobilenetv3_small_100",
]


class TorchvisionFeatureBackbone(nn.Module):
    """Wraps a torchvision CNN to behave like a timm backbone: exposes
    `num_features` and returns a pooled feature vector by default, or the
    last spatial feature map when `spatial=True` (used by HeatmapRegressionModel)."""

    def __init__(self, name, pretrained=True, spatial=False):
        super().__init__()
        self.spatial = spatial
        if name == "shufflenet_v2_x1_0":
            weights = "IMAGENET1K_V1" if pretrained else None
            net = torchvision.models.shufflenet_v2_x1_0(weights=weights)
            # torchvision ShuffleNetV2 exposes conv1, maxpool, stage2-4, conv5
            self.features = nn.Sequential(
                net.conv1, net.maxpool, net.stage2, net.stage3, net.stage4, net.conv5
            )
            self.num_features = net.fc.in_features
            self.reduction = 32
        else:
            raise ValueError(f"Unsupported torchvision backbone: {name}")

    def forward(self, x):
        feat_map = self.features(x)  # (batch, C, H, W)
        if self.spatial:
            return feat_map
        pooled = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)
        return pooled


def make_backbone(backbone_name, pretrained=True, spatial=False):
    """Returns (backbone_module, num_features). `spatial=True` requests a
    spatial feature map (for heatmap regression) instead of a pooled vector."""
    if backbone_name == "shufflenet_v2_x1_0":
        backbone = TorchvisionFeatureBackbone(backbone_name, pretrained, spatial=spatial)
        return backbone, backbone.num_features
    else:
        if spatial:
            backbone = timm.create_model(
                backbone_name, pretrained=pretrained, features_only=True, out_indices=(-1,)
            )
            num_features = backbone.feature_info[-1]["num_chs"]
        else:
            backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
            with torch.no_grad():
                dummy_out = backbone(torch.zeros(1, 3, 224, 224))
            num_features = dummy_out.shape[-1]
        return backbone, num_features


# ---------------------------------------------------------------------------
# Approach 1: direct coordinate regression
# ---------------------------------------------------------------------------

class CoordRegressionModel(nn.Module):
    def __init__(self, backbone_name="shufflenet_v2_x1_0", pretrained=True):
        super().__init__()
        self.backbone, feat_dim = make_backbone(backbone_name, pretrained, spatial=False)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, N_LANDMARKS * 2),
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)  # (batch, N_LANDMARKS*2)


# ---------------------------------------------------------------------------
# Approach 2: heatmap regression
# ---------------------------------------------------------------------------

class HeatmapRegressionModel(nn.Module):
    """
    Uses `timm`'s `features_only` mode to get a spatial feature map (rather
    than a pooled vector), then upsamples with transposed convolutions to
    produce one heatmap per landmark.
    """

    def __init__(self, backbone_name="shufflenet_v2_x1_0", pretrained=True):
        super().__init__()
        self.backbone, in_channels = make_backbone(backbone_name, pretrained, spatial=True)
        self._is_torchvision = backbone_name == "shufflenet_v2_x1_0"

        # Upsample back toward HEATMAP_SIZE (typically a 4x downsample from input)
        # Number of deconv stages depends on how far the backbone's final
        # feature map is from HEATMAP_SIZE; this default assumes a few stages.
        layers = []
        channels = in_channels
        n_upsample_stages = 3  # tune based on actual backbone's reduction factor
        for _ in range(n_upsample_stages):
            out_channels = max(channels // 2, 32)
            layers += [
                nn.ConvTranspose2d(channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            channels = out_channels
        self.upsample = nn.Sequential(*layers)
        self.final_conv = nn.Conv2d(channels, N_LANDMARKS, kernel_size=1)

    def forward(self, x):
        out = self.backbone(x)
        feats = out[-1] if isinstance(out, (list, tuple)) else out
        up = self.upsample(feats)
        heatmaps = self.final_conv(up)
        # Resize precisely to HEATMAP_SIZE in case upsampling stages don't land exactly
        heatmaps = F.interpolate(heatmaps, size=(HEATMAP_SIZE, HEATMAP_SIZE), mode="bilinear", align_corners=False)
        return heatmaps  # (batch, N_LANDMARKS, H, W)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def wing_loss(pred, target, w=10.0, epsilon=2.0):
    """Better suited than MSE for coordinate regression -- weights small
    errors more heavily, helping precision on subtle landmark shifts."""
    diff = torch.abs(pred - target)
    C = w - w * torch.log(torch.tensor(1.0 + w / epsilon))
    return torch.where(diff < w, w * torch.log(1.0 + diff / epsilon), diff - C).mean()


def heatmap_mse_loss(pred, target):
    return F.mse_loss(pred, target)


# ---------------------------------------------------------------------------
# Extracting coordinates from heatmaps (soft-argmax, differentiable and
# sub-pixel accurate, preferred over plain argmax)
# ---------------------------------------------------------------------------

def soft_argmax(heatmaps, temperature=10.0):
    b, n, h, w = heatmaps.shape
    flat = heatmaps.view(b, n, -1)
    probs = F.softmax(flat * temperature, dim=-1).view(b, n, h, w)
    xs = torch.arange(w, device=heatmaps.device).float()
    ys = torch.arange(h, device=heatmaps.device).float()
    exp_x = (probs.sum(dim=2) * xs).sum(dim=-1)
    exp_y = (probs.sum(dim=3) * ys).sum(dim=-1)
    return torch.stack([exp_x, exp_y], dim=-1)  # (batch, N_LANDMARKS, 2)


def build_model(approach, backbone_name, pretrained=True):
    assert approach in ("coord", "heatmap")
    if approach == "coord":
        return CoordRegressionModel(backbone_name, pretrained)
    return HeatmapRegressionModel(backbone_name, pretrained)
