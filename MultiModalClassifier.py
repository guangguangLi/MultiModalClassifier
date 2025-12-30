# MultiModalClassifier.py
import os
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, Normalize

from seg import Seg



class HeatmapVisualizer:
    def __init__(self, heatmaps_dir="heatmaps"):
        self.heatmaps_dir = heatmaps_dir
        os.makedirs(self.heatmaps_dir, exist_ok=True)

    def visualize_layer(self, layer_output, layer_name, img_np):
        activation = layer_output.detach().cpu().numpy()  # [B,C,H,W]
        heatmap = activation.mean(axis=1).squeeze()
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(self.heatmaps_dir, f"{layer_name}.jpg"), heatmap)



class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)



class DualAttentionFusion(nn.Module):
    def __init__(self, channels, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.project = nn.Conv2d(channels * 2, channels, 1, bias=False)

    def channel_attention(self, x):
        var = x.var(dim=(2, 3), keepdim=True)
        weight = torch.sigmoid(var)
        return x * weight

    def spatial_attention(self, x):
        pooled = F.avg_pool2d(x, self.patch_size, self.patch_size)
        attn = torch.sigmoid(pooled)
        attn = F.interpolate(attn, size=x.shape[2:], mode="bilinear", align_corners=False)
        return x * attn

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.channel_attention(x) + self.spatial_attention(x)
        return self.project(x)



class MultiScaleEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, num_scales=5):
        super().__init__()
        self.branch_img = nn.ModuleList()
        self.branch_enh = nn.ModuleList()
        self.fusions = nn.ModuleList()

        for i in range(num_scales):
            c = base_channels * (2 ** i)
            in_c = in_channels if i == 0 else base_channels * (2 ** (i - 1))

            self.branch_img.append(nn.Sequential(
                ConvBlock(in_c, c),
                ConvBlock(c, c)
            ))
            self.branch_enh.append(nn.Sequential(
                ConvBlock(in_c, c),
                ConvBlock(c, c)
            ))
            self.fusions.append(DualAttentionFusion(c))

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = base_channels * (2 ** (num_scales - 1))
        self.visualizer = HeatmapVisualizer()

    def forward(self, img, enh):
        f_img, f_enh = img, enh
        fused_last = None

        for i in range(len(self.fusions)):
            f_img = self.branch_img[i](f_img)
            f_enh = self.branch_enh[i](f_enh)
            fused = self.fusions[i](f_img, f_enh)
            fused_last = fused

            if i != len(self.fusions) - 1:
                f_img = F.avg_pool2d(fused + f_img, 2)
                f_enh = F.avg_pool2d(fused + f_enh, 2)

        feat = self.global_pool(fused_last).flatten(1)
        return feat, fused_last


class MultiModalClassifier(nn.Module):
    def __init__(self, num_classes=2, base_channels=32, num_scales=5):
        super().__init__()

        # segmentation branch
        self.unext = UNeXt(num_classes=1, input_channels=3)

        # encoder
        self.encoder = MultiScaleEncoder(
            in_channels=3,
            base_channels=base_channels,
            num_scales=num_scales
        )

        self.proj_seg = nn.Linear(64, self.encoder.feature_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, img):
        # segmentation
        mask_logits, seg_feat = self.unext(img)
        if mask_logits.dim() == 3:
            mask_logits = mask_logits.unsqueeze(1)

        mask = torch.sigmoid(mask_logits)
        mask3 = mask.repeat(1, 3, 1, 1)

        enhanced = img * mask3

        # encoder
        feat, fused_map = self.encoder(img, enhanced)

        # fuse segmentation embedding
        feat = feat + self.proj_seg(seg_feat)

        out = self.classifier(feat)

        return out, mask_logits, {
            "feat": feat,
            "fused_map": fused_map
        }
