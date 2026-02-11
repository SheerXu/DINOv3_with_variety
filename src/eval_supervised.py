from __future__ import annotations

import argparse
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets.supervised_dataset import PolygonJsonDataset, pad_collate
from .models.dinov3_backbone import Dinov3Backbone
from .models.supervised_segmentation import SegmentationModel
from .utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def compute_iou(pred: np.ndarray, target: np.ndarray, num_classes: int, ignore_index: int) -> np.ndarray:
    iou = np.zeros(num_classes, dtype=np.float64)
    for cls in range(1, num_classes):
        pred_c = pred == cls
        target_c = target == cls
        if ignore_index is not None:
            target_c = np.logical_and(target_c, target != ignore_index)
        intersection = np.logical_and(pred_c, target_c).sum()
        union = np.logical_or(pred_c, target_c).sum()
        if union > 0:
            iou[cls] = intersection / union
    return iou


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_cfg: Dict = cfg["data"]
    model_cfg: Dict = cfg["model"]

    val_ds = PolygonJsonDataset(
        root_dir=data_cfg["val_dir"],
        image_exts=data_cfg["image_exts"],
        label_map=data_cfg.get("label_map"),
        resize=tuple(data_cfg["resize"]) if data_cfg.get("resize") else None,
        random_flip=False,
        ignore_index=data_cfg.get("ignore_index", 255),
    )

    num_classes = (max(val_ds.label_map.values()) + 1) if val_ds.label_map else 1

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.get("eval", {}).get("batch_size", 1),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 2),
        collate_fn=lambda b: pad_collate(b, data_cfg.get("ignore_index", 255)),
    )

    backbone = Dinov3Backbone(
        dinov3_repo_path=model_cfg["dinov3_repo_path"],
        model_name=model_cfg["model_name"],
        pretrained_path=model_cfg.get("pretrained_path"),
        patch_size=model_cfg["patch_size"],
        embed_dim=model_cfg["embed_dim"],
    )
    model = SegmentationModel(backbone, num_classes=num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    ignore_index = data_cfg.get("ignore_index", 255)
    iou_sum = np.zeros(num_classes, dtype=np.float64)
    count = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            for p, t in zip(preds.cpu().numpy(), masks.cpu().numpy()):
                iou = compute_iou(p, t, num_classes, ignore_index)
                iou_sum += iou
                count += 1

    miou = (iou_sum[1:] / max(count, 1)).mean() if num_classes > 1 else 0.0
    print(f"mIoU: {miou:.4f}")


if __name__ == "__main__":
    main()
