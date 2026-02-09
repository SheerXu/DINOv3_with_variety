from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.anomaly_dataset import AnomalyDetectionDataset
from src.datasets.supervised_dataset import PolygonJsonDataset, pad_collate
from src.models.anomaly_detector import AnomalyDetector
from src.models.dinov3_backbone import Dinov3Backbone
from src.models.supervised_segmentation import SegmentationModel
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Supervised vs AD-DINOv3")
    parser.add_argument("--config_supervised", type=str, required=True, help="Supervised model config")
    parser.add_argument("--config_ad", type=str, required=True, help="AD-DINOv3 config")
    parser.add_argument("--checkpoint_supervised", type=str, required=True)
    parser.add_argument("--checkpoint_ad", type=str, required=True)
    parser.add_argument("--output", type=str, default="comparison_results.json")
    return parser.parse_args()


def evaluate_supervised(cfg: Dict, checkpoint_path: str, device: torch.device) -> Dict:
    """评估有监督分割模型"""
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

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
        batch_size=1,
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

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.to(device)
    model.eval()

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
                for cls in range(1, num_classes):
                    pred_c = p == cls
                    target_c = t == cls
                    target_c = np.logical_and(target_c, t != ignore_index)
                    intersection = np.logical_and(pred_c, target_c).sum()
                    union = np.logical_or(pred_c, target_c).sum()
                    if union > 0:
                        iou_sum[cls] += intersection / union
                count += 1

    miou = (iou_sum[1:] / max(count, 1)).mean() if num_classes > 1 else 0.0

    return {
        "method": "Supervised Segmentation",
        "mIoU": float(miou),
        "class_iou": {f"class_{i}": float(iou_sum[i] / max(count, 1)) for i in range(1, num_classes)},
    }


def evaluate_ad_dinov3(cfg: Dict, checkpoint_path: str, device: torch.device) -> Dict:
    """评估 AD-DINOv3 异常检测模型"""
    from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    test_ds = AnomalyDetectionDataset(
        root_dir=data_cfg["val_dir"],
        image_exts=data_cfg["image_exts"],
        mode="test",
        resize=tuple(data_cfg["resize"]) if data_cfg.get("resize") else None,
        use_anomaly_generation=False,
    )

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=data_cfg.get("num_workers", 2))

    backbone = Dinov3Backbone(
        dinov3_repo_path=model_cfg["dinov3_repo_path"],
        model_name=model_cfg["model_name"],
        pretrained_path=model_cfg.get("pretrained_path"),
        patch_size=model_cfg["patch_size"],
        embed_dim=model_cfg["embed_dim"],
    )

    detector = AnomalyDetector(
        backbone=backbone,
        memory_bank_size=model_cfg.get("memory_bank_size", 1000),
        use_multi_scale=model_cfg.get("use_multi_scale", True),
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    detector.load_state_dict(ckpt["detector"], strict=False)
    if "memory_bank" in ckpt:
        detector.memory_bank.load_state_dict(ckpt["memory_bank"])

    detector.to(device)
    detector.eval()

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for images, masks, labels in test_loader:
            images = images.to(device)
            labels = labels.numpy()
            anomaly_map = detector(images)
            img_score = anomaly_map.cpu().numpy().max()
            all_scores.append(img_score)
            all_labels.append(labels[0])

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    auroc = roc_auc_score(all_labels, all_scores)
    precision, recall, _ = precision_recall_curve(all_labels, all_scores)
    auprc = auc(recall, precision)

    return {
        "method": "AD-DINOv3",
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
    }


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # 评估有监督模型
    print("=" * 50)
    print("Evaluating Supervised Segmentation Model...")
    print("=" * 50)
    cfg_supervised = load_config(args.config_supervised)
    results_supervised = evaluate_supervised(cfg_supervised, args.checkpoint_supervised, device)
    print(f"Supervised mIoU: {results_supervised['mIoU']:.4f}")

    # 评估 AD-DINOv3
    print("\n" + "=" * 50)
    print("Evaluating AD-DINOv3 Model...")
    print("=" * 50)
    cfg_ad = load_config(args.config_ad)
    results_ad = evaluate_ad_dinov3(cfg_ad, args.checkpoint_ad, device)
    print(f"AD-DINOv3 AUROC: {results_ad['AUROC']:.4f}")
    print(f"AD-DINOv3 AUPRC: {results_ad['AUPRC']:.4f}")

    # 保存结果
    comparison = {
        "supervised": results_supervised,
        "ad_dinov3": results_ad,
        "summary": {
            "supervised_best_metric": f"mIoU={results_supervised['mIoU']:.4f}",
            "ad_best_metric": f"AUROC={results_ad['AUROC']:.4f}",
            "conclusion": (
                "Supervised is better for known defect classes with sufficient labels. "
                "AD-DINOv3 is better for unknown defects with limited labels."
            ),
        },
    }

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print(f"Comparison results saved to: {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
