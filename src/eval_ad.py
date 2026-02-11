from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from src.datasets.anomaly_dataset import AnomalyDetectionDataset
from src.models.anomaly_detector import AnomalyDetector
from src.models.dinov3_backbone import Dinov3Backbone
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    return parser.parse_args()


def compute_metrics(anomaly_scores: np.ndarray, labels: np.ndarray) -> Dict:
    """计算异常检测指标
    
    Args:
        anomaly_scores: 异常分数 (N,)
        labels: 真实标签 (N,) 0=正常, 1=异常
        
    Returns:
        metrics: 包含 AUROC, AUPRC, 最优 F1 等
    """
    # Image-level AUROC
    auroc = roc_auc_score(labels, anomaly_scores)
    
    # Precision-Recall
    precision, recall, thresholds = precision_recall_curve(labels, anomaly_scores)
    auprc = auc(recall, precision)
    
    # 计算最优 F1
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else thresholds[-1]
    
    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "Best_F1": best_f1,
        "Best_Threshold": best_threshold,
        "Precision@BestF1": precision[best_f1_idx],
        "Recall@BestF1": recall[best_f1_idx],
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_cfg: Dict = cfg["data"]
    model_cfg: Dict = cfg["model"]

    # 测试数据集
    test_ds = AnomalyDetectionDataset(
        root_dir=data_cfg["val_dir"],
        image_exts=data_cfg["image_exts"],
        mode="test",
        resize=tuple(data_cfg["resize"]) if data_cfg.get("resize") else None,
        use_anomaly_generation=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 2),
    )

    # 加载模型
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

    # 加载检查点
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    detector.load_state_dict(ckpt["detector"], strict=False)
    if "memory_bank" in ckpt:
        detector.memory_bank.load_state_dict(ckpt["memory_bank"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector.to(device)
    detector.eval()

    print("Evaluating AD-DINOv3...")

    all_scores = []
    all_labels = []
    pixel_scores = []
    pixel_labels = []

    with torch.no_grad():
        for images, masks, labels in test_loader:
            images = images.to(device)
            masks = masks.numpy()
            labels = labels.numpy()

            # 异常分数图
            anomaly_map = detector(images)
            anomaly_map = anomaly_map.cpu().numpy()

            # Image-level score: 取最大值
            img_score = anomaly_map.max()
            all_scores.append(img_score)
            all_labels.append(labels[0])

            # Pixel-level
            pixel_scores.append(anomaly_map.flatten())
            pixel_labels.append(masks.flatten())

    # Image-level metrics
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    img_metrics = compute_metrics(all_scores, all_labels)

    print("\n=== AD-DINOv3 Image-Level Metrics ===")
    for key, value in img_metrics.items():
        print(f"{key}: {value:.4f}")

    # Pixel-level metrics
    pixel_scores = np.concatenate(pixel_scores)
    pixel_labels = np.concatenate(pixel_labels)
    pixel_metrics = compute_metrics(pixel_scores, pixel_labels)

    print("\n=== AD-DINOv3 Pixel-Level Metrics ===")
    for key, value in pixel_metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
