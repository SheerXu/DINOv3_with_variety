from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F


def visualize_segmentation(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray | None = None,
    label_map: dict | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    可视化分割结果
    
    Args:
        image: 原始图像 (H, W, 3)
        pred_mask: 预测mask (H, W)，值为类别索引
        gt_mask: 真实mask (H, W)，可选
        label_map: 类别映射 {类别名: 索引}
        save_path: 保存路径
        show: 是否显示
    """
    num_plots = 3 if gt_mask is not None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    
    if num_plots == 2:
        axes = [axes[0], axes[1]]
    
    # 原图
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")
    
    # 预测结果
    pred_colored = colorize_mask(pred_mask, label_map)
    axes[1].imshow(image)
    axes[1].imshow(pred_colored, alpha=0.5)
    axes[1].set_title("Prediction", fontsize=14)
    axes[1].axis("off")
    
    # 真实标注（可选）
    if gt_mask is not None:
        gt_colored = colorize_mask(gt_mask, label_map)
        axes[2].imshow(image)
        axes[2].imshow(gt_colored, alpha=0.5)
        axes[2].set_title("Ground Truth", fontsize=14)
        axes[2].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_anomaly(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    threshold: float = 0.5,
    gt_mask: np.ndarray | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    可视化异常检测结果
    
    Args:
        image: 原始图像 (H, W, 3)
        anomaly_map: 异常分数图 (H, W)，范围 [0, 1]
        threshold: 异常阈值
        gt_mask: 真实异常mask (H, W)，可选
        save_path: 保存路径
        show: 是否显示
    """
    num_plots = 4 if gt_mask is not None else 3
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    
    # 原图
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")
    
    # 异常分数热图
    im = axes[1].imshow(anomaly_map, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title(f"Anomaly Score Map", fontsize=14)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 二值化异常区域
    binary_mask = (anomaly_map > threshold).astype(np.uint8) * 255
    axes[2].imshow(image)
    axes[2].imshow(binary_mask, cmap="Reds", alpha=0.5)
    axes[2].set_title(f"Detection (threshold={threshold:.2f})", fontsize=14)
    axes[2].axis("off")
    
    # 真实标注（可选）
    if gt_mask is not None:
        axes[3].imshow(image)
        axes[3].imshow(gt_mask * 255, cmap="Reds", alpha=0.5)
        axes[3].set_title("Ground Truth", fontsize=14)
        axes[3].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def colorize_mask(mask: np.ndarray, label_map: dict | None = None) -> np.ndarray:
    """
    为mask添加颜色
    
    Args:
        mask: (H, W) 类别索引
        label_map: {类别名: 索引}
    
    Returns:
        colored_mask: (H, W, 4) RGBA图像
    """
    num_classes = int(mask.max()) + 1
    
    # 生成颜色表
    colors = generate_colors(num_classes)
    
    h, w = mask.shape
    colored = np.zeros((h, w, 4), dtype=np.uint8)
    
    for cls_id in range(num_classes):
        if cls_id == 0:  # 背景透明
            colored[mask == cls_id] = [0, 0, 0, 0]
        else:
            colored[mask == cls_id] = colors[cls_id]
    
    return colored


def generate_colors(num_classes: int) -> np.ndarray:
    """生成区分度高的颜色表"""
    colors = np.zeros((num_classes, 4), dtype=np.uint8)
    colors[:, 3] = 180  # alpha通道
    
    # 预定义常用颜色
    predefined = [
        [0, 0, 0],       # 背景（透明）
        [255, 0, 0],     # 红
        [0, 255, 0],     # 绿
        [0, 0, 255],     # 蓝
        [255, 255, 0],   # 黄
        [255, 0, 255],   # 品红
        [0, 255, 255],   # 青
        [255, 128, 0],   # 橙
        [128, 0, 255],   # 紫
    ]
    
    for i in range(min(num_classes, len(predefined))):
        colors[i, :3] = predefined[i]
    
    # 超出预定义范围则随机生成
    for i in range(len(predefined), num_classes):
        np.random.seed(i * 42)
        colors[i, :3] = np.random.randint(0, 255, 3)
    
    return colors


def save_side_by_side(
    images: list[np.ndarray],
    titles: list[str],
    save_path: str | Path,
    dpi: int = 150,
) -> None:
    """
    并排保存多个图像
    
    Args:
        images: 图像列表
        titles: 标题列表
        save_path: 保存路径
        dpi: 分辨率
    """
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    
    if num_images == 1:
        axes = [axes]
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
