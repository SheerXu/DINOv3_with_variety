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
from PIL import Image
from torchvision.transforms import functional as F

from src.models.anomaly_detector import AnomalyDetector
from src.models.dinov3_backbone import Dinov3Backbone
from src.utils.config import load_config
from src.utils.visualization import visualize_anomaly


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test AD-DINOv3 Model")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--input", type=str, required=True, help="Input image or directory")
    parser.add_argument("--output", type=str, default="test_results/ad_dinov3", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Anomaly threshold")
    parser.add_argument("--show", action="store_true", help="Show visualization")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    return parser.parse_args()


def load_model(cfg: Dict, checkpoint_path: str, device: torch.device):
    """加载AD-DINOv3模型"""
    model_cfg: Dict = cfg["model"]
    
    # 构建Teacher骨干网络
    teacher_backbone = Dinov3Backbone(
        dinov3_repo_path=model_cfg["dinov3_repo_path"],
        model_name=model_cfg["model_name"],
        pretrained_path=model_cfg.get("pretrained_path"),
        patch_size=model_cfg["patch_size"],
        embed_dim=model_cfg["embed_dim"],
        download=model_cfg.get("download", False)
    )
    
    # 构建异常检测器
    detector = AnomalyDetector(
        teacher_backbone=teacher_backbone,
        embed_dim=model_cfg["embed_dim"],
        memory_bank_size=model_cfg.get("memory_bank_size", 2000),
        use_multi_scale=model_cfg.get("use_multi_scale", True),
    )
    
    # 加载权重
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    detector.load_state_dict(ckpt.get("model", ckpt), strict=False)
    detector.to(device)
    detector.eval()
    
    return detector


def preprocess_image(image_path: str | Path, resize: tuple | None = None):
    """预处理图像"""
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    
    if resize is not None:
        image_resized = F.resize(image, resize, interpolation=Image.BILINEAR)
    else:
        image_resized = image
    
    image_tensor = F.to_tensor(image_resized).unsqueeze(0)
    
    return image_tensor, np.array(image), original_size


@torch.no_grad()
def predict(detector, image_tensor: torch.Tensor, device: torch.device, original_size: tuple):
    """模型推理"""
    image_tensor = image_tensor.to(device)
    
    # 获取异常分数图
    anomaly_map = detector(image_tensor)
    anomaly_map = torch.sigmoid(anomaly_map).squeeze(0).squeeze(0).cpu().numpy()
    
    # 调整到原始尺寸
    anomaly_image = Image.fromarray((anomaly_map * 255).astype(np.uint8))
    anomaly_image = anomaly_image.resize(original_size, Image.BILINEAR)
    anomaly_map = np.array(anomaly_image).astype(np.float32) / 255.0
    
    return anomaly_map


def process_single_image(
    detector,
    image_path: Path,
    output_dir: Path,
    resize: tuple | None,
    threshold: float,
    device: torch.device,
    show: bool,
    relative_path: Path | None = None,
):
    """处理单张图像"""
    print(f"Processing: {image_path}")
    
    # 预处理
    image_tensor, image_np, original_size = preprocess_image(image_path, resize)
    
    # 推理
    anomaly_map = predict(detector, image_tensor, device, original_size)
    
    # 计算统计信息
    max_score = anomaly_map.max()
    mean_score = anomaly_map.mean()
    anomaly_pixels = (anomaly_map > threshold).sum()
    total_pixels = anomaly_map.size
    anomaly_ratio = anomaly_pixels / total_pixels * 100
    
    print(f"  Max Score: {max_score:.3f}, Mean Score: {mean_score:.3f}")
    print(f"  Anomaly Ratio: {anomaly_ratio:.2f}% (threshold={threshold})")
    
    # 确定输出路径（保持目录结构）
    if relative_path:
        output_subdir = output_dir / relative_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)
        output_path = output_subdir / f"{image_path.stem}_result.png"
    else:
        output_path = output_dir / f"{image_path.stem}_result.png"
    
    # 可视化
    visualize_anomaly(
        image=image_np,
        anomaly_map=anomaly_map,
        threshold=threshold,
        gt_mask=None,
        save_path=output_path,
        show=show,
    )


def main():
    args = parse_args()
    
    # 加载配置
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    resize = tuple(data_cfg["resize"]) if data_cfg.get("resize") else None
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载模型
    print(f"Loading model from: {args.checkpoint}")
    detector = load_model(cfg, args.checkpoint, device)
    
    # 准备输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理输入
    input_path = Path(args.input)
    if input_path.is_file():
        # 单张图片
        process_single_image(detector, input_path, output_dir, resize, args.threshold, device, args.show)
    elif input_path.is_dir():
        # 目录（递归查找）
        image_exts = data_cfg.get("image_exts", [".png", ".jpg", ".jpeg"])
        image_files = []
        for ext in image_exts:
            image_files.extend(input_path.rglob(f"*{ext}"))  # 使用 rglob 递归查找
        
        print(f"Found {len(image_files)} images in {input_path} (including subdirectories)")
        for image_file in sorted(image_files):
            # 计算相对路径以保持目录结构
            relative_path = image_file.relative_to(input_path)
            process_single_image(detector, image_file, output_dir, resize, args.threshold, device, args.show, relative_path)
    else:
        raise ValueError(f"Invalid input path: {input_path}")
    
    print(f"\n✓ All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
