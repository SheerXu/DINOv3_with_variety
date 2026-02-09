from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

from src.models.dinov3_backbone import Dinov3Backbone
from src.models.supervised_segmentation import SegmentationModel
from src.utils.config import load_config
from src.utils.visualization import visualize_segmentation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Supervised Segmentation Model")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--input", type=str, required=True, help="Input image or directory")
    parser.add_argument("--output", type=str, default="test_results/supervised", help="Output directory")
    parser.add_argument("--show", action="store_true", help="Show visualization")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    return parser.parse_args()


def load_model(cfg: Dict, checkpoint_path: str, device: torch.device):
    """加载有监督分割模型"""
    model_cfg: Dict = cfg["model"]
    data_cfg: Dict = cfg["data"]
    
    # 获取类别数
    label_map = data_cfg.get("label_map")
    if isinstance(label_map, list):
        num_classes = len(label_map)
    elif isinstance(label_map, dict):
        num_classes = max(label_map.values()) + 1
    else:
        raise ValueError("label_map not found in config")
    
    # 构建模型
    backbone = Dinov3Backbone(
        dinov3_repo_path=model_cfg["dinov3_repo_path"],
        model_name=model_cfg["model_name"],
        pretrained_path=None,  # 从checkpoint加载
        patch_size=model_cfg["patch_size"],
        embed_dim=model_cfg["embed_dim"],
    )
    model = SegmentationModel(backbone, num_classes=num_classes)
    
    # 加载权重
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.to(device)
    model.eval()
    
    return model, label_map


def preprocess_image(image_path: str | Path, resize: tuple | None = None):
    """预处理图像"""
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    
    if resize is not None:
        image_tensor = F.resize(image, resize, interpolation=Image.BILINEAR)
    else:
        image_tensor = image
    
    image_tensor = F.to_tensor(image_tensor).unsqueeze(0)
    
    return image_tensor, np.array(image), original_size


@torch.no_grad()
def predict(model, image_tensor: torch.Tensor, device: torch.device, original_size: tuple):
    """模型推理"""
    image_tensor = image_tensor.to(device)
    logits = model(image_tensor)
    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    
    # 调整到原始尺寸
    pred_image = Image.fromarray(pred.astype(np.uint8))
    pred_image = pred_image.resize(original_size, Image.NEAREST)
    pred = np.array(pred_image)
    
    return pred


def process_single_image(
    model,
    image_path: Path,
    output_dir: Path,
    resize: tuple | None,
    label_map,
    device: torch.device,
    show: bool,
):
    """处理单张图像"""
    print(f"Processing: {image_path.name}")
    
    # 预处理
    image_tensor, image_np, original_size = preprocess_image(image_path, resize)
    
    # 推理
    pred_mask = predict(model, image_tensor, device, original_size)
    
    # 可视化
    output_path = output_dir / f"{image_path.stem}_result.png"
    visualize_segmentation(
        image=image_np,
        pred_mask=pred_mask,
        gt_mask=None,
        label_map=label_map,
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
    model, label_map = load_model(cfg, args.checkpoint, device)
    
    # 准备输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理输入
    input_path = Path(args.input)
    if input_path.is_file():
        # 单张图片
        process_single_image(model, input_path, output_dir, resize, label_map, device, args.show)
    elif input_path.is_dir():
        # 目录
        image_exts = data_cfg.get("image_exts", [".png", ".jpg", ".jpeg"])
        image_files = []
        for ext in image_exts:
            image_files.extend(input_path.glob(f"*{ext}"))
        
        print(f"Found {len(image_files)} images")
        for image_file in sorted(image_files):
            process_single_image(model, image_file, output_dir, resize, label_map, device, args.show)
    else:
        raise ValueError(f"Invalid input path: {input_path}")
    
    print(f"\n✓ All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
