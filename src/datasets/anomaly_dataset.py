from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from .anomaly_generator import AnomalyGenerator


class AnomalyDetectionDataset(Dataset):
    """AD-DINOv3 数据集：支持正常样本训练和异常样本测试"""

    def __init__(
        self,
        root_dir: str | Path,
        image_exts: List[str],
        mode: str = "train",
        resize: Tuple[int, int] | None = None,
        use_anomaly_generation: bool = True,
        anomaly_ratio: float = 0.5,
    ) -> None:
        """
        Args:
            root_dir: 数据根目录
            image_exts: 图像扩展名
            mode: "train" (仅正常样本) 或 "test" (含异常样本)
            resize: 目标分辨率
            use_anomaly_generation: 训练时是否生成合成异常
            anomaly_ratio: 合成异常样本的比例
        """
        self.root_dir = Path(root_dir)
        self.image_exts = [e.lower() for e in image_exts]
        self.mode = mode
        self.resize = resize
        self.use_anomaly_generation = use_anomaly_generation and mode == "train"
        self.anomaly_ratio = anomaly_ratio

        self.image_paths = self._collect_images()
        
        if self.use_anomaly_generation:
            self.anomaly_gen = AnomalyGenerator()

    def _collect_images(self) -> List[Path]:
        """收集所有图像路径"""
        images = []
        for ext in self.image_exts:
            images.extend(self.root_dir.rglob(f"*{ext}"))
        return sorted(images)

    def _has_defect(self, image_path: Path) -> bool:
        """检查是否有对应的标注文件（判断异常/正常）"""
        json_path = image_path.with_suffix(".json")
        if not json_path.exists():
            return False
        
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return len(data.get("shapes", [])) > 0
        except:
            return False

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        
        # 判断是否为异常样本
        is_anomaly = self._has_defect(image_path)
        
        # 训练模式：合成异常
        if self.mode == "train" and self.use_anomaly_generation:
            if torch.rand(1).item() < self.anomaly_ratio:
                image, mask = self.anomaly_gen(image)
                is_anomaly_synthetic = 1
            else:
                # 正常样本
                image = self.anomaly_gen.generate_normal_only(image)
                mask = Image.new("L", image.size, 0)
                is_anomaly_synthetic = 0
        else:
            # 测试模式或不生成异常
            mask = Image.new("L", image.size, 0)
            is_anomaly_synthetic = int(is_anomaly)

        # 调整大小
        if self.resize is not None:
            image = F.resize(image, self.resize, interpolation=Image.BILINEAR)
            mask = F.resize(mask, self.resize, interpolation=Image.NEAREST)

        image_tensor = F.to_tensor(image)
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.float32)) / 255.0
        label = torch.tensor(is_anomaly_synthetic, dtype=torch.long)

        return image_tensor, mask_tensor, label
