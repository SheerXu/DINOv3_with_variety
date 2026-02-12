from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision.transforms import functional as F


class AnomalyGenerator:
    """异常样本生成器：用于无监督异常检测的数据增强"""

    def __init__(
        self,
        cutpaste_ratio: float = 0.5,
        perlin_ratio: float = 0.3,
        blend_alpha: Tuple[float, float] = (0.3, 0.7),
    ) -> None:
        self.cutpaste_ratio = cutpaste_ratio
        self.perlin_ratio = perlin_ratio
        self.blend_alpha = blend_alpha

    def __call__(self, image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """生成异常样本及其对应的二值掩码
        
        Args:
            image: 原始正常图像
            
        Returns:
            aug_image: 合成的异常图像
            mask: 异常区域掩码 (0=正常, 1=异常)
        """
        w, h = image.size
        mask = Image.new("L", (w, h), 0)
        
        # 该函数语义是“生成异常样本”。因此不应返回正常图像/空掩码，
        # 否则会出现 label=1 但 mask 全 0 的训练噪声。
        method = random.random()
        if method < self.cutpaste_ratio:
            aug_image, mask = self._cutpaste(image)
        else:
            aug_image, mask = self._perlin_noise(image)
        
        return aug_image, mask

    def _cutpaste(self, image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """CutPaste: 随机裁剪并粘贴到图像其他位置"""
        w, h = image.size
        
        # 随机裁剪区域
        cut_w = random.randint(int(w * 0.05), int(w * 0.3))
        cut_h = random.randint(int(h * 0.05), int(h * 0.3))
        cut_x = random.randint(0, w - cut_w)
        cut_y = random.randint(0, h - cut_h)
        
        # 裁剪并旋转
        patch = image.crop((cut_x, cut_y, cut_x + cut_w, cut_y + cut_h))
        patch = patch.rotate(random.randint(-45, 45), expand=False)
        
        # 粘贴到随机位置
        paste_x = random.randint(0, w - cut_w)
        paste_y = random.randint(0, h - cut_h)
        
        aug_image = image.copy()
        aug_image.paste(patch, (paste_x, paste_y))
        
        # 生成掩码
        mask = Image.new("L", (w, h), 0)
        mask_patch = Image.new("L", (cut_w, cut_h), 255)
        mask.paste(mask_patch, (paste_x, paste_y))
        
        return aug_image, mask

    def _perlin_noise(self, image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Perlin 噪声生成纹理异常"""
        w, h = image.size
        
        # 简化版：使用高斯噪声模拟（生成合法的 uint8 图像，避免负值 cast 回绕）
        noise = np.random.randn(h, w, 3) * 50 + 127.5
        noise = np.clip(noise, 0, 255).astype(np.uint8)
        noise_img = Image.fromarray(noise, mode="RGB")
        
        # 随机掩码区域
        mask_arr = np.zeros((h, w), dtype=np.uint8)
        num_blobs = random.randint(1, 3)
        for _ in range(num_blobs):
            cx = random.randint(0, w)
            cy = random.randint(0, h)
            radius = random.randint(int(min(w, h) * 0.1), int(min(w, h) * 0.3))
            y, x = np.ogrid[:h, :w]
            mask_circle = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
            mask_arr[mask_circle] = 255
        
        mask = Image.fromarray(mask_arr, mode="L")
        mask_blur = mask.filter(ImageFilter.GaussianBlur(radius=5))
        
        # 混合（mask=255 位置使用 noise_img，否则使用原图）
        aug_image = Image.composite(noise_img, image, mask_blur)
        
        return aug_image, mask

    def generate_normal_only(self, image: Image.Image) -> Image.Image:
        """仅应用正常增强（用于训练正常样本）"""
        # 轻微几何变换
        if random.random() < 0.5:
            image = F.hflip(image)
        if random.random() < 0.3:
            image = image.rotate(random.uniform(-10, 10), expand=False)
        return image
