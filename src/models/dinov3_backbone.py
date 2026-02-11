from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn


class Dinov3Backbone(nn.Module):
    def __init__(
        self,
        dinov3_repo_path: str | Path,
        model_name: str,
        pretrained_path: str | Path | None,
        patch_size: int,
        embed_dim: int,
        download: bool = False
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        repo_path = Path(dinov3_repo_path)
        if not repo_path.exists():
            raise FileNotFoundError(f"DINOv3 repo not found: {repo_path}")

        # 加载本地DINOv3模型结构
        if download:
            print(f"Loading model '{model_name}' with automatic weight download...")
        else:
            print(f"Loading model '{model_name}' structure (no automatic download)...")
        
        self.model = torch.hub.load(
            str(repo_path), 
            model_name, 
            source="local",
            pretrained=download  # 根据配置决定是否自动下载权重
        )

        # 手动加载预训练权重（当不自动下载时）
        if not download and pretrained_path:
            pretrained_path = Path(pretrained_path)
            if not pretrained_path.exists():
                raise FileNotFoundError(f"Pretrained weights not found: {pretrained_path}")
            
            print(f"Loading pretrained weights from: {pretrained_path}")
            ckpt = torch.load(pretrained_path, map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            missing, unexpected = self.model.load_state_dict(ckpt, strict=False)
            print(f"Loaded pretrained weights. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        elif not download and not pretrained_path:
            print("Warning: No pretrained weights specified, using random initialization")
        elif download:
            print(f"Using automatically downloaded pretrained weights.")

    def _tokens_to_feature(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, N, C), remove cls token if present
        if tokens.dim() != 3:
            raise ValueError("Expected tokens with shape (B, N, C)")
        if tokens.shape[1] % 2 == 1:
            tokens = tokens[:, 1:, :]   # 去除第一个token（CLS标记）
        b, n, c = tokens.shape
        h = w = int(n ** 0.5)   # 重建空间维度
        if h * w != n:
            raise ValueError("Token count is not a perfect square; please check patch size.")
        # 转换为特征图 (B, C, H, W)
        feat = tokens.transpose(1, 2).contiguous().view(b, c, h, w)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(x)
        else:
            feats = self.model(x)

        if isinstance(feats, dict):
            for key in ["x", "last_hidden_state", "tokens", "features"]:
                if key in feats:
                    feats = feats[key]
                    break

        if feats.dim() == 3:
            return self._tokens_to_feature(feats)
        if feats.dim() == 4:
            return feats
        raise ValueError("Unsupported feature shape from backbone.")
