from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryBank(nn.Module):
    """特征记忆库：存储正常样本的特征原型用于异常检测"""

    def __init__(self, feature_dim: int, bank_size: int = 1000) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.bank_size = bank_size
        self.register_buffer("features", torch.zeros(bank_size, feature_dim))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("is_full", torch.zeros(1, dtype=torch.bool))

    @torch.no_grad()
    def update(self, features: torch.Tensor) -> None:
        """更新记忆库，存储正常样本特征
        
        Args:
            features: (B, C, H, W) 或 (N, C)
        """
        if features.dim() == 4:
            # 空间特征展平
            b, c, h, w = features.shape
            features = features.permute(0, 2, 3, 1).reshape(-1, c)
        
        features = F.normalize(features, dim=1)
        batch_size = features.shape[0]

        ptr = int(self.ptr)
        if ptr + batch_size <= self.bank_size:
            self.features[ptr : ptr + batch_size] = features
            ptr = (ptr + batch_size) % self.bank_size
        else:
            # 循环覆盖
            remain = self.bank_size - ptr
            self.features[ptr:] = features[:remain]
            self.features[: batch_size - remain] = features[remain:]
            ptr = batch_size - remain
            self.is_full[0] = True

        self.ptr[0] = ptr

    def get_nearest_distance(self, query_features: torch.Tensor, k: int = 1) -> torch.Tensor:
        """计算查询特征到记忆库的最近距离（异常分数）
        
        Args:
            query_features: (B, C, H, W) 或 (N, C)
            k: 取 top-k 近邻的平均距离
            
        Returns:
            distances: 与输入同shape的异常分数图
        """
        original_shape = query_features.shape
        if query_features.dim() == 4:
            b, c, h, w = query_features.shape
            query_features = query_features.permute(0, 2, 3, 1).reshape(-1, c)
        
        query_features = F.normalize(query_features, dim=1)
        
        # 检查记忆库是否为空
        bank_size = int(self.ptr) if not self.is_full else self.bank_size
        if bank_size == 0:
            # 记忆库为空，返回中性异常分数（0.5）
            anomaly_scores = torch.full(
                (query_features.shape[0],), 
                0.5, 
                dtype=query_features.dtype, 
                device=query_features.device
            )
        else:
            # 相似度矩阵: (N, bank_size)
            similarity = torch.mm(query_features, self.features[:bank_size].t())
            
            # 距离 = (1 - 余弦相似度) / 2，确保范围在 [0, 1]
            # 余弦相似度范围 [-1, 1]，所以 (1 - similarity) 范围 [0, 2]
            # 除以 2 后范围变为 [0, 1]
            distances = (1 - similarity) / 2.0
            
            # 取 top-k 最近邻的平均距离
            k_actual = min(k, bank_size)
            topk_distances, _ = torch.topk(distances, k=k_actual, dim=1, largest=False)
            anomaly_scores = topk_distances.mean(dim=1)
            
            # 确保值在 [0, 1] 范围内，防止数值误差
            anomaly_scores = anomaly_scores.clamp(0.0, 1.0)
        
        # 恢复原始形状
        if len(original_shape) == 4:
            anomaly_scores = anomaly_scores.view(b, h, w, 1).permute(0, 3, 1, 2)
        
        return anomaly_scores

    def reset(self) -> None:
        """清空记忆库"""
        self.features.zero_()
        self.ptr.zero_()
        self.is_full.zero_()
