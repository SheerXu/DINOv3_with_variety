from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.dinov3_backbone import Dinov3Backbone
from src.models.memory_bank import MemoryBank


class AnomalyDetector(nn.Module):
    """AD-DINOv3 异常检测器：Teacher-Student 架构"""

    def __init__(
        self,
        backbone: Dinov3Backbone,
        memory_bank_size: int = 1000,
        use_multi_scale: bool = True,
    ) -> None:
        super().__init__()
        
        # Teacher: 冻结的预训练 DINOv3
        self.teacher = backbone
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        # Student: 可微调的分支
        self.student = copy.deepcopy(backbone)
        
        self.use_multi_scale = use_multi_scale
        self.embed_dim = backbone.embed_dim
        
        # 特征记忆库
        self.memory_bank = MemoryBank(
            feature_dim=self.embed_dim,
            bank_size=memory_bank_size,
        )
        
        # 特征融合
        if use_multi_scale:
            self.feature_fusion = nn.Sequential(
                nn.Conv2d(self.embed_dim * 2, self.embed_dim, 1),
                nn.BatchNorm2d(self.embed_dim),
                nn.ReLU(inplace=True),
            )

    @torch.no_grad()
    def extract_teacher_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取 Teacher 特征（推理模式）"""
        self.teacher.eval()
        return self.teacher(x)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """前向传播
        
        Args:
            x: 输入图像 (B, 3, H, W)
            return_features: 是否返回特征（用于记忆库更新）
            
        Returns:
            anomaly_map: 异常分数图 (B, 1, H, W)
            features: (可选) student 特征
        """
        # Teacher 特征（推理模式）
        with torch.no_grad():
            teacher_feat = self.extract_teacher_features(x)
        
        # Student 特征
        student_feat = self.student(x)
        
        if self.use_multi_scale:
            # 融合 Teacher 和 Student 特征
            combined = torch.cat([teacher_feat, student_feat], dim=1)
            fused_feat = self.feature_fusion(combined)
            feat_for_distance = fused_feat
        else:
            feat_for_distance = student_feat
        
        # 计算异常分数
        anomaly_map = self.memory_bank.get_nearest_distance(feat_for_distance, k=3)
        
        # 上采样到原始分辨率
        anomaly_map = F.interpolate(
            anomaly_map,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        
        if return_features:
            return anomaly_map, teacher_feat
        return anomaly_map

    def update_memory_bank(self, features: torch.Tensor) -> None:
        """更新记忆库（仅用正常样本）"""
        self.memory_bank.update(features)

    def compute_distance_loss(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        """计算 Student 和 Teacher 特征距离损失"""
        student_norm = F.normalize(student_feat, dim=1)
        teacher_norm = F.normalize(teacher_feat, dim=1)
        return 1 - F.cosine_similarity(student_norm, teacher_norm, dim=1).mean()
