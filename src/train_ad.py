from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.anomaly_dataset import AnomalyDetectionDataset
from src.models.anomaly_detector import AnomalyDetector
from src.models.dinov3_backbone import Dinov3Backbone
from src.utils.config import ensure_dir, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_cfg: Dict = cfg["data"]
    model_cfg: Dict = cfg["model"]
    train_cfg: Dict = cfg["train"]

    # 数据集
    train_ds = AnomalyDetectionDataset(
        root_dir=data_cfg["train_dir"],
        image_exts=data_cfg["image_exts"],
        mode="train",
        resize=tuple(data_cfg["resize"]) if data_cfg.get("resize") else None,
        use_anomaly_generation=train_cfg.get("use_anomaly_generation", True),
        anomaly_ratio=train_cfg.get("anomaly_ratio", 0.5),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 2),
    )

    # 模型
    backbone = Dinov3Backbone(
        dinov3_repo_path=model_cfg["dinov3_repo_path"],
        model_name=model_cfg["model_name"],
        pretrained_path=model_cfg.get("pretrained_path"),
        patch_size=model_cfg["patch_size"],
        embed_dim=model_cfg["embed_dim"],
        download=model_cfg.get("download", False)
    )

    detector = AnomalyDetector(
        backbone=backbone,
        memory_bank_size=model_cfg.get("memory_bank_size", 1000),
        use_multi_scale=model_cfg.get("use_multi_scale", True),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector.to(device)

    # 优化器（仅优化 Student 分支）
    optimizer = torch.optim.AdamW(
        detector.student.parameters(),
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.05),
    )

    amp_requested = bool(train_cfg.get("amp", True))
    amp_enabled = amp_requested and device.type == "cuda"
    if amp_requested and not amp_enabled:
        print("Warning: amp=True requested but CUDA is not available; AMP is disabled.")

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    save_dir = ensure_dir(train_cfg.get("save_dir", "outputs_ad"))

    epochs = train_cfg.get("epochs", 30)
    log_interval = train_cfg.get("log_interval", 10)
    warmup_epochs = train_cfg.get("warmup_epochs", 5)

    print(f"Training AD-DINOv3 for {epochs} epochs")
    print(f"Memory bank warmup: {warmup_epochs} epochs")
    print(f"Memory bank size: {model_cfg.get('memory_bank_size', 1000)}")
    print(f"Initial bank state - ptr: {detector.memory_bank.ptr.item()}, is_full: {detector.memory_bank.is_full.item()}\n")

    warned_no_grad = False

    for epoch in range(1, epochs + 1):
        detector.train()
        total_loss = 0.0
        total_focal = 0.0
        total_dist = 0.0
        
        # 打印当前 epoch 的记忆库状态
        bank_ptr = int(detector.memory_bank.ptr.item())
        print(f"Epoch {epoch} - Memory bank contains {bank_ptr} samples")

        for step, (images, masks, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            # 前向传播（在 autocast 下）
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                anomaly_map, teacher_feat = detector(images, return_features=True)
                
                # Feature distance loss (Student vs Teacher)
                if epoch > warmup_epochs:
                    student_feat = detector.student(images)
                    dist_loss = detector.compute_distance_loss(student_feat, teacher_feat)
                else:
                    dist_loss = None
            
            # 第一个 batch 输出调试信息
            if epoch == 1 and step == 1:
                print(f"  anomaly_map stats - min: {anomaly_map.min().item():.4f}, "
                      f"max: {anomaly_map.max().item():.4f}, "
                      f"mean: {anomaly_map.mean().item():.4f}, "
                      f"has_nan: {torch.isnan(anomaly_map).any().item()}, "
                      f"has_inf: {torch.isinf(anomaly_map).any().item()}")
            
            # Focal Loss for anomaly segmentation（在 autocast 外部）
            # 转换为 float32 以确保数值稳定性
            anomaly_map = anomaly_map.float()
            pred_flat = anomaly_map.view(-1)
            target_flat = masks.view(-1).float()
            
            # 检查并修正异常值
            pred_flat = torch.nan_to_num(pred_flat, nan=0.5, posinf=1.0, neginf=0.0)
            pred_flat = pred_flat.clamp(min=1e-7, max=1.0 - 1e-7)
            
            # Focal Loss 计算
            bce_loss = F.binary_cross_entropy(
                pred_flat, target_flat, reduction="none"
            )
            pt = torch.exp(-bce_loss)
            focal_loss = ((1 - pt) ** 2 * bce_loss).mean()
            
            # 合并损失
            if dist_loss is not None:
                loss = focal_loss + 0.5 * dist_loss.float()
                total_dist += dist_loss.item()
            else:
                loss = focal_loss

            total_focal += focal_loss.item()

            # 反向传播（处理 AMP）
            if amp_enabled:
                # 使用 scaler（仅对在 autocast 内的操作）
                # 由于损失部分在 autocast 外，需要特殊处理
                scaler.scale(loss).backward()

                # 如果这一轮没有任何梯度（例如：记忆库为空导致 anomaly_map 与参数无关），
                # GradScaler.step() 会触发断言："No inf checks were recorded for this optimizer."。
                has_any_grad = any(
                    (p.grad is not None)
                    for group in optimizer.param_groups
                    for p in group["params"]
                )
                if has_any_grad:
                    scaler.unscale_(optimizer)  # 在 step 前 unscale
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if not warned_no_grad:
                        print(
                            "Warning: no gradients found for optimizer params; skipping optimizer step this iteration."
                        )
                        warned_no_grad = True
            else:
                # 不使用 AMP
                loss.backward()
                optimizer.step()

            # 更新记忆库（仅用正常样本）
            with torch.no_grad():
                normal_mask = labels == 0
                if normal_mask.any():
                    normal_feat = teacher_feat[normal_mask]
                    detector.update_memory_bank(normal_feat)

            total_loss += loss.item()

            if step % log_interval == 0:
                avg_loss = total_loss / step
                avg_focal = total_focal / step
                avg_dist = total_dist / step if epoch > warmup_epochs else 0
                print(
                    f"Epoch {epoch}/{epochs} Step {step} "
                    f"Loss {avg_loss:.4f} (Focal {avg_focal:.4f}, Dist {avg_dist:.4f})"
                )

        # 保存检查点
        if epoch % train_cfg.get("save_every", 5) == 0:
            ckpt_path = Path(save_dir) / f"ad_model_epoch_{epoch}.pth"
            torch.save(
                {
                    "detector": detector.state_dict(),
                    "memory_bank": detector.memory_bank.state_dict(),
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print(f"Saved: {ckpt_path}")

    print("Training completed!")


if __name__ == "__main__":
    main()
