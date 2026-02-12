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

from src.datasets.supervised_dataset import PolygonJsonDataset, pad_collate
from src.models.dinov3_backbone import Dinov3Backbone
from src.models.supervised_segmentation import SegmentationModel
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

    train_ds = PolygonJsonDataset(
        root_dir=data_cfg["train_dir"],
        image_exts=data_cfg["image_exts"],
        label_map=data_cfg.get("label_map"),
        resize=tuple(data_cfg["resize"]) if data_cfg.get("resize") else None,
        random_flip=data_cfg.get("random_flip", False),
        ignore_index=data_cfg.get("ignore_index", 255),
    )

    num_classes = (max(train_ds.label_map.values()) + 1) if train_ds.label_map else 1

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg.get("batch_size", 2),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 2),
        collate_fn=lambda b: pad_collate(b, data_cfg.get("ignore_index", 255)),
    )

    backbone = Dinov3Backbone(
        dinov3_repo_path=model_cfg["dinov3_repo_path"],
        model_name=model_cfg["model_name"],
        pretrained_path=model_cfg.get("pretrained_path"),
        patch_size=model_cfg["patch_size"],
        embed_dim=model_cfg["embed_dim"],
        download=model_cfg.get("download", False)
    )
    model = SegmentationModel(backbone, num_classes=num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.05),
    )
    # AMP 混合精度训练 - 梯度缩放
    scaler = torch.amp.GradScaler('cuda', enabled=train_cfg.get("amp", True))
    save_dir = ensure_dir(train_cfg.get("save_dir", "outputs"))

    epochs = train_cfg.get("epochs", 50)
    log_interval = train_cfg.get("log_interval", 10)
    ignore_index = data_cfg.get("ignore_index", 255)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for step, (images, masks) in enumerate(train_loader, start=1):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=train_cfg.get("amp", True)):
                logits = model(images)
                loss = F.cross_entropy(logits, masks, ignore_index=ignore_index)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            if step % log_interval == 0:
                avg = total_loss / step
                print(f"Epoch {epoch} Step {step} Loss {avg:.4f}")

        if epoch % train_cfg.get("save_every", 5) == 0:
            ckpt_path = Path(save_dir) / f"model_epoch_{epoch}.pth"
            torch.save({"model": model.state_dict()}, ckpt_path)
            print(f"Saved: {ckpt_path}")


if __name__ == "__main__":
    main()
