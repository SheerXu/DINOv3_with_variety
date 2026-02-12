from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.anomaly_dataset import AnomalyDetectionDataset
from src.loss.ad_losses import BinaryDiceLoss, FocalLoss
from src.models.ad_adapter import CLIPInplanted as AdapterModel
from src.models.dinov3_backbone import Dinov3Backbone
from src.utils.ad_clip import create_clip_model_and_tokenizer, encode_text_with_prompt_ensemble
from src.utils.config import ensure_dir, load_config


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AD-DINOv3 Adapter-style")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def normalize_imagenet(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def get_feature_dinov3(batch_img: torch.Tensor, device: torch.device, dino_model):
    with torch.no_grad():
        depth = len(dino_model.blocks)
        idxs = [int(round((depth - 1) * k / (4 - 1))) for k in range(4)]
        layers = sorted(set([max(0, min(depth - 1, idx)) for idx in idxs]))

        patch_tokens_dict = {idx: [] for idx in layers}
        cls_tokens_dict = {idx: [] for idx in layers}

        anchor = getattr(dino_model, "norm", None) or getattr(dino_model, "fc_norm", None)
        if anchor is None:
            raise ValueError("DINO model has no norm/fc_norm module for feature extraction.")

        batch_size = batch_img.shape[0]
        for sample_idx in range(batch_size):
            tokens_dict = {}
            handles = []

            image = batch_img[sample_idx].unsqueeze(0).to(device)

            for layer_idx in layers:
                def _mk_hook(idx):
                    def _hook(_module, _inp, out):
                        tokens_dict[idx] = anchor(out[0]).detach().cpu()

                    return _hook

                handles.append(dino_model.blocks[layer_idx].register_forward_hook(_mk_hook(layer_idx)))

            _ = dino_model(image)

            for handle in handles:
                handle.remove()

            for layer_idx, tokens in tokens_dict.items():
                patch_tokens = tokens[:, 5:, :]
                patch_tokens = (patch_tokens - patch_tokens.mean(dim=1, keepdim=True)) / (
                    patch_tokens.std(dim=1, keepdim=True) + 1e-6
                )
                cls_tokens = tokens[:, 0, :].unsqueeze(1)

                patch_tokens_dict[layer_idx].append(patch_tokens)
                cls_tokens_dict[layer_idx].append(cls_tokens)

        patch_tokens = [torch.cat(patch_tokens_dict[idx], dim=0).to(device) for idx in layers]
        cls_token = [torch.cat(cls_tokens_dict[idx], dim=0).to(device) for idx in layers]
        return cls_token, patch_tokens


def get_anomaly_map(
    clip_model,
    clip_tokenizer,
    images: torch.Tensor,
    masks: torch.Tensor,
    labels: torch.Tensor,
    obj_name: str,
    device: torch.device,
    adapter_model,
    dino_model,
):
    masks = masks.clone()
    masks[masks > 0.5] = 1
    masks[masks <= 0.5] = 0

    batch_size = images.shape[0]

    text_feature = torch.zeros(batch_size, 768, 2, device=device)
    with torch.no_grad():
        for i in range(batch_size):
            text_feature[i] = encode_text_with_prompt_ensemble(clip_model, clip_tokenizer, obj_name, device)

    adjusted_feats_0 = []
    adjusted_feats_1 = []
    for i in range(batch_size):
        f0 = adapter_model.prompt_adapter[0](text_feature[i, :, 0])[0]
        f1 = adapter_model.prompt_adapter[1](text_feature[i, :, 1])[0]
        adjusted_feats_0.append(f0)
        adjusted_feats_1.append(f1)

    adjusted_feats_0 = torch.stack(adjusted_feats_0, dim=0)
    adjusted_feats_1 = torch.stack(adjusted_feats_1, dim=0)
    adjusted_text_feature = torch.cat([adjusted_feats_0, adjusted_feats_1], dim=1).view(batch_size, 768, 2)

    cls_token, patch_tokens = get_feature_dinov3(images, device, dino_model)

    anomaly_map = []
    anomaly_maps_cross_modal = []
    global_anomaly_scores = []

    for i in range(4):
        cls_features = adapter_model.cls_token_adapter[i](cls_token[i])[0]
        cls_features = cls_features / (cls_features.norm(dim=-1, keepdim=True) + 1e-6)

        patch_features = adapter_model.patch_token_adapter[i](patch_tokens[i])[0]
        patch_features = patch_features / (patch_features.norm(dim=-1, keepdim=True) + 1e-6)

        anomaly_map_cross_modal = 100 * patch_features @ adjusted_text_feature
        s = int((patch_tokens[i].shape[1]) ** 0.5)
        anomaly_map_cross_modal = F.interpolate(
            anomaly_map_cross_modal.permute(0, 2, 1).view(-1, 2, s, s),
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        anomaly_map_cross_modal = torch.softmax(anomaly_map_cross_modal, dim=1)
        anomaly_maps_cross_modal.append(anomaly_map_cross_modal)

        anomaly_awareness_cls_patch = 10 * patch_features @ cls_features.squeeze().unsqueeze(-1)
        anomaly_awareness_cls_patch = F.interpolate(
            anomaly_awareness_cls_patch.permute(0, 2, 1).view(-1, 1, s, s),
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        anomaly_awareness_cls_patch = torch.sigmoid(anomaly_awareness_cls_patch)
        anomaly_map.append(torch.cat([1 - anomaly_awareness_cls_patch, anomaly_awareness_cls_patch], dim=1))

        anomaly_score = 100 * cls_features @ adjusted_text_feature
        global_anomaly_scores.append(anomaly_score)

    anomaly_map_cross_modal = torch.mean(torch.stack(anomaly_maps_cross_modal, dim=0), dim=0)
    anomaly_awareness = torch.mean(torch.stack(anomaly_map, dim=0), dim=0)
    global_anomaly_score = torch.mean(torch.stack(global_anomaly_scores, dim=0), dim=0)

    return anomaly_awareness, masks.unsqueeze(1), anomaly_map_cross_modal, global_anomaly_score, labels


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_cfg: Dict = cfg["data"]
    model_cfg: Dict = cfg["model"]
    train_cfg: Dict = cfg["train"]
    clip_cfg: Dict = model_cfg.get("clip", {})

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
        batch_size=data_cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 2),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DINOv3 backbone（参考仓库：冻结特征提取器）
    backbone = Dinov3Backbone(
        dinov3_repo_path=model_cfg["dinov3_repo_path"],
        model_name=model_cfg["model_name"],
        pretrained_path=model_cfg.get("pretrained_path"),
        patch_size=model_cfg["patch_size"],
        embed_dim=model_cfg["embed_dim"],
        download=model_cfg.get("download", False),
    )
    dino_model = backbone.model.to(device)
    dino_model.eval()
    for param in dino_model.parameters():
        param.requires_grad = False

    # CLIP（参考仓库：ViT-L-14-336）
    clip_model_name = clip_cfg.get("model_name", "ViT-L-14-336")
    clip_pretrained = clip_cfg.get("pretrained", "CLIP/ViT-L-14-336px.pt")
    clip_download = bool(clip_cfg.get("download", False))

    clip_model, clip_tokenizer = create_clip_model_and_tokenizer(
        model_name=clip_model_name,
        pretrained=clip_pretrained,
        device=device,
        allow_download=clip_download,
    )
    clip_model.to(device)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False

    # Adapter model（参考仓库核心可训练模块）
    adapter_model = AdapterModel(c_in=model_cfg.get("embed_dim", 1024), _device=device)
    adapter_model.to(device)
    adapter_model.train()

    update_params = ["patch_token_adapter", "cls_token_adapter", "prompt_adapter"]
    params_to_update: List[torch.nn.Parameter] = []
    for name, param in adapter_model.named_parameters():
        if any(key in name for key in update_params):
            params_to_update.append(param)

    optimizer = torch.optim.AdamW(
        params_to_update,
        lr=train_cfg.get("lr", 1e-5),
        betas=(0.9, 0.999),
        weight_decay=train_cfg.get("weight_decay", 1e-2),
    )

    total_steps = max(1, train_cfg.get("epochs", 20) * max(1, len(train_loader)))
    warmup_steps = int(0.03 * total_steps)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    save_dir = ensure_dir(train_cfg.get("save_dir", "outputs_ad"))
    epochs = train_cfg.get("epochs", 20)
    save_every = train_cfg.get("save_every", 10)
    log_interval = train_cfg.get("log_interval", 10)
    obj_name = train_cfg.get("object_name", "defect")

    print(f"Training AD-DINOv3 Adapter for {epochs} epochs")
    print(f"Using object prompt name: {obj_name}")

    global_step = 0
    for epoch in range(epochs):
        start_time = time.time()
        adapter_model.train()

        awareness_loss_list = []
        seg_loss_list = []
        global_anomaly_loss_list = []
        total_loss_list = []

        for step, (images, masks, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            # 对齐参考仓库数据预处理（ImageNet 归一化）
            images_norm = normalize_imagenet(images)

            anomaly_map, mask, anomaly_map_cross_modal, global_anomaly_score, gt_label = get_anomaly_map(
                clip_model=clip_model,
                clip_tokenizer=clip_tokenizer,
                images=images_norm,
                masks=masks,
                labels=labels,
                obj_name=obj_name,
                device=device,
                adapter_model=adapter_model,
                dino_model=dino_model,
            )

            anomaly_awareness_loss = loss_focal(anomaly_map, mask) + loss_dice(anomaly_map[:, 1, :, :], mask)
            seg_loss = loss_focal(anomaly_map_cross_modal, mask) + loss_dice(anomaly_map_cross_modal[:, 1, :, :], mask)
            global_anomaly_loss = F.cross_entropy(global_anomaly_score.squeeze(1), gt_label.long())
            loss = 0.25 * anomaly_awareness_loss + 0.5 * seg_loss + 0.25 * global_anomaly_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            awareness_loss_list.append(float(anomaly_awareness_loss.item()))
            seg_loss_list.append(float(seg_loss.item()))
            global_anomaly_loss_list.append(float(global_anomaly_loss.item()))
            total_loss_list.append(float(loss.item()))

            global_step += 1
            if step % log_interval == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} Step {step}/{len(train_loader)} "
                    f"Loss {np.mean(total_loss_list):.4f} "
                    f"(Awareness {np.mean(awareness_loss_list):.4f}, "
                    f"Seg {np.mean(seg_loss_list):.4f}, Global {np.mean(global_anomaly_loss_list):.4f})"
                )

        if (epoch + 1) % save_every == 0:
            ckpt_path = Path(save_dir) / f"ad_adapter_epoch_{epoch + 1}.pth"
            torch.save(
                {
                    "cls_token_adapter": adapter_model.cls_token_adapter.state_dict(),
                    "patch_token_adapter": adapter_model.patch_token_adapter.state_dict(),
                    "prompt_adapter": adapter_model.prompt_adapter.state_dict(),
                    "epoch": epoch + 1,
                    "config": cfg,
                },
                ckpt_path,
            )
            print(f"Saved: {ckpt_path}")

        print(
            f"epoch_{epoch + 1}: awareness_loss={np.mean(awareness_loss_list):.6f}, "
            f"seg_loss={np.mean(seg_loss_list):.6f}, "
            f"global_anomaly_loss={np.mean(global_anomaly_loss_list):.6f}, "
            f"total_loss={np.mean(total_loss_list):.6f}, "
            f"time={time.time() - start_time:.2f}s"
        )

    print("Training completed!")


if __name__ == "__main__":
    main()
