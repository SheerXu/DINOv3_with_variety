from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.datasets.anomaly_dataset import AnomalyDetectionDataset
from src.models.ad_adapter import CLIPInplanted as AdapterModel
from src.models.dinov3_backbone import Dinov3Backbone
from src.utils.ad_clip import create_clip_model_and_tokenizer, encode_text_with_prompt_ensemble
from src.utils.config import load_config


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--object-name", type=str, default="defect")
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

        for sample_idx in range(batch_img.shape[0]):
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


def infer_anomaly_map_batch(
    clip_model,
    clip_tokenizer,
    adapter_model,
    dino_model,
    image_tensor: torch.Tensor,
    device: torch.device,
    obj_name: str,
) -> torch.Tensor:
    image_tensor = normalize_imagenet(image_tensor.to(device))

    batch_size = image_tensor.shape[0]
    text_feature = torch.zeros(batch_size, 768, 2, device=device)
    with torch.no_grad():
        for i in range(batch_size):
            text_feature[i] = encode_text_with_prompt_ensemble(clip_model, clip_tokenizer, obj_name, device)

    adjusted_feats_0: List[torch.Tensor] = []
    adjusted_feats_1: List[torch.Tensor] = []
    for i in range(batch_size):
        f0 = adapter_model.prompt_adapter[0](text_feature[i, :, 0])[0]
        f1 = adapter_model.prompt_adapter[1](text_feature[i, :, 1])[0]
        adjusted_feats_0.append(f0)
        adjusted_feats_1.append(f1)

    adjusted_feats_0 = torch.stack(adjusted_feats_0, dim=0)
    adjusted_feats_1 = torch.stack(adjusted_feats_1, dim=0)
    adjusted_text_feature = torch.cat([adjusted_feats_0, adjusted_feats_1], dim=1).view(batch_size, 768, 2)

    _, patch_tokens = get_feature_dinov3(image_tensor, device, dino_model)

    anomaly_maps_cross_modal = []
    for i in range(4):
        patch_features = adapter_model.patch_token_adapter[i](patch_tokens[i])[0]
        patch_features = patch_features / (patch_features.norm(dim=-1, keepdim=True) + 1e-6)

        anomaly_map_cross_modal = 100 * patch_features @ adjusted_text_feature
        s = int((patch_tokens[i].shape[1]) ** 0.5)
        anomaly_map_cross_modal = F.interpolate(
            anomaly_map_cross_modal.permute(0, 2, 1).view(-1, 2, s, s),
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        anomaly_map_cross_modal = torch.softmax(anomaly_map_cross_modal, dim=1)
        anomaly_maps_cross_modal.append(anomaly_map_cross_modal)

    anomaly_map_cross_modal = torch.mean(torch.stack(anomaly_maps_cross_modal, dim=0), dim=0)
    return anomaly_map_cross_modal[:, 1, :, :]


def load_models(cfg: Dict, checkpoint_path: str, device: torch.device):
    model_cfg: Dict = cfg["model"]
    clip_cfg: Dict = model_cfg.get("clip", {})

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

    adapter_model = AdapterModel(c_in=model_cfg.get("embed_dim", 1024), _device=device)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    adapter_model.cls_token_adapter.load_state_dict(ckpt["cls_token_adapter"], strict=False)
    adapter_model.patch_token_adapter.load_state_dict(ckpt["patch_token_adapter"], strict=False)
    adapter_model.prompt_adapter.load_state_dict(ckpt["prompt_adapter"], strict=False)
    adapter_model.to(device)
    adapter_model.eval()

    print("Loaded adapter checkpoint successfully.")
    return clip_model, clip_tokenizer, adapter_model, dino_model


def compute_metrics(anomaly_scores: np.ndarray, labels: np.ndarray) -> Dict:
    """计算异常检测指标
    
    Args:
        anomaly_scores: 异常分数 (N,)
        labels: 真实标签 (N,) 0=正常, 1=异常
        
    Returns:
        metrics: 包含 AUROC, AUPRC, 最优 F1 等
    """
    # Image-level AUROC
    auroc = roc_auc_score(labels, anomaly_scores)
    
    # Precision-Recall
    precision, recall, thresholds = precision_recall_curve(labels, anomaly_scores)
    auprc = auc(recall, precision)
    
    # 计算最优 F1
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else thresholds[-1]
    
    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "Best_F1": best_f1,
        "Best_Threshold": best_threshold,
        "Precision@BestF1": precision[best_f1_idx],
        "Recall@BestF1": recall[best_f1_idx],
    }


def compute_threshold_metrics(anomaly_scores: np.ndarray, labels: np.ndarray, threshold: float) -> Dict:
    """基于固定阈值的二值指标（与 test_ad 可视化口径一致）"""
    pred = (anomaly_scores > threshold).astype(np.uint8)
    gt = labels.astype(np.uint8)

    return {
        "Threshold": threshold,
        "Accuracy": accuracy_score(gt, pred),
        "Precision": precision_score(gt, pred, zero_division=0),
        "Recall": recall_score(gt, pred, zero_division=0),
        "F1": f1_score(gt, pred, zero_division=0),
        "Positive_Ratio": float(pred.mean()),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_cfg: Dict = cfg["data"]
    model_cfg: Dict = cfg["model"]

    # 测试数据集
    test_ds = AnomalyDetectionDataset(
        root_dir=data_cfg["val_dir"],
        image_exts=data_cfg["image_exts"],
        mode="test",
        resize=tuple(data_cfg["resize"]) if data_cfg.get("resize") else None,
        use_anomaly_generation=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 2),
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    clip_model, clip_tokenizer, adapter_model, dino_model = load_models(cfg, args.checkpoint, device)
    obj_name = args.object_name

    print("Evaluating AD-DINOv3...")

    all_scores = []
    all_labels = []
    pixel_scores = []
    pixel_labels = []

    threshold = float(args.threshold)

    with torch.no_grad():
        for images, masks, labels in test_loader:
            images = images.to(device)
            masks_np = masks.numpy()
            labels_np = labels.numpy()

            anomaly_map = infer_anomaly_map_batch(
                clip_model=clip_model,
                clip_tokenizer=clip_tokenizer,
                adapter_model=adapter_model,
                dino_model=dino_model,
                image_tensor=images,
                device=device,
                obj_name=obj_name,
            ).cpu().numpy()

            # Image-level score: 取最大值
            img_score = anomaly_map.max()
            all_scores.append(img_score)
            all_labels.append(labels_np[0])

            # Pixel-level
            pixel_scores.append(anomaly_map.flatten())
            pixel_labels.append((masks_np.flatten() > 0.5).astype(np.uint8))

    # Image-level metrics
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    img_metrics = compute_metrics(all_scores, all_labels)
    img_threshold_metrics = compute_threshold_metrics(all_scores, all_labels, threshold)

    print("\n=== AD-DINOv3 Image-Level Metrics ===")
    for key, value in img_metrics.items():
        print(f"{key}: {value:.4f}")
    print("\n=== AD-DINOv3 Image-Level Metrics @ Fixed Threshold ===")
    for key, value in img_threshold_metrics.items():
        if key == "Threshold":
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value:.4f}")

    # Pixel-level metrics
    pixel_scores = np.concatenate(pixel_scores)
    pixel_labels = np.concatenate(pixel_labels)
    pixel_metrics = compute_metrics(pixel_scores, pixel_labels)
    pixel_threshold_metrics = compute_threshold_metrics(pixel_scores, pixel_labels, threshold)

    print("\n=== AD-DINOv3 Pixel-Level Metrics ===")
    for key, value in pixel_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\n=== AD-DINOv3 Pixel-Level Metrics @ Fixed Threshold ===")
    for key, value in pixel_threshold_metrics.items():
        if key == "Threshold":
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
