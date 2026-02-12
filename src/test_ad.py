from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ad_adapter import CLIPInplanted as AdapterModel
from src.models.dinov3_backbone import Dinov3Backbone
from src.utils.ad_clip import create_clip_model_and_tokenizer, encode_text_with_prompt_ensemble
from src.utils.config import load_config
from src.utils.visualization import visualize_segmentation

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test AD-DINOv3 Adapter Model")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Adapter checkpoint path")
    parser.add_argument("--input", type=str, required=True, help="Input image or directory")
    parser.add_argument("--output", type=str, default="test_results/ad_dinov3", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Anomaly threshold")
    parser.add_argument("--show", action="store_true", help="Show visualization")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--object-name", type=str, default="defect", help="Prompt object name (fallback)")
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


def infer_anomaly_map(
    clip_model,
    clip_tokenizer,
    adapter_model,
    dino_model,
    image_tensor: torch.Tensor,
    device: torch.device,
    obj_name: str,
) -> np.ndarray:
    image_tensor = image_tensor.to(device)
    image_tensor = normalize_imagenet(image_tensor)

    batch_size = image_tensor.shape[0]
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

    cls_token, patch_tokens = get_feature_dinov3(image_tensor, device, dino_model)

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
    anomaly_map = anomaly_map_cross_modal[:, 1, :, :]
    return anomaly_map.squeeze(0).detach().cpu().numpy()


def preprocess_image(image_path: Path, resize: tuple | None):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    if resize is not None:
        image_resized = F.resize(image, resize, interpolation=Image.BILINEAR)
    else:
        image_resized = image

    image_tensor = F.to_tensor(image_resized).unsqueeze(0)
    return image_tensor, np.array(image), original_size


def resolve_object_name(image_path: Path, fallback_name: str) -> str:
    # 参考仓库默认从路径中取类名；这里保留回退策略
    parts = image_path.as_posix().split("/")
    if len(parts) >= 4:
        return parts[-4]
    return fallback_name


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


def main():
    args = parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    resize = tuple(data_cfg["resize"]) if data_cfg.get("resize") else None

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from: {args.checkpoint}")
    clip_model, clip_tokenizer, adapter_model, dino_model = load_models(cfg, args.checkpoint, device)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)

    if input_path.is_file():
        image_tensor, image_np, original_size = preprocess_image(input_path, resize)
        obj_name = resolve_object_name(input_path, args.object_name)
        anomaly_map = infer_anomaly_map(
            clip_model, clip_tokenizer, adapter_model, dino_model, image_tensor, device, obj_name
        )

        anomaly_image = Image.fromarray((anomaly_map * 255).astype(np.uint8))
        anomaly_image = anomaly_image.resize(original_size, Image.BILINEAR)
        anomaly_map = np.array(anomaly_image).astype(np.float32) / 255.0

        score_min = float(anomaly_map.min())
        score_max = float(anomaly_map.max())
        anomaly_map_norm = (
            (anomaly_map - score_min) / (score_max - score_min)
            if score_max - score_min > 1e-8
            else np.zeros_like(anomaly_map, dtype=np.float32)
        )
        pred_mask = (anomaly_map_norm > args.threshold).astype(np.uint8)

        print(f"Processing: {input_path}")
        print(f"  Raw Score - min: {score_min:.4f}, max: {float(anomaly_map.max()):.4f}, mean: {float(anomaly_map.mean()):.4f}")
        print(
            f"  Norm Score - min: {float(anomaly_map_norm.min()):.4f}, max: {float(anomaly_map_norm.max()):.4f}, mean: {float(anomaly_map_norm.mean()):.4f}"
        )

        output_path = output_dir / f"{input_path.stem}_result.png"
        visualize_segmentation(
            image=image_np,
            pred_mask=pred_mask,
            gt_mask=None,
            label_map={"background": 0, "anomaly": 1},
            save_path=output_path,
            show=args.show,
        )
    elif input_path.is_dir():
        image_exts = data_cfg.get("image_exts", [".png", ".jpg", ".jpeg"])
        image_files: List[Path] = []
        for ext in image_exts:
            image_files.extend(input_path.rglob(f"*{ext}"))

        print(f"Found {len(image_files)} images in {input_path} (including subdirectories)")

        raw_results = []
        for image_file in sorted(image_files):
            image_tensor, image_np, original_size = preprocess_image(image_file, resize)
            obj_name = resolve_object_name(image_file, args.object_name)
            anomaly_map = infer_anomaly_map(
                clip_model, clip_tokenizer, adapter_model, dino_model, image_tensor, device, obj_name
            )

            anomaly_image = Image.fromarray((anomaly_map * 255).astype(np.uint8))
            anomaly_image = anomaly_image.resize(original_size, Image.BILINEAR)
            anomaly_map = np.array(anomaly_image).astype(np.float32) / 255.0
            raw_results.append((image_file, image_np, anomaly_map, image_file.relative_to(input_path)))

        if len(raw_results) > 0:
            global_min = min(float(r[2].min()) for r in raw_results)
            global_max = max(float(r[2].max()) for r in raw_results)
            denom = global_max - global_min
        else:
            global_min, global_max, denom = 0.0, 1.0, 1.0

        print(f"Global score range - min: {global_min:.4f}, max: {global_max:.4f}")

        for image_file, image_np, anomaly_map, relative_path in raw_results:
            if denom > 1e-8:
                anomaly_map_norm = (anomaly_map - global_min) / denom
            else:
                anomaly_map_norm = np.zeros_like(anomaly_map, dtype=np.float32)

            pred_mask = (anomaly_map_norm > args.threshold).astype(np.uint8)

            anomaly_pixels = int((anomaly_map_norm > args.threshold).sum())
            total_pixels = anomaly_map_norm.size
            anomaly_ratio = anomaly_pixels / max(total_pixels, 1) * 100

            print(f"Processing: {image_file}")
            print(
                f"  Raw Score - min: {float(anomaly_map.min()):.4f}, max: {float(anomaly_map.max()):.4f}, mean: {float(anomaly_map.mean()):.4f}"
            )
            print(
                f"  Norm Score - min: {float(anomaly_map_norm.min()):.4f}, max: {float(anomaly_map_norm.max()):.4f}, mean: {float(anomaly_map_norm.mean()):.4f}"
            )
            print(f"  Anomaly Ratio: {anomaly_ratio:.2f}% (threshold={args.threshold})")

            output_subdir = output_dir / relative_path.parent
            output_subdir.mkdir(parents=True, exist_ok=True)
            output_path = output_subdir / f"{image_file.stem}_result.png"

            visualize_segmentation(
                image=image_np,
                pred_mask=pred_mask,
                gt_mask=None,
                label_map={"background": 0, "anomaly": 1},
                save_path=output_path,
                show=args.show,
            )
    else:
        raise ValueError(f"Invalid input path: {input_path}")

    print(f"\n✓ All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
