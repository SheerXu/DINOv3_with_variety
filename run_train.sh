#!/bin/bash
# AD-DINOv3 Adapter（使用 configs/ad_dinov3.yaml 中的 CLIP 本地权重配置）
python src/train_ad.py --config configs/ad_dinov3.yaml

# supervised（如需）
# python src/train_supervised.py --config configs/supervised.yaml
