#!/bin/bash
# AD-DINOv3 Adapter（使用 configs/ad_dinov3.yaml 中的 CLIP 本地权重配置）
python src/test_ad.py \
    --config configs/ad_dinov3.yaml \
    --checkpoint outputs_ad/ad_adapter_epoch_x.pth \
    --input test_data/ \
    --threshold 0.5 \
    --output test_results/ad_dinov3

# supervised（如需）
# python src/test_supervised.py \
#     --config configs/supervised.yaml \
#     --checkpoint outputs/model_epoch_x.pth \
#     --input test_data/ \
#     --output test_results/supervised
