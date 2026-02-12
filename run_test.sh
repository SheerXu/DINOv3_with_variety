#!/bin/bash
# supervised
python src/test_supervised.py \
    --config configs/supervised.yaml \
    --checkpoint outputs/model_epoch_x.pth \
    --input test_images/ \
    --output test_results/supervised
# AD-DINOv3
# python src/test_ad.py \
#     --config configs/ad_dinov3.yaml \
#     --checkpoint outputs_ad/ad_model_epoch_x.pth \
#     --input test_images/ \
#     --threshold 0.5 \
#     --output test_results/ad_dinov3
