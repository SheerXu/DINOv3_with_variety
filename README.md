# DINOv3 工业缺陷检测项目

本项目提供**两种**基于 DINOv3 的缺陷检测方法：
1. **有监督语义分割** (`*_supervised.py`)：适用于已知缺陷类别 + 充足像素级标注
2. **AD-DINOv3 异常检测** (`*_ad.py`)：适用于未知缺陷类型 + 标注稀缺场景

## 快速开始

### 环境安装
```bash
pip install -r requirements.txt
```

### 数据准备
- 训练数据：同一目录下的图片与同名 JSON（polygon标注），例如：
  - `xxx.png` + `xxx.json`
- 示例模板：`train_data/template/`
- DINOv3 预训练权重：放入 `dinov3_pretrained/`

## 配置文件

### supervised.yaml（有监督分割）
```yaml
data:
  train_dir: train_data/template
  label_map:  # 类别映射（首位为bg）
    - bg
    - 方槽毛刺
    - 焊接面毛刺
  resize: null  # 或 [512, 512]

model:
  dinov3_repo_path: dinov3  # DINOv3源码路径
  pretrained_path: dinov3_pretrained/dinov3_vitb16.pth

train:
  epochs: 50
  save_dir: outputs
  save_every: 5  # 每5轮保存一次
```

### ad_dinov3.yaml（异常检测）
```yaml
data:
  train_dir: train_data/normal  # 仅正常样本
  resize: [512, 512]

model:
  memory_bank_size: 2000
  use_multi_scale: true

train:
  epochs: 30
  save_dir: outputs_ad
  use_anomaly_generation: true  # 异常合成
```

---

## 方法 1: 有监督分割（DINOv3 Supervised）

### 训练
```bash
python src/train_supervised.py --config configs/supervised.yaml
```
**模型保存位置**: `outputs/model_epoch_*.pth`（每 5 个 epoch 保存一次）

### 评估
```bash
python src/eval_supervised.py --config configs/supervised.yaml --checkpoint outputs/model_epoch_50.pth
```

### 测试（可视化检测效果）
```bash
# 单张图片
python src/test_supervised.py --config configs/supervised.yaml \
    --checkpoint outputs/model_epoch_50.pth \
    --input test_images/sample.png \
    --output test_results/supervised

# 批量测试（目录）
python src/test_supervised.py --config configs/supervised.yaml \
    --checkpoint outputs/model_epoch_50.pth \
    --input test_images/ \
    --output test_results/supervised
```
**输出**: 原图 + 预测分割图的可视化对比

---

## 方法 2: AD-DINOv3 异常检测

### 训练（仅需正常样本）
```bash
python src/train_ad.py --config configs/ad_dinov3.yaml
```
**模型保存位置**: `outputs_ad/ad_model_epoch_*.pth`（每 5 个 epoch 保存一次）

### 评估
```bash
python src/eval_ad.py --config configs/ad_dinov3.yaml --checkpoint outputs_ad/ad_model_epoch_30.pth
```

### 测试（可视化异常检测效果）
```bash
# 单张图片
python src/test_ad.py --config configs/ad_dinov3.yaml \
    --checkpoint outputs_ad/ad_model_epoch_30.pth \
    --input test_images/sample.png \
    --threshold 0.5 \
    --output test_results/ad_dinov3

# 批量测试
python src/test_ad.py --config configs/ad_dinov3.yaml \
    --checkpoint outputs_ad/ad_model_epoch_30.pth \
    --input test_images/ \
    --threshold 0.5 \
    --output test_results/ad_dinov3
```
**输出**: 原图 + 异常分数热图 + 二值化检测结果的可视化对比

---

## 对比两种方法

```bash
python src/compare_methods.py \
  --config_supervised configs/supervised.yaml \
  --config_ad configs/ad_dinov3.yaml \
  --checkpoint_supervised outputs/model_epoch_50.pth \
  --checkpoint_ad outputs_ad/ad_model_epoch_30.pth \
  --output comparison_results.json
```

**输出指标**：
- Supervised: mIoU（分割精度）
- AD-DINOv3: AUROC（检测准确率）、PRO（像素级准确率）

---

## 核心特性

### 有监督分割
- ✅ 精准的多类别像素级分割
- ✅ 清晰的类别边界
- ❌ 需要完整标注数据
- ❌ 新缺陷需重新训练

### AD-DINOv3 异常检测
- ✅ 仅需正常样本训练
- ✅ 零样本检测未知缺陷
- ✅ Teacher-Student 架构 + 特征记忆库
- ✅ CutPaste/Perlin 噪声增强
- ❌ 边界可能不如有监督精确

---

## 测试参数说明

### test_supervised.py 参数
- `--config`: 配置文件路径（必需）
- `--checkpoint`: 模型权重路径（必需）
- `--input`: 输入图片或目录（必需）
- `--output`: 输出目录（默认: `test_results/supervised`）
- `--show`: 显示可视化窗口
- `--device`: 设备选择（默认: `cuda`）

### test_ad.py 参数
- `--threshold`: 异常阈值 0-1（默认: 0.5）
  - 0.3-0.4：高灵敏度，检测更多疑似异常
  - 0.5：平衡（默认）
  - 0.6-0.7：低灵敏度，只标记明显异常
- 其他参数同上

---

## 项目结构

```
DINOv3_with_Φeat/
├── configs/
│   ├── supervised.yaml          # 有监督分割配置
│   └── ad_dinov3.yaml           # AD-DINOv3 配置
├── src/
│   ├── train_supervised.py      # 有监督训练
│   ├── eval_supervised.py       # 有监督评估
│   ├── test_supervised.py       # 有监督测试（可视化）
│   ├── train_ad.py              # AD-DINOv3 训练
│   ├── eval_ad.py               # AD-DINOv3 评估
│   ├── test_ad.py               # AD-DINOv3 测试（可视化）
│   ├── compare_methods.py       # 方法对比
│   ├── models/
│   │   ├── dinov3_backbone.py           # DINOv3 骨干（共用）
│   │   ├── supervised_segmentation.py   # 有监督分割模型
│   │   ├── anomaly_detector.py          # AD-DINOv3 检测器
│   │   └── memory_bank.py               # 特征记忆库
│   ├── datasets/
│   │   ├── supervised_dataset.py        # 有监督数据集
│   │   ├── anomaly_dataset.py           # 异常检测数据集
│   │   └── anomaly_generator.py         # 异常合成
│   └── utils/
│       ├── visualization.py             # 可视化工具
│       └── config.py, label_map.py
├── outputs/                     # 有监督模型输出
├── outputs_ad/                  # AD-DINOv3 模型输出
└── test_results/                # 测试可视化结果
```

---

## 方法选择指南

| 场景 | 推荐方法 | 理由 |
|------|---------|------|
| 已知缺陷类别 + 充足标注 | Supervised | 精准分割，类别清晰 |
| 未知缺陷类型 + 标注稀缺 | AD-DINOv3 | 零样本检测新缺陷 |
| 仅有正常样本 | AD-DINOv3 | 无需异常标注 |
| 需要精确边界 | Supervised | 像素级监督更准确 |

---

## 常见问题

**Q: 找不到模型文件？**  
A: 确认模型已训练保存，检查路径是否正确。

**Q: CUDA内存不足？**  
A: 使用 `--device cpu` 或减小配置文件中的 `data.resize` 分辨率。

**Q: 如何选择方法？**  
A: 有标注 → Supervised；无标注但有正常样本 → AD-DINOv3。

**Q: 如何调整检测灵敏度？**  
A: AD-DINOv3 修改 `--threshold` 参数（0.3高灵敏，0.7低灵敏）。
