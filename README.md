# Person ReID Baseline

A strong baseline for person re-identification (ReID) implemented in PyTorch.

Based on the paper: **Bag of Tricks and A Strong Baseline for Deep Person Re-identification** (CVPRW 2019)

Paper: https://arxiv.org/abs/1903.07071

## Features

- **Backbone**: ResNet50-IBN-a (domain generalization friendly)
- **Training Tricks**:
  - Warmup learning rate
  - Random erasing augmentation
  - Random horizontal flip
  - Label smoothing
  - BNNeck (Batch Normalization neck)
  - Triplet loss + Cross-entropy loss
- **Multi-GPU training support**
- **Standard evaluation**: mAP and CMC Rank-1/5/10

## Performance (on Market1501)

| Model | Rank-1 | mAP |
|-------|--------|-----|
| ResNet50-IBN-a + BoT | 94.1% | 85.3% |

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
python train.py --data_dir /path/to/market1501 --output_dir ./logs
```

### Inference / Demo

```bash
python demo.py --query_image query.jpg --gallery_dir ./gallery --checkpoint model.pth
```

## Dataset Preparation

Download Market1501 dataset and organize as follows:

```
market1501/
├── bounding_box_train/
├── bounding_box_test/
└── query/
```

## Project Structure

```
├── model.py          # ReID model architecture
├── train.py          # Training script
├── demo.py           # Inference demo
├── utils.py          # Utility functions
└── requirements.txt # Dependencies
```

## Citation

```bibtex
@inproceedings{luo2019bag,
  title={Bag of tricks and a strong baseline for deep person re-identification},
  author={Luo, Hao and Gu, Youzhi and Liao, Xingyu and Lai, Shenqi and Jiang, Wei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={0--0},
  year={2019}
}
```
