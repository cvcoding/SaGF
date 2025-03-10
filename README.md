# SaGF
# SaGF: Saliency-Aware Graph Fusion for Endometrial Cancer Diagnosis using Ultrasound Imaging

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)

Official implementation of **Saliency-Aware Graph Fusion (SaGF)** - a novel Graph-Transformer framework for robust endometrial cancer diagnosis and myometrial invasion depth assessment in ultrasound imaging. Achieves human-expert-surpassing performance with computational efficiency and interpretability.

![SaGF Framework Overview](docs/framework.png)

## Key Features
- **Dual-Resolution Processing**: Disentangles tissue textures (low-rank) and pathological saliency (sparse) via Triplet Attention
- **Saliency-Guided Sparse Attention**: Reduces computational complexity from \(O(N^2)\) to \(O(Nk)\) via top-\(k\) node filtering
- **Self-Supervised Learning**: Dual-view contrastive learning pipeline addressing data scarcity and long-tail distributions
- **Clinically Interpretable**: t-SNE visualization of learned features with 99.01% binary classification accuracy
- **Computationally Efficient**: 11.71ms per inference on standard GPU hardware

## Installation
```bash
# Clone repository
git clone https://github.com/cvcoding/SaGF.git
cd SaGF

# Create conda environment
conda create -n sagf python=3.8
conda activate sagf

# Install dependencies
pip install -r requirements.txt



Dataset
The model was trained on 31,910 ultrasound images from 7,882 patients across multicenter cohorts. Due to patient privacy regulations, please contact the corresponding author for dataset access requests.

Preprocessing steps:

Adaptive histogram equalization

Anatomical ROI detection

Dual-resolution normalization (512×512 and 256×256)


Performance Highlights
Metric	Binary Classification	Ternary Classification
Accuracy	99.01%	86.69%
Sensitivity	0.995	0.912
Specificity	0.931	0.894
AUC	0.991	0.889
Inference Speed	11.71ms/image	13.02ms/image
Comparative Performance (vs Human Experts):

27.2% higher macro AUC in ternary classification

19.1% improvement in sensitivity

12.9% higher specificity
