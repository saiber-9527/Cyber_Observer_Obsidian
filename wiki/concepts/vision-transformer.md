---
type: concept
title: "视觉Transformer"
date: 2026-06-01
updated: 2026-06-01
tags: [wiki, wiki/concept]
source_count: 2
confidence: low
domain_volatility: medium
last_reviewed: 2026-06-01
aliases:
  - "视觉Transformer"
  - "Vision Transformer"
  - "ViT"
---

## Definition

视觉Transformer（Vision Transformer，ViT）：将 NLP 中的 Transformer 架构直接应用于图像处理的模型。核心做法是把图像切成固定大小的 patch（如 16×16 像素），将每个 patch 线性投影为向量，作为 token 序列输入标准 Transformer 编码器。

## Key Points

- **图像 Patch 化**：图像 → patch 序列，打破 CNN 的局部卷积假设
- **需要大量预训练数据**：ViT 在小数据集上弱于 CNN，需大规模预训练（JFT-300M 等）
- **无归纳偏置**：Transformer 没有局部性、平移不变性等 CNN 内置先验，但更灵活
- **发展**：ViT → DeiT（数据高效）→ Swin Transformer（滑动窗口）→ ViT-L/H（扩展规模）
- CLIP 的图像编码器即采用 ViT 架构

## My Position

## Contradictions

- ViT vs CNN 在小数据场景下的效率争议仍然存在

## Sources

- [[vit-2020]]
- [[clip-2021]]

## Evolution Log

- 2026-06-01（2 sources）：初始创建，来自 ViT 和 CLIP 论文
