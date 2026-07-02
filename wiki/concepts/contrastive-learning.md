---
type: concept
title: "对比学习"
date: 2026-06-01
updated: 2026-06-01
tags: [wiki, wiki/concept]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-06-01
aliases:
  - "对比学习"
  - "Contrastive Learning"
  - "InfoNCE"
---

## Definition

对比学习（Contrastive Learning）：通过最大化正样本对（语义相似）的相似度、最小化负样本对的相似度来学习表示。CLIP 中的对比目标是：最大化配对图文嵌入的余弦相似度，最小化批次内其他图文对的相似度（InfoNCE/CLIP Loss）。

## Key Points

- **正样本对**：（图像，对应文本描述）
- **负样本对**：批次内所有其他图文组合
- **大 batch size 关键**：更多负样本 → 更难的负例 → 更好的表示
- **应用**：CLIP（图文）、SimCLR（图像自监督）、MoCo 等
- CLIP 用 4 亿图文对训练，学到了极强的通用视觉-语言表示

## My Position

## Contradictions

## Sources

- [[clip-2021]]

## Evolution Log

- 2026-06-01（1 source）：初始创建，来自 CLIP 论文
