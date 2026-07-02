---
type: concept
title: "扩散模型"
date: 2026-06-01
updated: 2026-06-01
tags: [wiki, wiki/concept]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-06-01
aliases:
  - "扩散模型"
  - "Diffusion Model"
  - "DDPM"
  - "去噪扩散概率模型"
---

## Definition

扩散模型（Diffusion Model）：一类生成模型，通过逐步向数据添加噪声（正向过程）和学习逐步去噪（反向过程）来生成数据。核心思想是将复杂分布的生成问题分解为 T 步简单去噪步骤。

## Key Points

- **正向过程**：逐步向图像加高斯噪声，T 步后变为纯噪声
- **反向过程**：神经网络（U-Net）学习预测每步噪声，从纯噪声逐步恢复图像
- **条件生成**：通过 Classifier-Free Guidance 实现文本/类别控制
- **与 GAN 对比**：训练更稳定，无模式坍塌，但推理速度慢（需 T≈1000 步）
- **潜空间扩散（LDM）**：Stable Diffusion 的核心创新，将扩散过程移至压缩潜空间，大幅加速

## My Position

## Contradictions

## Sources

- [[latent-diffusion-2022]]

## Evolution Log

- 2026-06-01（1 source）：初始创建，来自 LDM/Stable Diffusion 论文
