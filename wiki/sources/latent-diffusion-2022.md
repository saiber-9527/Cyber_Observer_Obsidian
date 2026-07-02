---
type: source-summary
title: "High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)"
date: 2022-04-13
source_url: "https://arxiv.org/abs/2112.10752"
domain: arxiv.org
author: "Robin Rombach, Andreas Blattmann et al. (LMU Munich)"
tags: [wiki, wiki/source]
processed: true
raw_file: "raw/articles/ai-papers/14-StableDiffusion-扩散模型(2022).md"
raw_sha256: "38582c62f6d22832b8ebf056fd04e45b823cf0b912d4533a53fe772c216294b7"
last_verified: 2026-06-01
possibly_outdated: false
language: en
---

## 摘要（中文）

提出**潜空间扩散模型（LDM）**：将扩散过程从像素空间移至 VAE 压缩后的**潜空间**，大幅降低计算成本，同时引入交叉注意力机制实现文本条件控制。这是 Stable Diffusion 的理论基础。

## 核心要点

- **核心创新**：先用 VAE 将图像压缩到低维潜空间（512×512 → 64×64×4），再在潜空间运行扩散模型，计算量减少 4-16 倍
- **去噪过程**：U-Net 架构预测噪声，通过 T 步反向扩散生成图像
- **交叉注意力（Cross-Attention）**：文本 token 作为 K/V，图像特征作为 Q，实现文本控制
- **CLIP 文本编码器**：将文本提示编码为条件向量
- 与像素空间扩散模型相比推理速度快 10 倍以上，可在消费级 GPU 运行
- Stable Diffusion 的直接理论来源

## 提取概念

- [[diffusion-model]]
- [[latent-diffusion]]
- [[multimodal-model]]

## 我的笔记
