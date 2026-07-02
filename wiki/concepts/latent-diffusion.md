---
type: concept
title: "潜空间扩散"
date: 2026-06-01
updated: 2026-06-01
tags: [wiki, wiki/concept]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-06-01
aliases:
  - "潜空间扩散"
  - "Latent Diffusion"
  - "LDM"
---

## Definition

潜空间扩散（Latent Diffusion Model，LDM）：将扩散过程从高维像素空间移至 VAE 编码后的低维潜空间，大幅降低计算成本。Stable Diffusion 的直接理论基础。

## Key Points

- **VAE 压缩**：图像 512×512×3 → 潜码 64×64×4（约 48 倍压缩）
- 在潜空间运行扩散模型，推理速度比像素空间快 10 倍以上
- 可在消费级 GPU（8GB 显存）上生成 512×512 高质量图像
- **条件控制**：通过交叉注意力接入文本（CLIP 编码）、深度图、语义图等条件

## My Position

## Contradictions

## Sources

- [[latent-diffusion-2022]]

## Evolution Log

- 2026-06-01（1 source）：初始创建
