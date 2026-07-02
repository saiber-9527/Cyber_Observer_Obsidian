---
type: concept
title: "缩放定律"
date: 2026-06-01
updated: 2026-06-01
tags: [wiki, wiki/concept]
source_count: 4
confidence: low
domain_volatility: medium
last_reviewed: 2026-06-01
aliases:
  - "缩放定律"
  - "Scaling Law"
  - "规模效应"
---

## Definition

缩放定律（Scaling Law）：语言模型的性能（损失）随参数量（N）、训练数据量（D）、计算量（C）的增大而幂律下降，三者之间存在最优配比关系。由 OpenAI（Kaplan et al. 2020）系统化，后由 DeepMind（Chinchilla）修正。

## Key Points

- **幂律关系**：Loss ∝ N^{-α} ∝ D^{-β}，α≈0.076，β≈0.095（Kaplan et al.）
- **计算最优配比（Chinchilla 定律）**：给定算力预算，模型参数量与训练 token 数应大致相等（各翻倍）
- **涌现能力**：某些任务在特定规模阈值前性能接近随机，超过阈值后突然涌现
- GPT-2 → GPT-3 → GPT-4 的规模跨越均基于缩放定律的预测
- 缩放定律的适用边界正在被质疑（数据质量、架构创新的影响）

## My Position

## Contradictions

- 缩放定律预测性能会持续提升，但"涌现能力"的突变现象与平滑幂律矛盾

## Sources

- [[gpt-2-2019]]
- [[gpt-3-2020]]
- [[switch-transformer-2021]]
- [[gpt-4-2023]]

## Evolution Log

- 2026-06-01（4 sources）：初始创建
