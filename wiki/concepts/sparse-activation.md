---
type: concept
title: "稀疏激活"
date: 2026-06-01
updated: 2026-06-01
tags: [wiki, wiki/concept]
source_count: 2
confidence: low
domain_volatility: low
last_reviewed: 2026-06-01
aliases:
  - "稀疏激活"
  - "Sparse Activation"
  - "Conditional Computation"
  - "条件计算"
---

## Definition

稀疏激活（Sparse Activation）：网络中并非所有参数都参与每次前向传播，只有被"激活"的部分参数参与计算。在 MoE 语境中，指每次只激活少数专家网络。使参数量和计算量解耦，是扩展超大规模模型的关键技术。

## Key Points

- 与稠密模型（Dense）对比：稠密模型每次激活全部参数
- MoE 是最成功的稀疏激活实现：参数量可达 1.6T，实际计算量仅激活约 30B
- 挑战：负载均衡、通信开销（分布式训练中专家分布在不同 GPU）

## My Position

## Contradictions

## Sources

- [[moe-2017]]
- [[switch-transformer-2021]]

## Evolution Log

- 2026-06-01（2 sources）：初始创建
