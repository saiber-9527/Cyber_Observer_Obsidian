---
type: concept
title: "混合专家模型"
date: 2026-06-01
updated: 2026-06-01
tags: [wiki, wiki/concept]
source_count: 2
confidence: low
domain_volatility: medium
last_reviewed: 2026-06-01
aliases:
  - "混合专家模型"
  - "Mixture of Experts"
  - "MoE"
---

## Definition

混合专家模型（Mixture of Experts，MoE）：将模型的 FFN 层替换为多个并行的"专家"子网络，每次前向传播由门控网络选择性激活少数专家（稀疏激活），使参数量和计算量解耦——参数量可以极大，实际计算量维持恒定。

## Key Points

- **专家**：通常是独立的 FFN 层，数量从几十到数千不等
- **门控网络（Router）**：决定每个 token 路由到哪些专家，通常用 softmax 加 Top-k 选择
- **稀疏激活**：每次只激活 k 个专家（k=1 或 k=2），计算量≈单个专家的计算量
- **负载均衡**：需要辅助损失防止"专家坍塌"（所有 token 路由到同一专家）
- **代表模型**：Switch Transformer（1.6T参数）、Mixtral 8×7B、GPT-4（猜测）

## My Position

## Contradictions

- MoE 训练不稳定，需要精细调优；推理时需加载所有专家权重，对显存要求高

## Sources

- [[moe-2017]]
- [[switch-transformer-2021]]

## Evolution Log

- 2026-06-01（2 sources）：初始创建，来自 Shazeer 2017 MoE 和 Switch Transformer
