---
type: concept
title: "线性注意力"
date: 2026-06-01
updated: 2026-06-01
tags: [wiki, wiki/concept]
source_count: 3
confidence: low
domain_volatility: high
last_reviewed: 2026-06-01
aliases:
  - "线性注意力"
  - "Linear Attention"
  - "次二次注意力"
  - "Sub-quadratic Attention"
---

## Definition

线性注意力（Linear Attention）：广义上指将标准 O(n²) 自注意力的计算/内存复杂度降低到 O(n) 或更低的注意力变体。包括局部窗口注意力（Longformer）、线性化核注意力、状态空间模型（Mamba）、保持机制（RetNet）等不同路线。

## Key Points

- **动机**：标准注意力 O(n²) 在长序列（n > 4K）时显存和计算爆炸
- **局部窗口**（Longformer）：每个 token 只关注附近 w 个邻居，O(n·w)
- **SSM/RNN 路线**（Mamba/RetNet）：递归表示，O(n) 计算，O(1) 推理内存
- **近似注意力**（Linformer 等）：低秩近似，但精度有损失
- FlashAttention 解决的是 IO 效率问题，数值上仍是精确的 O(n²) 注意力

## My Position

## Contradictions

- 各种线性注意力变体在不同任务上各有优劣，尚无统一最优方案

## Sources

- [[longformer-2020]]
- [[mamba-2024]]
- [[retnet-2023]]

## Evolution Log

- 2026-06-01（3 sources）：初始创建
