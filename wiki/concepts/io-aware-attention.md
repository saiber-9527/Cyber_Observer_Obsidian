---
type: concept
title: "IO感知注意力"
date: 2026-06-01
updated: 2026-06-01
tags: [wiki, wiki/concept]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-06-01
aliases:
  - "IO感知注意力"
  - "IO-Aware Attention"
  - "FlashAttention"
---

## Definition

IO感知注意力（IO-Aware Attention）：从 GPU 内存层次结构角度优化注意力计算的方法，核心洞见是注意力计算的实际瓶颈在 HBM（High Bandwidth Memory）读写次数，而非浮点运算量（FLOPS）。FlashAttention 是最成功的实现。

## Key Points

- **HBM vs SRAM**：HBM 带宽约 2TB/s，SRAM（片上缓存）带宽约 19TB/s，差距约 10 倍
- **标准注意力问题**：需将 n×n 注意力矩阵（O(n²) 显存）写回 HBM，IO 次数多
- **FlashAttention 解法**：分块计算（Tiling），在 SRAM 内完成整块注意力计算，大幅减少 HBM 读写
- **数值精确**：与标准注意力完全等价，无近似损失
- 已成为现代 LLM 训练的标准基础设施

## My Position

## Contradictions

## Sources

- [[flashattention-2022]]

## Evolution Log

- 2026-06-01（1 source）：初始创建，来自 FlashAttention
