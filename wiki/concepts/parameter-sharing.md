---
type: concept
title: "参数共享"
date: 2026-06-01
updated: 2026-06-01
tags: [wiki, wiki/concept]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-06-01
aliases:
  - "参数共享"
  - "Parameter Sharing"
  - "Cross-layer Parameter Sharing"
  - "跨层参数共享"
---

## Definition

参数共享（Parameter Sharing）：多个计算单元使用相同的权重参数，从而在不降低表达能力的前提下大幅减少参数量。ALBERT 中的跨层参数共享指所有 Transformer 层共享同一套注意力头权重和 FFN 权重。

## Key Points

- ALBERT 跨层共享使参数量从 334M（BERT-LARGE）降至 18M（但计算量不变）
- 实验证明共享注意力头权重比共享 FFN 层对性能影响更小
- 权衡：参数量↓ 但计算量不变；模型容量可能受限
- 与知识蒸馏的对比：参数共享是架构设计；蒸馏是训练策略

## My Position

## Contradictions

- 参数共享限制了模型每层学习不同特征的能力，在复杂任务上可能劣于非共享版本

## Sources

- [[albert-2019]]

## Evolution Log

- 2026-06-01（1 source）：初始创建，来自 ALBERT
